"""Experiment orchestration: model construction, run bookkeeping, result storage.

The notebooks in `notebooks/` are meant to read as the argument of the thesis,
not as plumbing, so the repeated parts of every experiment live here: building a
model of a given kind with a given initializer, running one condition end to end,
and recording the outcome so that a sweep measured in days survives a kernel
restart.

That last point is not incidental. A full InsectSound sweep runs for hours, and a
notebook that loses everything when the kernel dies is a notebook nobody dares
re-run. Every result is appended to a CSV keyed by the experimental condition, and
`run_grid` skips conditions already present, so an interrupted sweep resumes where
it stopped and a finished one re-executes in seconds.
"""

import json
import os

import pandas as pd
import torch

from edgesnn.evaluation_metrics import evaluate_efficiency
from edgesnn.initializers import generate_sparse_mask, initialize_reservoir, is_deterministic
from edgesnn.models import LiquidStateMachine, LSTMBaseline, TruncatedSpikingReservoir
from edgesnn.training import train_model
from edgesnn.tuning import BATCH_SIZES

__all__ = [
    "RESULTS_DIR",
    "INIT_METHODS",
    "N_RES",
    "LSTM_HIDDEN",
    "CONNECTIVITY",
    "FRACTIONS",
    "build_model",
    "seeds_for",
    "k_values",
    "ensure_results_dir",
    "load_results",
    "append_result",
    "run_condition",
    "run_grid",
    "save_config",
    "load_config",
]

RESULTS_DIR = "../results"

# The five initializers of the study, Gaussian first because it is the baseline
# every other method is measured against.
INIT_METHODS = ("gaussian", "lcg", "lfsr", "sobol", "halton")

# Settled in Methods 3.3 and 3.6; repeated here so a notebook never hard-codes them.
N_RES = 256
LSTM_HIDDEN = 128
CONNECTIVITY = 0.1
FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)


def build_model(kind, *, num_classes, k, init_method="gaussian", seed=0,
                n_res=N_RES, hidden=LSTM_HIDDEN, connectivity=CONNECTIVITY,
                match_variance=True, input_size=1, **neuron_kwargs):
    """Construct one model, seeded and initialized, ready to train.

    The generator is seeded *before* construction rather than before training.
    The reservoir is pinned by the seeded initializer, but the readout is an
    ordinary linear layer drawn from the ambient global RNG at construction time;
    seeding late leaves it arbitrary, which was measured to move validation
    Macro-F1 by as much as 0.06 between otherwise identical runs.

    `kind` is one of `snn`, `lsm`, `lstm`. The LSTM ignores `init_method`, having
    no reservoir to initialize; it keeps PyTorch's default, which is the point of
    a baseline.
    """
    torch.manual_seed(seed)

    if kind == "lstm":
        return LSTMBaseline(input_size, hidden, num_classes, truncation_k=k)

    if kind not in ("snn", "lsm"):
        raise ValueError(f"Unknown model kind {kind!r}; expected snn, lsm or lstm.")

    cls = TruncatedSpikingReservoir if kind == "snn" else LiquidStateMachine
    model = cls(input_size, n_res, num_classes, truncation_k=k, **neuron_kwargs)

    w_in = initialize_reservoir((n_res, input_size), init_method, fan_in=input_size,
                                seed=seed, match_variance=match_variance)
    w_res = initialize_reservoir((n_res, n_res), init_method, fan_in=n_res,
                                 connectivity=connectivity, seed=seed,
                                 match_variance=match_variance)
    # The same seed reproduces the mask that `initialize_reservoir` already
    # applied, so the pattern pinned into the model is the pattern in the weights.
    mask = generate_sparse_mask((n_res, n_res), connectivity, seed=seed)
    model.load_custom_initialization(w_in, w_res, res_mask=mask)
    return model


def seeds_for(init_method, n_seeds=5):
    """Seeds to run for an initializer.

    Sobol and Halton are deterministic constructions: the n-th point is a fixed
    function of n, so repeating them would produce bit-identical weights and a
    standard deviation of exactly zero. Running them once and reporting no error
    bar is the honest treatment; five identical runs would manufacture the
    appearance of a replication that never happened.
    """
    return [0] if is_deterministic(init_method) else list(range(n_seeds))


def k_values(seq_len, fractions=FRACTIONS):
    """Absolute truncation windows for a sequence length (Methods Table 3.4)."""
    return sorted({min(seq_len, max(1, int(round(f * seq_len)))) for f in fractions})


def ensure_results_dir():
    """Create the results directory. Call once before writing tables directly."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def _path(name):
    return os.path.join(RESULTS_DIR, f"{name}.csv")


def load_results(name):
    """Rows recorded so far for an experiment, or an empty frame."""
    path = _path(name)
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def append_result(name, row):
    """Add one result row, rewriting the file so that columns stay aligned.

    The whole frame is rewritten rather than appended to in text mode. Different
    model kinds report different fields -- the spiking models carry firing rates
    and synaptic operations that the LSTM has no analogue for -- and a plain CSV
    append writes the new row's values positionally under the *existing* header,
    silently filing an LSTM's operation count under the column named for
    something else. Rewriting lets pandas align by name and fill the gaps with
    NaN, which is the honest representation of a field that does not apply.

    The file is still written after every condition, so an interrupted sweep
    keeps everything it had finished.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    existing = load_results(name)
    frame = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    frame.to_csv(_path(name), index=False)


def _same_value(stored, wanted):
    """Compare a stored cell against a wanted value, tolerating type drift.

    A CSV round-trip turns the integer 28 into the float 28.0, so a string
    comparison would judge a completed condition unfinished and re-run the whole
    sweep. Numeric values are therefore compared numerically and everything else
    as text.
    """
    try:
        return float(stored) == float(wanted)
    except (TypeError, ValueError):
        return str(stored) == str(wanted)


def _already_done(existing, condition):
    if existing.empty:
        return False
    if any(column not in existing.columns for column in condition):
        return False
    match = pd.Series(True, index=existing.index)
    for column, value in condition.items():
        match &= existing[column].map(lambda stored: _same_value(stored, value))
    return bool(match.any())


def run_condition(kind, train_ds, test_ds, *, k, init_method="gaussian", seed=0,
                  config=None, max_epochs=80, min_epochs=15, patience=12,
                  device="auto", progress=False, **build_kwargs):
    """Train one model on one condition and evaluate it on the test split.

    Returns a flat dictionary combining the condition, the training outcome and
    every metric of Methods Section 3.7 — one row of a results table.

    `config` carries the tuned nuisance hyperparameters (learning rate, neuron
    settings, surrogate). They are selected once and passed in unchanged, never
    re-tuned per condition: re-tuning at each k would fold tuning effects into
    the accuracy-versus-k curve, and re-tuning per initializer would erase the
    very differences this study measures.
    """
    config = dict(config or {})
    lr = config.pop("lr", 1e-3)
    batch_size = config.pop("batch_size",
                            BATCH_SIZES.get(train_ds.dataset_name, 64))
    # Neuron settings apply to the spiking models only; the LSTM has no such knobs.
    neuron_kwargs = config if kind != "lstm" else {}

    model = build_model(kind, num_classes=train_ds.num_classes, k=k,
                        init_method=init_method, seed=seed,
                        **neuron_kwargs, **build_kwargs)

    fit = train_model(model, train_ds, lr=lr, batch_size=batch_size,
                      max_epochs=max_epochs, min_epochs=min_epochs,
                      patience=patience, seed=seed, device=device,
                      progress=progress)

    metrics = evaluate_efficiency(model, test_ds, k=k, device=device)
    metrics.pop("confusion_matrix", None)      # kept out of the flat CSV
    metrics.pop("per_class_f1", None)
    metrics.pop("support", None)

    return {
        "dataset": train_ds.dataset_name,
        "model": kind,
        "init_method": init_method if kind != "lstm" else "default",
        "seed": seed,
        "lr": lr,
        "batch_size": batch_size,
        **{f"cfg_{key}": value for key, value in neuron_kwargs.items()},
        "val_macro_f1": fit.best_val_f1,
        "best_epoch": fit.best_epoch,
        "epochs_run": fit.epochs_run,
        "stopped_early": fit.stopped_early,
        **metrics,
    }


def run_grid(name, conditions, train_ds, test_ds, *, key_fields=None, **run_kwargs):
    """Run every condition not already recorded, appending each result as it lands.

    `conditions` is an iterable of dicts accepted by `run_condition`. Conditions
    already present in the stored results are skipped, so re-running a finished
    notebook is a no-op and an interrupted one resumes.
    """
    from tqdm.auto import tqdm

    key_fields = key_fields or ("model", "init_method", "seed", "k")
    existing = load_results(name)
    conditions = list(conditions)
    pending = []
    for condition in conditions:
        # `run_condition` records the model kind under `model` and gives the LSTM
        # the initializer name `default`, so the lookup key has to be built the
        # same way or nothing would ever match and every sweep would restart.
        recorded = dict(condition)
        recorded["model"] = condition["kind"]
        if condition["kind"] == "lstm":
            recorded["init_method"] = "default"
        key = {"dataset": train_ds.dataset_name,
               **{f: recorded.get(f, "default") for f in key_fields}}
        if not _already_done(existing, key):
            pending.append(condition)

    done = len(conditions) - len(pending)
    print(f"{name}: {len(pending)} conditions to run"
          + (f" ({done} already recorded)" if done else ""))

    for condition in tqdm(pending, desc=name, unit="run"):
        row = run_condition(train_ds=train_ds, test_ds=test_ds,
                            **condition, **run_kwargs)
        append_result(name, row)

    return load_results(name)


def save_config(name, config):
    """Persist a tuned hyperparameter configuration as JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"{name}.json"), "w") as handle:
        json.dump(config, handle, indent=2)


def load_config(name):
    """Read back a tuned configuration, or `None` if the search has not been run."""
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)
