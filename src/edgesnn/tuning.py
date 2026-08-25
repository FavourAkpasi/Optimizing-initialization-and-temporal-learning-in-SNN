"""Grid search over the nuisance hyperparameters, run once per model per dataset.

The hyperparameters of this study fall into three categories, and the boundary
between them is what makes the results interpretable:

1. **Independent variables** -- the truncation point `k` and the initialization
   method. These are never tuned. They are the quantities under study, and their
   full sweep is the result; selecting the best `k` would discard the finding.
2. **Matched constants** -- `N_res = 256`, `H = 128`, `p_c = 0.1`, the weighted
   loss and the stopping protocol. Fixed by design, because the parameter
   matching between the LSTM and the reservoir depends on them.
3. **Nuisance parameters** -- learning rate, batch size, and for the spiking
   models the neuron and surrogate settings. These are searched here.

Category 3 is tuned **per model but not per condition**. Each architecture gets
its own optimum, which is the fair choice: forcing one learning rate across an
LSTM and a spiking network would penalise whichever architecture happens to
dislike it. But the search runs once, at `k = T` with the Gaussian
initialization, and the winning configuration is then frozen across the entire
truncation and initialization sweep. Re-tuning at each `k` would fold tuning
effects into the accuracy-versus-`k` curve, and re-tuning per initializer would
erase the very differences the study exists to measure -- a cheap generator that
needs a different learning rate is a finding, not a nuisance to be optimised away.

Selection is on validation Macro-F1 from a split carved out of TRAIN, and the
identical split is reused for every configuration so that a difference in score
cannot be a difference in partition.
"""

import itertools

import pandas as pd
import torch
from tqdm.auto import tqdm

from edgesnn.training import (MAX_EPOCHS, MIN_EPOCHS, PATIENCE, SELECTION_FOLDS,
                              cross_validate)

__all__ = ["BATCH_SIZES", "DEFAULT_GRIDS", "DEFAULT_STAGES", "expand_grid",
           "grid_search", "staged_search"]

# Batch size is FIXED PER DATASET, not searched, and held constant across every
# model and every configuration.
#
# It is not a nuisance parameter in the sense the others are. On MPS the per-batch
# cost is flat at ~0.13 s regardless of batch size, because the k sequential
# timestep iterations dominate and not the batch dimension -- so on InsectSound a
# batch of 256 runs an epoch in 12.7 s against 50.3 s at 64. But the same choice
# on ECG5000, whose training portion is 400 samples, would leave a single gradient
# update per epoch. The parameter therefore trades throughput against update count
# and is determined by dataset size, which differs 50x across these three sets.
#
# Searching it per model would let it silently absorb differences that belong to
# the architecture, and it interacts with the learning rate besides -- a larger
# batch gives fewer, lower-variance updates and generally wants a larger lr. Fixed
# per dataset it confounds nothing, and it is stated as a throughput decision.
BATCH_SIZES = {
    "ECG5000": 64,
    "Wafer": 64,
    "InsectSound": 256,
}

# The grids actually used in this thesis. Learning rate and surrogate family are
# crossed deliberately: the four surrogates return 1, beta/2, 1 and beta/4 at the
# threshold, so at beta = 10 their gradient magnitudes differ by up to 5x. Tuned
# separately, a family would be rejected for having the wrong gradient scale
# rather than the wrong shape, which is not the question being asked.
DEFAULT_GRIDS = {
    "snn": {
        "lr": [1e-4, 3e-4, 1e-3, 3e-3],
        "leak_rate": [0.8, 0.9, 0.95],
        "threshold": [0.5, 1.0],
        "surrogate": ["fast_sigmoid", "arctan", "triangular", "sigmoid"],
    },
    # The LSM never propagates a gradient through a spike, so the surrogate
    # family cannot affect it and is omitted rather than searched pointlessly.
    # Its neuron settings still matter: they shape the frozen reservoir's firing.
    "lsm": {
        "lr": [1e-4, 3e-4, 1e-3, 3e-3],
        "leak_rate": [0.8, 0.9, 0.95],
        "threshold": [0.5, 1.0],
    },
    # The LSTM has exactly ONE tunable parameter, so its grid is deliberately the
    # densest of the three: ten values at roughly 2.5 points per decade across
    # three decades, against the four values each spiking model gets on its lr
    # axis. The reasoning is search-budget parity. The LSM searches 24
    # configurations and the SNN 16; giving the LSTM only 4 would mean the
    # baseline received a quarter of the tuning effort of the model it exists to
    # challenge, and "the baseline was undertrained" is the first objection to a
    # favourable efficiency result.
    #
    # The range was first extended downward to 1e-5 and upward to 1e-2 because
    # the original [1e-4, 3e-3] window was bounded by failure at both ends on
    # Wafer: 1e-4 collapsed to the all-majority solution (0.485 Macro-F1, best
    # epoch 1.8) while 3e-3 was unstable. A window whose every interior point is
    # a compromise between not-learning and diverging is too narrow to have
    # contained an optimum.
    #
    # Extended again to 5e-2 (2026-08-25), UPWARD ONLY. The requirement is that
    # the selected value be INTERIOR to the grid, since a winner sitting on the
    # boundary leaves it unknown whether the optimum lies outside. After the full
    # re-run, ECG5000 selected 2e-3 and Wafer 5e-4, both interior with worse
    # neighbours on either side; only InsectSound selected the largest value in
    # the grid.
    #
    # The lower end is deliberately NOT extended. It is not truncating the
    # search: at 1e-5 on Wafer the LSTM scores 0.4745 with a mean best epoch of
    # 1.0, which is a clean "does not learn" failure rather than a boundary that
    # might be hiding something. Extending below it would buy nothing and would
    # cost search time on all three datasets to re-confirm a known result.
    "lstm": {
        "lr": [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4,
               1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2],
    },
}

# Staged (coordinate-descent) alternative to the exhaustive grids above.
#
# The full SNN grid is 4 x 3 x 2 x 4 = 96 configurations. On ECG5000 and Wafer
# that costs minutes, but on InsectSound at k = 600 a single configuration runs
# several minutes even on MPS, putting the exhaustive search well into the tens
# of hours for the SNN alone -- before any of the actual sweeps have started.
#
# Each stage searches one group, fixes its winner, and passes it to the next, so
# the SNN costs 16 + 6 = 22 configurations instead of 96.
#
# The GROUPING is the design. Parameters that interact must share a stage, since
# a stage cannot revisit what an earlier one froze; parameters that do not may be
# separated. Learning rate and surrogate are crossed for exactly that reason (see
# the gradient-scale note above): tuned apart, a family would be rejected for its
# gradient magnitude rather than its shape. Treating the neuron parameters as
# separable from them is the assumption this trade buys.
#
# The same staged protocol runs on all three datasets. A search protocol that
# varies per dataset is one more thing to defend for no scientific gain.
DEFAULT_STAGES = {
    "snn": [
        {"lr": [1e-4, 3e-4, 1e-3, 3e-3], "surrogate": ["fast_sigmoid", "arctan",
                                                       "triangular", "sigmoid"]},
        {"leak_rate": [0.8, 0.9, 0.95], "threshold": [0.5, 1.0]},
    ],
    "lsm": [
        {"lr": [1e-4, 3e-4, 1e-3, 3e-3]},
        {"leak_rate": [0.8, 0.9, 0.95], "threshold": [0.5, 1.0]},
    ],
    "lstm": [
        {"lr": [1e-4, 3e-4, 1e-3, 3e-3]},
    ],
}


def expand_grid(grid):
    """Cartesian product of a `{name: [values]}` grid, as a list of dicts."""
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def grid_search(model_factory, train_dataset, grid, *, max_epochs=MAX_EPOCHS,
                min_epochs=MIN_EPOCHS, patience=PATIENCE, n_splits=None, seed=0,
                device="auto", progress=True, desc=None, train_kwargs=None):
    """Exhaustively score every configuration in `grid`, best first.

    Exhaustive rather than staged: measured head-to-head on ECG5000, the staged
    coordinate-descent alternative missed the optimum by 0.0102 Macro-F1 and its
    winner ranked only tenth of ninety-six, so the separability assumption it
    rests on does not hold here. The full product is affordable (roughly fifteen
    hours across all three datasets) and carries no assumption to defend.

    `model_factory(**config)` must build a fresh model for each configuration --
    fresh, because reusing one would carry the previous configuration's weights
    into the next and score the sequence rather than the settings.

    `n_splits` sets the selection protocol: `None` reads it from
    `SELECTION_FOLDS` by dataset name, 1 uses a single held-out split, and k > 1
    averages over stratified k-fold. Configuration keys are routed automatically,
    those the factory understands going to the model and `lr`/`batch_size` to the
    trainer. The returned frame carries the fold standard deviation alongside the
    mean, so a difference between configurations can be read against the noise
    rather than trusted blindly.
    """
    configs = expand_grid(grid)
    train_kwargs = dict(train_kwargs or {})

    if n_splits is None:
        n_splits = SELECTION_FOLDS.get(getattr(train_dataset, "dataset_name", None), 1)

    trainer_keys = {"lr", "batch_size"}
    rows = []

    outer = tqdm(configs, desc=desc or "grid search", unit="cfg",
                 disable=not progress, leave=True)

    for i, config in enumerate(outer):
        model_kwargs = {k: v for k, v in config.items() if k not in trainer_keys}
        fit_kwargs = {k: v for k, v in config.items() if k in trainer_keys}

        outer.set_postfix({k: str(v) for k, v in config.items()})

        # The factory is wrapped rather than called, because cross_validate must
        # build a fresh model per fold -- and must seed immediately before doing
        # so. The reservoir is pinned by the seeded initializer, but the readout
        # is an ordinary nn.Linear drawn from the global RNG at construction
        # time; without that seeding, part of every score difference would be
        # readout luck rather than the hyperparameter under test.
        score = cross_validate(
            lambda: model_factory(**model_kwargs), train_dataset,
            n_splits=n_splits, seed=seed, device=device,
            max_epochs=max_epochs, min_epochs=min_epochs, patience=patience,
            progress=progress, desc=f"  cfg {i + 1}/{len(configs)}",
            **fit_kwargs, **train_kwargs,
        )

        rows.append({
            **config,
            "val_macro_f1": score["mean"],
            "val_macro_f1_std": score["std"],
            "mean_best_epoch": score["mean_best_epoch"],
            # False means the epoch ceiling stopped this configuration rather
            # than the patience rule -- i.e. it was still improving when the
            # budget ran out, and its score is a lower bound on what it would
            # have reached. A search containing such a row is partly ranking
            # configurations by speed of convergence rather than by quality.
            "stopped_early": score["all_stopped_early"],
            "n_splits": n_splits,
        })

        best_so_far = max(r["val_macro_f1"] for r in rows)
        outer.set_description(f"{desc or 'grid search'} (best {best_so_far:.4f})")

    outer.close()

    return (pd.DataFrame(rows)
            .sort_values("val_macro_f1", ascending=False)
            .reset_index(drop=True))


def staged_search(model_factory, train_dataset, stages, *, base=None, **kwargs):
    """Coordinate-descent search: one group at a time, winner carried forward.

    Returns `(best_config, results)` where `results` is the concatenation of every
    stage's table with a `stage` column, so the search itself is reportable --
    including the case where a later stage failed to improve on the defaults,
    which is worth knowing before claiming a parameter mattered.

    Cheaper than the exhaustive product by roughly an order of magnitude, at the
    cost of assuming the groups are separable. Keep any genuinely interacting
    parameters inside a single stage.
    """
    config = dict(base or {})
    tables = []

    for stage_index, stage in enumerate(stages, start=1):
        # Fold the winners so far in as constants for this stage.
        stage_grid = {k: [v] for k, v in config.items() if k not in stage}
        stage_grid.update(stage)

        table = grid_search(model_factory, train_dataset, stage_grid,
                            desc=f"stage {stage_index}/{len(stages)}", **kwargs)
        table.insert(0, "stage", stage_index)
        tables.append(table)

        # `.item()` unwraps the numpy scalars pandas produces, so the returned
        # config holds plain Python values that can be logged, serialised to
        # JSON, and passed straight back into a constructor.
        winner = table.iloc[0]
        config = {k: (winner[k].item() if hasattr(winner[k], "item") else winner[k])
                  for k in stage_grid}

    return config, pd.concat(tables, ignore_index=True)
