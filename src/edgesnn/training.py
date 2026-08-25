"""Truncated surrogate-gradient training loop, shared by all three models.

One trainer serves the spiking reservoir, the LSM and the LSTM. That is a
requirement rather than a convenience: if each model were fitted by its own loop,
a difference in results could always be attributed to a difference in the
optimisation protocol instead of to the architecture. Every model here sees the
same loss, the same split, the same stopping rule and the same schedule.

The policy this implements, decided before any results were produced:

- **Class-weighted cross-entropy**, with weights `N / (C n_c)` estimated on the
  training portion alone. On a balanced dataset those weights are exactly 1.0, so
  the rule is inert on InsectSound, nearly inert on ECG5000 (1.4:1) and only
  binds on Wafer (9.3:1). One rule, no per-dataset special case to defend.
- **A stratified validation split carved out of TRAIN**, used for early stopping
  and for hyperparameter selection. The test split is touched exactly once, at
  the very end. Selecting a stopping epoch against test would leak it into every
  number reported afterwards.
- **Early stopping on validation Macro-F1, not on loss.** On an imbalanced task
  the all-majority solution is a low-loss plateau: a Wafer run sitting at
  Macro-F1 0.47 (precisely the majority solution) looks converged by loss. A
  `min_epochs` floor guards the same failure, since a model still on the plateau
  must not be scored as though it had converged.

Truncation is a property of the model (`model.k`), not of the trainer, so the
same loop trains a full-sequence and a truncated model without modification.
"""

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

__all__ = [
    "SELECTION_FOLDS",
    "GRAD_CLIP_NORM",
    "MAX_EPOCHS",
    "MIN_EPOCHS",
    "PATIENCE",
    "resolve_device",
    "class_weights",
    "stratified_split",
    "TrainResult",
    "train_model",
    "evaluate_model",
    "cross_validate",
]

# How many folds to use when SELECTING hyperparameters, per dataset.
#
# A single 20% split of ECG5000 leaves 100 validation samples, and Macro-F1 over
# 100 samples is quantised so coarsely that 96 configurations produced only 37
# distinct scores with a seven-way tie at the maximum -- the "best" configuration
# was then whichever tied row sorted first. Averaging over 5 folds uses every
# training sample for validation exactly once and resolves the ties.
#
# InsectSound needs none of this: a single split already yields 5,000 validation
# samples, and at ~13 h for one exhaustive sweep, five folds would be indefensible
# for a signal that is already stable. The rule is set by validation-set size, the
# same reasoning that fixes batch size per dataset.
SELECTION_FOLDS = {
    "ECG5000": 5,
    "Wafer": 5,
    "InsectSound": 1,
}

# Maximum global gradient norm, applied to every model before every step.
#
# Added 2026-08-20 after the LSTM baseline was found to be failing on Wafer for
# optimization reasons rather than capability ones. Per seed, at k = 91 of 152,
# ALL FIVE seeds landed between 0.601 and 0.685 Macro-F1, while the same model at
# k = 61 reached 0.966 -- a baseline that does worse on more of the sequence is
# measuring a broken fit, not a limitation. Notebook 03 had already shown the
# symptom: every LSTM learning rate on Wafer scored between 0.485 and 0.680 with
# a fold standard deviation as high as 0.165.
#
# The important point is one of FAIRNESS, not of tuning. Running all three models
# through one trainer with no clipping sounds neutral and is not: the surrogate
# gradient bounds the spiking models' gradient by construction (each surrogate
# returns at most 1, beta/2, 1 or beta/4 at threshold), whereas an LSTM
# backpropagating through 152 timesteps has no such bound. Omitting clipping
# therefore penalises exactly the baseline the thesis is measured against.
# Clipping is standard practice for BPTT (Pascanu, Mikolov and Bengio, 2013).
#
# 1.0 is the conventional default. It is deliberately NOT tuned: it is a matched
# constant shared by every model and condition, so it confounds nothing, and
# tuning it per model would reintroduce the asymmetry it exists to remove.
GRAD_CLIP_NORM = 1.0

# The epoch budget, shared by every model, dataset and experimental stage.
#
# `MAX_EPOCHS` is a SAFETY CEILING, not a training length. It is set high enough
# that no run is expected to reach it, so that training always terminates on the
# early-stopping criterion below and never on the budget. That distinction
# matters: a ceiling that binds is a hyperparameter chosen by accident, and it
# binds hardest on precisely the configuration that is still improving -- which
# is the one that least deserves to be cut off. Under the previous ceiling of 80
# the InsectSound LSM was still improving when it ran out of epochs, so the
# search there was partly ranking configurations by how quickly they converged
# rather than by how well.
#
# Raising it is close to free, because patience does the stopping. On ECG5000 and
# Wafer the LSM search peaked at a mean best epoch of 18.6 and 25.0, with no
# configuration above 52.4 out of 80; those runs stopped because they had stopped
# improving, so a higher ceiling changes their result not at all and costs only
# the epochs a still-improving run actually uses. Nor can a larger budget inflate
# the reported figure: `train_model` restores the best-validation weights, so
# extra epochs can only find a better peak or leave the existing one alone.
#
# ONE budget for all three models, for the same reason gradient clipping is one
# constant: a budget that binds on one architecture and not another silently
# penalises whichever model converges more slowly. Note the side effect, which is
# correct behaviour rather than a flaw -- a longer budget favours smaller
# learning rates, which need more epochs to arrive at the same place, so the
# selected learning rate may fall relative to a search run under a tight ceiling.
#
# `PATIENCE` is the number of consecutive epochs allowed to pass without a new
# best validation Macro-F1 before the run is abandoned; any new best resets the
# count to zero. `MIN_EPOCHS` is a floor that outranks it, because on Wafer a
# model can sit at the all-majority solution with a falling loss and a flat
# Macro-F1 for several epochs, which patience alone would misread as convergence.
MAX_EPOCHS = 200
MIN_EPOCHS = 15
PATIENCE = 12


def resolve_device(device="auto"):
    """Pick a device, preferring Apple's Metal backend when it is available.

    MPS is roughly 2.6x faster than CPU for the spiking loop and 10x for the
    LSTM on the development machine, which is the difference between a two-day
    and a week-long sweep. Spike counts were verified bit-identical to CPU and
    logits agree to ~1e-6, so the speedup costs nothing in fidelity.
    """
    if device != "auto":
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def class_weights(y, num_classes):
    """Inverse-frequency weights `N / (C n_c)`, normalised to mean 1.

    Returns all-ones for a balanced problem, which is what makes it safe to
    apply unconditionally to every dataset.
    """
    counts = torch.bincount(torch.as_tensor(y), minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)  # a class absent from this split
    return counts.sum() / (num_classes * counts)


def stratified_split(y, val_fraction=0.2, seed=0):
    """Stratified (train, val) index arrays carved out of the training split.

    Stratified because Wafer's minority class is 9.7% of training: a uniform
    split could hand the validation set too few positives for Macro-F1 to be a
    stable stopping signal.
    """
    y = np.asarray(y)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction,
                                      random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(y)), y))
    return train_idx, val_idx


def _with_feature_dim(x):
    """Datasets store `(N, T)`; the models expect `(N, T, features)`."""
    return x.unsqueeze(-1) if x.dim() == 2 else x


@dataclass
class TrainResult:
    """Everything needed to report a run without re-running it."""

    best_val_f1: float = 0.0
    best_epoch: int = 0
    epochs_run: int = 0
    stopped_early: bool = False
    device: str = "cpu"
    history: list = field(default_factory=list)

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame(self.history)


@torch.no_grad()
def evaluate_model(model, dataset, batch_size=256, device="auto", indices=None):
    """Macro-F1, accuracy and mean spike count over a dataset or a subset.

    The spike count is returned because it is the quantity the energy model
    consumes; capturing it here avoids a second pass over the data purely to
    count spikes. It is `None` for the LSTM, whose cost is analytic.
    """
    device = resolve_device(device) if not isinstance(device, torch.device) else device
    model = model.to(device).eval()

    data = dataset if indices is None else Subset(dataset, indices)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    preds, targets, spike_total, n_seen = [], [], 0.0, 0
    is_spiking = hasattr(model, "lif_layer")

    for xb, yb in loader:
        xb = _with_feature_dim(xb).to(device)
        logits, aux = model(xb)
        preds.append(logits.argmax(dim=1).cpu())
        targets.append(yb)
        if is_spiking:
            spike_total += aux.sum().item()
            n_seen += xb.shape[0]

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    return {
        "macro_f1": f1_score(targets, preds, average="macro"),
        "accuracy": float((preds == targets).mean()),
        "y_true": targets,
        "y_pred": preds,
        "mean_spikes": (spike_total / n_seen) if is_spiking and n_seen else None,
    }


def train_model(model, train_dataset, *, batch_size=64, lr=1e-3,
                max_epochs=MAX_EPOCHS, min_epochs=MIN_EPOCHS, patience=PATIENCE,
                val_fraction=0.2, weighted_loss=True,
                device="auto", seed=0, progress=True, desc=None, split=None,
                grad_clip=GRAD_CLIP_NORM):
    """Fit `model` on `train_dataset`, returning the best-validation weights.

    The model is mutated in place and left holding the parameters from its best
    validation epoch, not from its last one -- otherwise a run that peaked and
    then overfitted would be reported at its worst.

    Training stops when `patience` consecutive epochs have passed without a new
    best validation Macro-F1 (any new best resets that counter), subject to the
    `min_epochs` floor. `max_epochs` is a safety ceiling that no run is expected
    to reach; `TrainResult.stopped_early` records whether it was reached, which
    is what makes the claim "every run terminated on the stopping criterion"
    checkable rather than assumed.

    `grad_clip` rescales the gradient to that maximum global norm before each
    optimizer step. See `GRAD_CLIP_NORM`: it is applied identically to all three
    models, and omitting it does NOT treat them equally, because only the LSTM
    has an unbounded gradient. Pass `None` to disable.

    `split` accepts a precomputed `(train_idx, val_idx)` pair so that every
    configuration in a grid search sees the identical split; comparing
    configurations across different splits would confound the hyperparameter
    with the partition.

    REPRODUCIBILITY: `seed` here fixes the batch order and any stochasticity
    during the fit, but it CANNOT fix the model's initial weights -- the model
    already exists by the time this function runs. The reservoir is pinned by
    `initialize_reservoir(..., seed=)`, but the readout is an ordinary
    `nn.Linear` drawn from the global RNG when the model was constructed. Call
    `torch.manual_seed(...)` immediately before building the model, or two
    "identical" runs will differ by ~0.03 Macro-F1 on a small validation split.
    `tuning.grid_search` does this for you.
    """
    device = resolve_device(device)
    torch.manual_seed(seed)

    y = train_dataset.y.numpy() if torch.is_tensor(train_dataset.y) else np.asarray(train_dataset.y)
    train_idx, val_idx = split if split is not None else stratified_split(
        y, val_fraction=val_fraction, seed=seed)

    num_classes = train_dataset.num_classes
    # Weights come from the TRAINING PORTION only: the validation labels are
    # part of the selection procedure, not of the fit.
    weight = class_weights(y[train_idx], num_classes).to(device) if weighted_loss else None
    criterion = nn.CrossEntropyLoss(weight=weight)

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(Subset(train_dataset, train_idx), batch_size=batch_size,
                        shuffle=True, generator=generator, drop_last=False)

    model = model.to(device)
    # Only parameters that require a gradient: for the LSM this is the readout
    # alone, and handing frozen tensors to Adam would allocate optimizer state
    # for weights that never move.
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)

    result = TrainResult(device=str(device))
    best_state, epochs_since_best = None, 0

    epoch_bar = tqdm(range(max_epochs), desc=desc or "training", unit="ep",
                     disable=not progress, leave=False)

    for epoch in epoch_bar:
        model.train()
        running_loss, n_batches = 0.0, 0

        for xb, yb in loader:
            xb = _with_feature_dim(xb).to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(trainable, grad_clip)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        val = evaluate_model(model, train_dataset, device=device, indices=val_idx)
        train_loss = running_loss / max(n_batches, 1)

        result.history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_macro_f1": val["macro_f1"],
            "val_accuracy": val["accuracy"],
            "mean_spikes": val["mean_spikes"],
        })

        if val["macro_f1"] > result.best_val_f1:
            result.best_val_f1 = val["macro_f1"]
            result.best_epoch = epoch + 1
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        epoch_bar.set_postfix({
            "loss": f"{train_loss:.4f}",
            "val_F1": f"{val['macro_f1']:.4f}",
            "best": f"{result.best_val_f1:.4f}@{result.best_epoch}",
            "wait": f"{epochs_since_best}/{patience}",
        })

        result.epochs_run = epoch + 1

        # The floor matters: on Wafer a model can sit at the all-majority
        # solution for several epochs with a falling loss and a flat Macro-F1,
        # which patience alone would read as convergence.
        if epoch + 1 >= min_epochs and epochs_since_best >= patience:
            result.stopped_early = True
            break

    epoch_bar.close()

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return result


def cross_validate(model_factory, train_dataset, *, n_splits=5, seed=0,
                   progress=True, desc=None, **fit_kwargs):
    """Stratified k-fold selection score for one configuration.

    Returns `{"mean", "std", "folds", "mean_best_epoch", "all_stopped_early"}`.
    The last is False if any fold exhausted the epoch ceiling instead of
    triggering the patience rule, which is the signal that the budget -- rather
    than convergence -- decided when that configuration stopped learning.

    The mean is the
    quantity to select on; the standard deviation says whether a difference
    between two configurations is worth believing, which a single split cannot
    tell you at all.

    A fresh model is built per fold -- via `model_factory()`, seeded immediately
    before construction so the readout is identical across configurations -- since
    reusing one would carry fold 1's weights into fold 2 and measure the sequence
    rather than the configuration.
    """
    y = train_dataset.y.numpy() if torch.is_tensor(train_dataset.y) else np.asarray(train_dataset.y)

    if n_splits <= 1:
        torch.manual_seed(seed)
        result = train_model(model_factory(), train_dataset, seed=seed,
                             progress=progress, desc=desc, **fit_kwargs)
        return {"mean": result.best_val_f1, "std": 0.0,
                "folds": [result.best_val_f1], "mean_best_epoch": result.best_epoch,
                "all_stopped_early": result.stopped_early}

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores, epochs, stopped = [], [], []

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(np.zeros(len(y)), y), start=1):
        torch.manual_seed(seed)
        result = train_model(model_factory(), train_dataset, split=(tr_idx, va_idx),
                             seed=seed, progress=progress,
                             desc=f"{desc or 'cv'} fold {fold}/{n_splits}", **fit_kwargs)
        scores.append(result.best_val_f1)
        epochs.append(result.best_epoch)
        stopped.append(result.stopped_early)

    scores = np.asarray(scores)
    return {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "folds": scores.tolist(),
        "mean_best_epoch": float(np.mean(epochs)),
        "all_stopped_early": bool(all(stopped)),
    }
