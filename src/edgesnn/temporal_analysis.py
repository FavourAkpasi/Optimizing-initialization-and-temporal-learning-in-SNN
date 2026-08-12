"""Model-free analysis of *where in time* a series carries its class information.

This module answers a question the truncation study cannot answer about itself:
when a spiking reservoir truncated at timestep `k` scores well, is that because
the network handles truncation gracefully, or because the data was already
separable that early? Both routines here are deliberately **independent of the
SNN** — one is a plain statistic, the other a linear probe — so their curves form
a data-side reference against which the model-side k-sweep is read.

Two views, deliberately different in kind:

- `per_timestep_f_statistic` — a one-way ANOVA F per timestep. It is *marginal*
  and *instantaneous*: it asks how well timestep `t` alone separates the classes,
  ignoring every other timestep. Cheap, assumption-light, and it localizes the
  discriminative region of the waveform.
- `prefix_classifier_curve` — the cumulative view. A linear classifier is fit on
  the prefix `x[:, :k]` for each `k` and scored by cross-validated Macro-F1. This
  captures information carried *jointly* across timesteps, which the marginal
  F-statistic cannot see, and it yields an upper reference for "how much of the
  label is knowable from the first k samples".

Both run on the **TRAIN split only** and the prefix curve is cross-validated
inside that split. Choosing `k` — or even eyeballing it — from test data would
leak the test set into a headline hyperparameter of the thesis.

Nothing here plots. Curves are returned as arrays/DataFrames and rendered by
`edgesnn.plotting`, keeping the modeling/plotting split intact.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

__all__ = [
    "FRACTION_GRID",
    "k_grid",
    "per_timestep_f_statistic",
    "prefix_classifier_curve",
    "earliest_k_within",
]

# The truncation grid of the thesis, expressed as fractions of the sequence
# length T rather than absolute timesteps, so a single grid is comparable across
# T in {140, 152, 600}.
FRACTION_GRID = (0.2, 0.4, 0.6, 0.8, 1.0)


def _as_numpy(x):
    """Accept torch tensors and numpy arrays alike, without importing torch."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def k_grid(seq_len, fractions=FRACTION_GRID):
    """Absolute timestep cutoffs for the given fractions of `seq_len`.

    Rounds to whole timesteps, clamps to at least 1, and de-duplicates — short
    sequences with a fine fraction grid can otherwise map two fractions onto the
    same `k` and silently double the work.
    """
    ks = sorted({min(seq_len, max(1, int(round(f * seq_len)))) for f in fractions})
    return ks


def per_timestep_f_statistic(x, y):
    """One-way ANOVA F-statistic per timestep. Returns `(T,)` float64.

    `F = (SSB / (C-1)) / (SSW / (N-C))`, computed in one vectorized pass over the
    classes: between-group scatter of the per-class means against the grand mean,
    over within-group scatter. A high F at timestep `t` means the class means are
    far apart there relative to the within-class spread.

    Reported as the raw statistic, not a p-value: with T up to 600 the p-values
    would demand a multiple-comparison correction to be interpretable, and the
    curve is read for its *shape* (where the discriminative mass sits) rather
    than for per-timestep significance. Timesteps with zero within-class variance
    yield `inf` and are returned as such rather than silently clipped.
    """
    x = _as_numpy(x).astype(np.float64, copy=False)
    y = _as_numpy(y)
    if x.ndim != 2:
        raise ValueError(f"x must be (N, T); got shape {x.shape}")

    classes = np.unique(y)
    n_samples, _ = x.shape
    if len(classes) < 2:
        raise ValueError("The F-statistic needs at least two classes.")
    if n_samples <= len(classes):
        raise ValueError("Need more samples than classes to form a within-group variance.")

    grand_mean = x.mean(axis=0)
    ssb = np.zeros(x.shape[1], dtype=np.float64)
    ssw = np.zeros(x.shape[1], dtype=np.float64)
    for c in classes:
        xc = x[y == c]
        class_mean = xc.mean(axis=0)
        ssb += len(xc) * (class_mean - grand_mean) ** 2
        ssw += ((xc - class_mean) ** 2).sum(axis=0)

    df_between = len(classes) - 1
    df_within = n_samples - len(classes)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (ssb / df_between) / (ssw / df_within)


def _stratified_subsample(x, y, max_samples, seed):
    """Class-proportional subsample, used only to keep the probe tractable."""
    if max_samples is None or len(y) <= max_samples:
        return x, y
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        take = max(1, int(round(len(idx) * max_samples / len(y))))
        keep.append(rng.choice(idx, size=min(take, len(idx)), replace=False))
    keep = np.sort(np.concatenate(keep))
    return x[keep], y[keep]


def prefix_classifier_curve(x, y, fractions=FRACTION_GRID, ks=None, n_splits=5,
                            seed=0, max_iter=200, max_samples=None, verbose=False):
    """Cross-validated Macro-F1 of a linear probe fit on the prefix `x[:, :k]`.

    Returns a DataFrame with one row per `k`: `k`, `fraction`, `n_features`,
    `macro_f1_mean`, `macro_f1_std`, and the per-fold scores in `folds`.

    The probe is multinomial logistic regression behind a `StandardScaler`, fit
    on raw prefix samples as features. A *linear* model is the point: it is a
    lower bound on decodable information that no architectural cleverness could
    be credited for, so a prefix curve that is already flat at `k = 0.2T` says
    the *data* permits early classification. Any gap between this curve and the
    SNN's k-sweep is then attributable to the model, which is precisely the
    comparison the truncation chapter needs.

    Macro-F1 (not accuracy) because Wafer and ECG5000 are heavily imbalanced;
    stratified k-fold keeps the minority class present in every fold. All folds
    are drawn from the TRAIN split — see the module docstring.

    `max_samples` stratifies a subsample before fitting, for InsectSound's 25k
    x 600: it affects only this diagnostic, never the trained models.
    """
    x = _as_numpy(x).astype(np.float64, copy=False)
    y = _as_numpy(y)
    x, y = _stratified_subsample(x, y, max_samples, seed)

    seq_len = x.shape[1]
    ks = list(ks) if ks is not None else k_grid(seq_len, fractions)

    # A fold must be able to hold every class; fewer members than folds in the
    # rarest class would make StratifiedKFold warn and produce empty classes.
    min_class = int(np.bincount(y.astype(int)).min())
    n_splits = max(2, min(n_splits, min_class))

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for k in ks:
        prefix = x[:, :k]
        scores = []
        for train_idx, val_idx in splitter.split(prefix, y):
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=max_iter, random_state=seed),
            )
            clf.fit(prefix[train_idx], y[train_idx])
            scores.append(
                f1_score(y[val_idx], clf.predict(prefix[val_idx]), average="macro")
            )
        scores = np.asarray(scores)
        if verbose:
            print(f"  k={k:>4} ({k / seq_len:5.1%})  macro-F1 "
                  f"{scores.mean():.4f} +/- {scores.std():.4f}")
        rows.append({
            "k": k,
            "fraction": k / seq_len,
            "n_features": k,
            "macro_f1_mean": scores.mean(),
            "macro_f1_std": scores.std(),
            "folds": scores,
        })
    return pd.DataFrame(rows)


def earliest_k_within(curve, tolerance=0.01, column="macro_f1_mean"):
    """Smallest `k` whose score is within `tolerance` of the curve's best.

    The one-number summary of a prefix curve: "the first 40% of the sequence
    already buys 99% of what the full sequence buys". Returns `(k, fraction,
    score)`.
    """
    best = curve[column].max()
    hit = curve[curve[column] >= best - tolerance]
    row = hit.iloc[0]
    return int(row["k"]), float(row["fraction"]), float(row[column])
