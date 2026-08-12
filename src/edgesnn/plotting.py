"""EDA plotting helpers.

All functions operate on DataFrames in the shape produced by
`edgesnn.data_utils` (column 0 = label, columns 1+ = series), so plotting is
decoupled from file loading. Kept separate from `data_utils` so that modeling
code can import the data helpers without pulling in seaborn/matplotlib.

Because `UCRDataset.to_frame()` returns that same shape, these functions also
plot the *encoded* data with no changes — which is what makes a raw-vs-
aggregated figure pair a matter of calling the same function twice.

Every function takes `save_path=`; the written figure (not any intermediate
array) is the artifact that belongs in the thesis, so point it at
`reports/figures/`.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Aesthetic style matching the academic thesis look.
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# Publication-grade raster; matches what LaTeX expects for a full-width figure.
SAVE_DPI = 300


def _finish(fig, save_path, show=True):
    """Save (if asked) then show/close a finished figure."""
    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight")
        print(f"[plot] saved {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _named(labels, class_names):
    """Map integer-encoded labels to readable names, leaving others untouched."""
    if class_names is None:
        return labels

    def lookup(value):
        try:
            return class_names[int(float(value))]
        except (ValueError, IndexError):
            return value

    return labels.map(lookup)


def _class_order(*label_series):
    """Deterministic bar order shared by every panel of a figure.

    `sns.countplot` without `order=` falls back to order of first appearance for
    string labels, which makes bar order depend on row order and lets the Train
    and Test panels disagree. Sort numerically when the labels are numeric
    (so class 10 follows class 9, not class 1), lexically otherwise, and build it
    from the union of all splits so a class missing from one panel still holds
    its slot.
    """
    values = set()
    for s in label_series:
        values.update(s.unique())

    def key(v):
        try:
            return (0, float(v), "")
        except ValueError:
            return (1, 0.0, str(v))

    return sorted(values, key=key)


def plot_class_distribution(df_train, df_test, dataset_name, save_path=None,
                            show=True, class_names=None, sharey=True):
    """Plot instances-per-class for the Train and Test splits side by side.

    Helps identify severe class imbalances. Pass a `UCRDataset.to_frame()` to
    see the distribution *after* label aggregation, with
    `class_names=ds.class_names` so the bars read "normal (1)" rather than "0".

    `sharey=False` is worth using when the splits differ wildly in size (as in
    ECG5000's 500/4500), where a shared axis flattens the smaller split into
    unreadability.
    """
    # Cast to string to safely handle both int and string labels
    train_raw = df_train.iloc[:, 0].astype(str)
    test_raw = df_test.iloc[:, 0].astype(str)

    # Order is derived from the *encoded* labels, before naming, so bars sit in
    # class-index order ("abnormal (-1)" = class 0 first) rather than being
    # sorted by the alphabet of their display names.
    order = _named(pd.Series(_class_order(train_raw, test_raw)), class_names).tolist()
    train_labels = _named(train_raw, class_names)
    test_labels = _named(test_raw, class_names)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=sharey)
    fig.suptitle(f"Class Distribution: {dataset_name}", fontsize=16, fontweight='bold')

    sns.countplot(x=train_labels, hue=train_labels, order=order, hue_order=order,
                  legend=False, ax=axes[0], palette="viridis")
    axes[0].set_title(f"Train Set (N={len(train_labels)})")
    axes[0].set_xlabel("Class Label")
    axes[0].set_ylabel("Count")

    sns.countplot(x=test_labels, hue=test_labels, order=order, hue_order=order,
                  legend=False, ax=axes[1], palette="viridis")
    axes[1].set_title(f"Test Set (N={len(test_labels)})")
    axes[1].set_xlabel("Class Label")
    if not sharey:
        axes[1].set_ylabel("Count")

    # Rotate tick labels so long class names (e.g. InsectSound's "Stigma_female")
    # stay readable instead of overlapping.
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('right')

    plt.tight_layout()
    _finish(fig, save_path, show)


def plot_mean_std_series(df, dataset_name, split_name="Train", save_path=None,
                         show=True, class_names=None):
    """Plot the per-class mean time series with a ±1 standard-deviation band.

    When plotting a `UCRDataset.to_frame()`, pass `class_names=ds.class_names`
    for readable titles, and `to_frame(denormalize=True)` if the figure is about
    waveform shape rather than the effect of normalization.
    """
    labels = df.iloc[:, 0].astype(str).values
    unique_classes = np.unique(labels)
    time_series_data = df.iloc[:, 1:].values.astype(float)

    num_classes = len(unique_classes)
    cols = min(3, num_classes)
    rows = int(np.ceil(num_classes / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    fig.suptitle(f"Mean & Std Dev of Time Series per Class: {dataset_name} ({split_name})",
                 fontsize=16, fontweight='bold')

    axes = axes.flatten()

    for idx, cls in enumerate(unique_classes):
        class_data = time_series_data[labels == cls]

        mean_series = np.mean(class_data, axis=0)
        std_series = np.std(class_data, axis=0)
        time_steps = np.arange(len(mean_series))

        # Encoded frames carry integer labels; map them back to readable names.
        title_cls = cls
        if class_names is not None:
            try:
                title_cls = class_names[int(float(cls))]
            except (ValueError, IndexError):
                pass

        ax = axes[idx]
        ax.plot(time_steps, mean_series, label=f"Class {title_cls} Mean", color="blue", linewidth=2)
        ax.fill_between(time_steps,
                        mean_series - std_series,
                        mean_series + std_series,
                        color="blue", alpha=0.2, label="±1 Std Dev")

        ax.set_title(f"Class {title_cls} (N={len(class_data)})")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right")

    # Hide any unused subplots
    for idx in range(num_classes, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    _finish(fig, save_path, show)


def plot_separability(f_stats, dataset_name, k_marks=None, save_path=None, show=True):
    """Plot the per-timestep ANOVA F-statistic from `temporal_analysis`.

    `k_marks` (absolute timesteps) draws the truncation grid over the curve, so
    the figure reads as "cutting at 20% of T discards this much of the
    discriminative mass". Takes the plain `(T,)` array — the statistic is
    computed in `temporal_analysis.py`, keeping analysis out of the plotter.
    """
    f_stats = np.asarray(f_stats, dtype=float)
    time_steps = np.arange(len(f_stats))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_steps, f_stats, color="darkred", linewidth=1.5)
    ax.fill_between(time_steps, 0, f_stats, color="darkred", alpha=0.15)

    for k in k_marks or []:
        ax.axvline(k - 1, color="grey", linestyle="--", linewidth=1, alpha=0.8)
        ax.text(k - 1, ax.get_ylim()[1], f" k={k}", va="top", ha="left",
                fontsize=9, color="grey")

    ax.set_title(f"Per-Timestep Class Separability: {dataset_name}",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("ANOVA F-statistic")
    ax.set_xlim(0, len(f_stats) - 1)

    plt.tight_layout()
    _finish(fig, save_path, show)


def plot_prefix_curve(curves, save_path=None, show=True, x="fraction"):
    """Plot cross-validated prefix-classifier Macro-F1 against the cutoff `k`.

    `curves` is either one `prefix_classifier_curve` DataFrame or a
    `{dataset_name: DataFrame}` mapping, which puts all three datasets on one
    axis — the comparison the truncation chapter actually argues from. Error
    bands are ±1 std over the cross-validation folds. Use `x="fraction"` to
    compare across datasets of differing T, `x="k"` for a single dataset.
    """
    if isinstance(curves, pd.DataFrame):
        curves = {"": curves}

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = sns.color_palette("viridis", max(len(curves), 2))

    for colour, (name, curve) in zip(palette, curves.items()):
        xs = curve[x].to_numpy()
        mean = curve["macro_f1_mean"].to_numpy()
        std = curve["macro_f1_std"].to_numpy()
        ax.plot(xs, mean, marker="o", color=colour, linewidth=2, label=name or None)
        ax.fill_between(xs, mean - std, mean + std, color=colour, alpha=0.2)

    ax.set_title("Prefix-Classifier Performance vs. Truncation Point",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Fraction of sequence observed" if x == "fraction" else "Truncation point k")
    ax.set_ylabel("Macro-F1 (cross-validated, train split)")
    if x == "fraction":
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    if any(curves):
        ax.legend(title="Dataset")

    plt.tight_layout()
    _finish(fig, save_path, show)
