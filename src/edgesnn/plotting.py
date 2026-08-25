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


def plot_lif_dynamics(potential, spikes, current, threshold=1.0, reset_potential=0.0,
                      leak_rate=0.9, save_path=None, show=True):
    """Illustrate one LIF neuron's membrane trace, for the background chapter.

    Takes plain arrays rather than a model, so this module stays free of
    modeling imports; `scripts/make_lif_figure.py` runs the real `LIFCell` and
    passes its output here, which is what keeps the figure honest — it shows the
    dynamics the experiments actually use rather than a redrawn idealisation.

    The annotation that matters is the reset: the trace drops by exactly one
    threshold and keeps whatever charge lay above it, so a neuron driven well
    past threshold starts the next step already part-way back.
    """
    potential = np.asarray(potential, dtype=float)
    spikes = np.asarray(spikes, dtype=float)
    current = np.asarray(current, dtype=float)
    steps = np.arange(len(potential))

    # `potential` is V[t], recorded AFTER the reset, so on its own it never
    # reaches the threshold and the figure would appear to show a neuron that
    # fires without ever arriving. The pre-reset value is recovered exactly from
    # V[t] = H[t] - threshold * S[t], and a spiking step is drawn as two points
    # at the same t, giving the vertical drop that the subtraction performs.
    pre_reset = potential + threshold * spikes
    xs, ys = [], []
    for t in steps:
        if spikes[t] > 0:
            xs.extend([t, t])
            ys.extend([pre_reset[t], potential[t]])
        else:
            xs.append(t)
            ys.append(potential[t])

    fig, (ax_i, ax_v) = plt.subplots(
        2, 1, figsize=(10, 5.5), sharex=True,
        gridspec_kw={"height_ratios": [1, 3], "hspace": 0.12})

    ax_i.bar(steps, current, width=0.35, color="steelblue")
    ax_i.set_ylabel("Input\n$X[t]$")
    ax_i.grid(axis="y", alpha=0.3)

    ax_v.plot(xs, ys, color="black", linewidth=2, marker="o", markersize=3.5,
              label="Membrane potential")
    ax_v.axhline(threshold, color="green", linestyle="--", linewidth=1.5,
                 label=r"Threshold $\vartheta$")
    ax_v.axhline(reset_potential, color="firebrick", linestyle=":", linewidth=1.5,
                 label="$V_{reset}$")

    fired = steps[spikes > 0]
    for t in fired:
        ax_v.plot([t], [pre_reset[t]], marker="*", color="darkorange",
                  markersize=15, zorder=5)
        # The retained charge: from V_reset up to the post-reset potential.
        ax_v.annotate("", xy=(t + 0.42, potential[t]), xytext=(t + 0.42, reset_potential),
                      arrowprops=dict(arrowstyle="<->", color="darkorange", lw=1.4))

    if len(fired):
        ax_v.plot([], [], marker="*", color="darkorange", linestyle="none",
                  markersize=13, label=r"Spike emitted ($H[t] \geq \vartheta$)")
        t0 = fired[-1]
        ax_v.annotate(
            "residual $H[t]-\\vartheta$ is\ncarried into the next step",
            xy=(t0 + 0.42, potential[t0] / 2), xytext=(t0 + 1.1, threshold * 0.72),
            fontsize=9, color="darkorange",
            arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.2))

    ax_v.set_xlabel("Time step $t$")
    ax_v.set_ylabel("Membrane potential")
    ax_v.set_xticks(steps[::2])          # timesteps are integers, not 2.5, 7.5, ...
    ax_v.set_ylim(-0.08, max(pre_reset.max(), threshold) * 1.28)
    ax_v.grid(alpha=0.3)
    ax_v.legend(loc="upper left", fontsize=9, framealpha=0.95)

    equations = (r"LIF dynamics:  $H[t]=\beta(V[t-1]-V_{reset})+X[t]$,   "
                 r"$S[t]=\Theta(H[t]-\vartheta)$,   "
                 r"$V[t]=H[t]-\vartheta\,S[t]$")
    ax_i.set_title(equations + "\n"
                   + rf"($\beta={leak_rate}$, $\vartheta={threshold}$)",
                   fontsize=11, fontweight="bold")

    _finish(fig, save_path, show)


def plot_confusion_matrix(matrix, class_names=None, title=None, normalize=True,
                          save_path=None, show=True):
    """Confusion matrix heatmap, as promised by Methods Section 3.7.1.

    Row-normalised by default, which is the reading that matters under class
    imbalance: with 665 faults against 5,499 normals, raw counts make the
    minority row almost invisible, while row fractions show what proportion of
    each true class was recovered. Raw counts are annotated underneath so
    nothing is hidden by the normalisation.
    """
    matrix = np.asarray(matrix)
    n = matrix.shape[0]
    names = list(class_names) if class_names is not None else [str(i) for i in range(n)]

    with np.errstate(invalid="ignore", divide="ignore"):
        shown = matrix / matrix.sum(axis=1, keepdims=True) if normalize else matrix
    shown = np.nan_to_num(shown)

    annotations = np.empty(matrix.shape, dtype=object)
    for i in range(n):
        for j in range(n):
            annotations[i, j] = (f"{shown[i, j]:.2f}\n({matrix[i, j]:,})"
                                 if normalize else f"{matrix[i, j]:,}")

    fig, ax = plt.subplots(figsize=(1.5 + 1.1 * n, 1.2 + 0.95 * n))
    sns.heatmap(shown, annot=annotations, fmt="", cmap="Blues", square=True,
                cbar_kws={"label": "fraction of true class" if normalize else "count"},
                xticklabels=names, yticklabels=names, ax=ax,
                vmin=0, vmax=1 if normalize else None, annot_kws={"fontsize": 9})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or "Confusion matrix", fontsize=13, fontweight="bold")

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


def plot_prefix_curve(curves, save_path=None, show=True, x="fraction",
                      value="macro_f1_mean", spread="macro_f1_std", title=None,
                      ylabel=None, legend_title="Dataset", ax=None):
    """Plot a metric against the truncation point, with a ±1 spread band.

    `curves` is one DataFrame or a `{label: DataFrame}` mapping. It serves both
    the prefix-classifier curves of the temporal analysis and the
    accuracy-versus-`k` curves of the truncation study, because those two are
    meant to be read against each other — the point of the comparison is whether
    a model's curve sits above the model-free probe at the same cutoff. Passing
    an existing `ax` overlays them on one axis.

    Use `x="fraction"` to compare across datasets of differing `T`, `x="k"` for
    a single dataset.
    """
    if isinstance(curves, pd.DataFrame):
        curves = {"": curves}

    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    palette = sns.color_palette("viridis", max(len(curves), 2))

    for colour, (name, curve) in zip(palette, curves.items()):
        curve = curve.sort_values(x)
        xs = curve[x].to_numpy()
        mean = curve[value].to_numpy()
        ax.plot(xs, mean, marker="o", color=colour, linewidth=2, label=name or None)
        if spread and spread in curve:
            band = curve[spread].fillna(0).to_numpy()
            ax.fill_between(xs, mean - band, mean + band, color=colour, alpha=0.2)

    ax.set_title(title or "Prefix-Classifier Performance vs. Truncation Point",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Fraction of sequence observed" if x == "fraction" else "Truncation point $k$")
    ax.set_ylabel(ylabel or "Macro-F1 (cross-validated, train split)")
    if x == "fraction":
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    if any(curves):
        ax.legend(title=legend_title)

    if own_figure:
        plt.tight_layout()
        _finish(fig, save_path, show)
    return ax


def plot_initializer_effect(frames, baseline="gaussian", value="macro_f1",
                            group="init_method", save_path=None, show=True,
                            title=None, ylabel=None, order=None, sharex=False):
    """Per-initializer performance relative to a baseline, with seed spread.

    `frames` is one DataFrame of PER-SEED result rows or a `{dataset: DataFrame}`
    mapping, in the shape written by `experiments.run_grid` — one row per
    (initializer, seed). The aggregation happens here rather than in the caller
    so that the error bar and the point can never come from different groupings.

    Each initializer is drawn as its mean advantage over `baseline` with an error
    bar of ±1 standard deviation ACROSS SEEDS, and the baseline itself sits on a
    reference line at zero. That layout is the point of the figure. The
    differences between cheap generators in this study are smaller than the
    spread across seeds, and a table of five numbers invites a reader to rank
    them regardless; drawn with error bars, the overlap is visible before any
    ordering is. A method whose interval crosses zero is not distinguishable
    from the baseline, and two methods whose intervals overlap are not
    distinguishable from each other.

    The spread is across seeds, which for a sparse reservoir is the meaningful
    replicate even for the deterministic sequences: Sobol and Halton produce a
    bit-identical dense draw at every seed, but the connectivity mask, the
    validation split, the batch order and the readout are all seeded, so their
    realised networks differ and their spread is real rather than an artefact.

    `sharex` puts every panel on one horizontal scale. Left independent (the
    default) each panel resolves its own differences, but the panels are then
    NOT comparable by eye: a deficit of 0.15 Macro-F1 and one of 0.04 occupy the
    same width. Shared, the panels are honestly comparable and the datasets with
    small effects correctly look small. Choose per figure and say which in the
    caption.
    """
    if isinstance(frames, pd.DataFrame):
        frames = {"": frames}

    panels = len(frames)
    fig, axes = plt.subplots(1, panels, figsize=(4.6 * panels, 4.6),
                             squeeze=False, sharex=sharex)

    for ax, (name, frame) in zip(axes[0], frames.items()):
        stats = (frame.groupby(group)[value]
                 .agg(mean="mean", sd="std", n="count")
                 .fillna({"sd": 0.0}))
        if baseline not in stats.index:
            raise ValueError(f"baseline {baseline!r} absent from column {group!r}")

        reference = stats.loc[baseline, "mean"]
        # The baseline is the reference line, so plotting it as a point too would
        # place a marker at exactly zero that means something different from the
        # others -- it is a definition, not a measurement.
        methods = [m for m in (order or stats.index) if m != baseline]
        stats = stats.loc[methods]

        y = np.arange(len(methods))
        ax.axvline(0.0, color="0.35", linewidth=1.4, zorder=1)
        ax.errorbar(stats["mean"] - reference, y, xerr=stats["sd"], fmt="o",
                    color=sns.color_palette("viridis", 5)[1], markersize=7,
                    capsize=4, linewidth=1.6, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(methods)
        ax.invert_yaxis()
        ax.set_title(f"{name}\n{baseline} = {reference:.4f}" if name
                     else f"{baseline} = {reference:.4f}", fontsize=12)
        ax.set_xlabel(ylabel or f"Δ {value} vs. {baseline}")
        ax.grid(axis="y", visible=False)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    _finish(fig, save_path, show)
    return fig
