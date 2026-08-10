"""EDA plotting helpers.

All functions operate on DataFrames in the shape produced by
`edgesnn.data_utils` (column 0 = label, columns 1+ = series), so plotting is
decoupled from file loading. Kept separate from `data_utils` so that modeling
code can import the data helpers without pulling in seaborn/matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Aesthetic style matching the academic thesis look.
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def plot_class_distribution(df_train, df_test, dataset_name):
    """Plot instances-per-class for the Train and Test splits side by side.

    Helps identify severe class imbalances.
    """
    # Cast to string to safely handle both int and string labels
    train_labels = df_train.iloc[:, 0].astype(str)
    test_labels = df_test.iloc[:, 0].astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle(f"Class Distribution: {dataset_name}", fontsize=16, fontweight='bold')

    sns.countplot(x=train_labels, ax=axes[0], palette="viridis")
    axes[0].set_title(f"Train Set (N={len(train_labels)})")
    axes[0].set_xlabel("Class Label")
    axes[0].set_ylabel("Count")

    sns.countplot(x=test_labels, ax=axes[1], palette="viridis")
    axes[1].set_title(f"Test Set (N={len(test_labels)})")
    axes[1].set_xlabel("Class Label")

    plt.tight_layout()
    plt.show()


def plot_mean_std_series(df, dataset_name, split_name="Train"):
    """Plot the per-class mean time series with a ±1 standard-deviation band."""
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

        ax = axes[idx]
        ax.plot(time_steps, mean_series, label=f"Class {cls} Mean", color="blue", linewidth=2)
        ax.fill_between(time_steps,
                        mean_series - std_series,
                        mean_series + std_series,
                        color="blue", alpha=0.2, label="±1 Std Dev")

        ax.set_title(f"Class {cls} (N={len(class_data)})")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right")

    # Hide any unused subplots
    for idx in range(num_classes, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()
