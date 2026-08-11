"""UCR dataset parsing, caching and Dataset/DataLoader construction.

The parser normalizes both whitespace `.txt`/`.tsv` and sktime `.ts` files into a
single in-memory shape, so the rest of the codebase never has to care about the
on-disk format: **column 0 is the class label, columns 1+ are the time series**.
Because of this, the raw files are kept exactly as downloaded — no `.ts`→`.txt`
conversion is needed.

On-disk layout (all git-ignored)::

    Data/raw/      the .ts files as downloaded — immutable input
    Data/cache/    the *parse result*, verbatim, as binary .npz — derived, disposable

`cache/` holds the same numbers `parse_ucr_file` returns (column 0 still the
original label: "Aedes_female", "-1", "5"). It is a *format* conversion only —
163 MB of ASCII floats re-encoded as 60 MB of float32 — and is keyed on the raw
file's size+mtime so a changed raw file transparently rebuilds it.

Label aggregation and normalization deliberately do **not** touch disk: they are
cheap, and they are experiment knobs (binary vs 5-class ECG5000, which
normalization), so a stale `processed/` folder could silently disagree with the
code. They happen in `UCRDataset.__init__`, in RAM, every run.

Tiers:
    parse_ucr_file(filepath)              low-level, one file -> DataFrame
    load_ucr_data(data_dir, name)         convenience, resolves TRAIN/TEST paths
    make_datasets(data_dir, name)         -> (train_ds, test_ds), encoded + normalized
    make_dataloaders(data_dir, name)      -> (train_loader, test_loader, train_ds, test_ds)
"""

import os
import shutil
import tempfile
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# The raw datasets are NOT committed to git (InsectSound alone is >160 MB per
# split, over GitHub's 100 MB limit). They are downloaded on demand from the
# official UEA/UCR archive. `ensure_dataset` fetches + extracts them into
# `data_dir`, and `load_ucr_data` calls it automatically, so a fresh clone works
# without any manual data setup.
# Formats the parser can read (used when locating already-present files).
UCR_EXTENSIONS = ('.txt', '.ts')
# When extracting a downloaded archive we keep ONLY this format
_EXTRACT_EXTENSION = '.ts'

# Subfolders of `data_dir`. `raw/` is the immutable download target; `cache/`
# holds derived binaries and is safe to delete at any time.
RAW_SUBDIR = "raw"
CACHE_SUBDIR = "cache"

DATASET_URLS = {
    "ECG5000": "https://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip",
    "Wafer": "https://www.timeseriesclassification.com/aeon-toolkit/Wafer.zip",
    "InsectSound": "https://www.timeseriesclassification.com/aeon-toolkit/InsectSound.zip",
}


def raw_dir(data_dir):
    """Path to the immutable raw-download folder inside `data_dir`."""
    return os.path.join(data_dir, RAW_SUBDIR)


def cache_dir(data_dir):
    """Path to the derived-binary cache folder inside `data_dir`."""
    return os.path.join(data_dir, CACHE_SUBDIR)


def _search_bases(data_dir, dataset_name):
    """Folders to look in for split files, most-preferred first.

    `raw/` is canonical; the flat `data_dir` and a nested `data_dir/name/` are
    still searched so pre-existing downloads keep working without a migration.
    """
    return (
        raw_dir(data_dir),
        os.path.join(raw_dir(data_dir), dataset_name),
        data_dir,
        os.path.join(data_dir, dataset_name),
    )


def _resolve_split_paths(data_dir, dataset_name):
    """Locate the TRAIN and TEST files for a dataset, or (None, None).

    Searches `data_dir/raw/` first, then the legacy flat/nested locations,
    across the supported UCR extensions.
    """
    for base in _search_bases(data_dir, dataset_name):
        train_path = test_path = None
        for ext in UCR_EXTENSIONS:
            t = os.path.join(base, f"{dataset_name}_TRAIN{ext}")
            if os.path.exists(t):
                train_path = t
            tst = os.path.join(base, f"{dataset_name}_TEST{ext}")
            if os.path.exists(tst):
                test_path = tst
        if train_path and test_path:
            return train_path, test_path
    return None, None


def migrate_to_raw(data_dir="Data", verbose=True):
    """Move any split files sitting flat in `data_dir` into `data_dir/raw/`.

    One-shot helper for repos that downloaded data under the old flat layout.
    Safe to call repeatedly — it is a no-op once `data_dir` has no loose splits.
    Returns the number of files moved.
    """
    if not os.path.isdir(data_dir):
        return 0
    target = raw_dir(data_dir)
    os.makedirs(target, exist_ok=True)

    moved = 0
    for filename in os.listdir(data_dir):
        stem, ext = os.path.splitext(filename)
        if ext not in UCR_EXTENSIONS:
            continue
        if not (stem.endswith("_TRAIN") or stem.endswith("_TEST")):
            continue
        src = os.path.join(data_dir, filename)
        if not os.path.isfile(src):
            continue
        shutil.move(src, os.path.join(target, filename))
        moved += 1

    if verbose and moved:
        print(f"[data] migrated {moved} file(s) from {data_dir} -> {target}")
    return moved


def ensure_dataset(dataset_name, data_dir="Data", url=None):
    """Ensure a dataset's TRAIN/TEST files exist locally, downloading if needed.

    If the files are already present, does nothing. Otherwise, downloads the
    official archive zip and extracts the `_TRAIN`/`_TEST` splits as `.ts` only
    (for a consistent Data folder; other formats in the archive E.g.
    `.txt`/`.arff` are ignored) into `data_dir/raw/`. Returns
    `(train_path, test_path)`.
    """
    train_path, test_path = _resolve_split_paths(data_dir, dataset_name)
    if train_path and test_path:
        return train_path, test_path

    url = url or DATASET_URLS.get(dataset_name)
    if url is None:
        raise ValueError(
            f"No download URL known for '{dataset_name}'. Pass `url=` or place the "
            f"{dataset_name}_TRAIN/_TEST files in {data_dir} manually."
        )

    target_dir = raw_dir(data_dir)
    os.makedirs(target_dir, exist_ok=True)
    print(f"[data] {dataset_name} not found in {target_dir} — downloading from {url}")

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, f"{dataset_name}.zip")

        last_pct = [-1]

        def _progress(block_num, block_size, total_size):
            if total_size <= 0:
                return
            done = min(block_num * block_size, total_size)
            pct = int(done * 100 / total_size)
            if pct != last_pct[0]:  # only redraw on whole-percent change
                last_pct[0] = pct
                print(f"\r[data]   {done / 1e6:6.1f} / {total_size / 1e6:.1f} MB ({pct:3d}%)", end="")

        urllib.request.urlretrieve(url, zip_path, reporthook=_progress)
        print()  # newline after progress

        extract_dir = os.path.join(tmp, "extracted")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

        # Copy only the .ts TRAIN/TEST splits into raw/ (archives are flat or
        # nested; walk to be safe, and skip .txt/.arff for a uniform folder).
        wanted = {f"{dataset_name}_TRAIN", f"{dataset_name}_TEST"}
        copied = 0
        for root, _dirs, files in os.walk(extract_dir):
            for filename in files:
                stem, ext = os.path.splitext(filename)
                if stem in wanted and ext == _EXTRACT_EXTENSION:
                    shutil.copy(os.path.join(root, filename), os.path.join(target_dir, filename))
                    copied += 1

    train_path, test_path = _resolve_split_paths(data_dir, dataset_name)
    if not (train_path and test_path):
        raise RuntimeError(
            f"Downloaded and extracted {dataset_name} but could not find TRAIN/TEST "
            f"files in the archive (copied {copied} file(s))."
        )
    print(f"[data] {dataset_name} ready in {target_dir}")
    return train_path, test_path


def parse_ucr_file(filepath):
    """Parse a single UCR file (`.txt`/`.tsv`/`.ts`) into a DataFrame.

    Column 0 holds the class label (as parsed; may be str for `.ts`), the
    remaining columns hold the time-series values.
    """
    if filepath.endswith('.ts'):
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            data_start = False
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if not data_start:
                    if line.lower().startswith('@data'):
                        data_start = True
                    continue

                # Parse the actual data line
                if ':' in line:
                    series, label = line.split(':', 1)
                    vals = [label.strip()] + [float(x) for x in series.split(',') if x.strip()]
                else:
                    parts = line.split(',')
                    vals = [parts[-1].strip()] + [float(x) for x in parts[:-1] if x.strip()]
                data.append(vals)
        return pd.DataFrame(data)
    else:
        return pd.read_csv(filepath, sep=r'\s+', header=None)


# --------------------------------------------------------------------------
# Parse cache
#
# Purely a format conversion of `parse_ucr_file`'s output: the same labels and
# the same numbers, stored as binary instead of ASCII. InsectSound is ~163 MB of
# text per split (~15 bytes of ASCII per 4-byte float) and takes tens of seconds
# to parse; as float32 it is ~60 MB and loads near-instantly. Keyed on the raw
# file's size+mtime, so editing or re-downloading a raw file rebuilds it.
# --------------------------------------------------------------------------


def _cache_path(filepath, cache_root):
    return os.path.join(cache_root, os.path.basename(filepath) + ".npz")


def _df_to_arrays(df):
    """Split a parsed DataFrame into (labels, values) arrays for storage.

    Object-dtype label columns (the `.ts` case, where labels are strings) are
    narrowed to a unicode array so `np.savez` can store them without pickling.
    Numeric label columns keep their dtype, so round-tripping a `.txt` file
    returns the same DataFrame it went in as.
    """
    labels = df.iloc[:, 0].to_numpy()
    if labels.dtype == object:
        labels = labels.astype(str)
    values = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    return labels, values


def _arrays_to_df(labels, values):
    """Rebuild the parser's DataFrame shape (column 0 = label, 1+ = series)."""
    df = pd.DataFrame(values)
    df.columns = pd.RangeIndex(1, values.shape[1] + 1)
    df.insert(0, 0, labels)
    return df


def parse_ucr_file_cached(filepath, cache_root, verbose=True):
    """`parse_ucr_file`, memoized to a binary `.npz` under `cache_root`.

    Returns the identical DataFrame the uncached parser would. The cache is
    derived state — deleting `cache_root` only costs one re-parse.
    """
    path = _cache_path(filepath, cache_root)
    stat = os.stat(filepath)

    if os.path.exists(path):
        try:
            with np.load(path) as z:
                # Stale cache (raw file re-downloaded or edited) -> fall through.
                if int(z["src_size"]) == stat.st_size and float(z["src_mtime"]) == stat.st_mtime:
                    return _arrays_to_df(z["labels"], z["values"])
        except Exception as e:  # truncated/corrupt cache is never fatal
            if verbose:
                print(f"[data] ignoring unreadable cache {os.path.basename(path)}: {e}")

    df = parse_ucr_file(filepath)
    labels, values = _df_to_arrays(df)
    os.makedirs(cache_root, exist_ok=True)
    np.savez(path, labels=labels, values=values,
             src_size=np.int64(stat.st_size), src_mtime=np.float64(stat.st_mtime))
    if verbose:
        print(f"[data] cached {os.path.basename(filepath)} -> {os.path.relpath(path, cache_root)}")
    return df


def load_ucr_data(data_dir, dataset_name, download=True, use_cache=True):
    """Load the TRAIN and TEST splits for a dataset by name.

    Looks for the files in `data_dir/raw/` (falling back to the legacy flat and
    nested locations) across the supported UCR extensions. When `download=True`
    (default), missing datasets are fetched automatically via `ensure_dataset`,
    so this works on a fresh clone with no data committed. With `use_cache=True`
    the parse result is memoized under `data_dir/cache/`. Returns
    `(df_train, df_test)`, or `(None, None)` if the files are missing and cannot
    be obtained.
    """
    train_path, test_path = _resolve_split_paths(data_dir, dataset_name)

    if not (train_path and test_path) and download:
        try:
            train_path, test_path = ensure_dataset(dataset_name, data_dir)
        except Exception as e:  # network error, unknown dataset, bad archive
            print(f"[ERROR] Could not obtain {dataset_name}: {e}")
            return None, None

    if not (train_path and test_path):
        print(f"[ERROR] Could not find {dataset_name} files in {data_dir}")
        return None, None

    if use_cache:
        root = cache_dir(data_dir)
        return parse_ucr_file_cached(train_path, root), parse_ucr_file_cached(test_path, root)

    return parse_ucr_file(train_path), parse_ucr_file(test_path)


# --------------------------------------------------------------------------
# Label encoding
#
# Driven by the EDA findings. These transforms are applied in memory at Dataset
# construction, never written to disk — see the module docstring for why.
# --------------------------------------------------------------------------

# Maps from *canonical* original label -> target integer class.
# A dataset absent here gets an auto-derived map (sorted unique TRAIN labels
# -> 0..C-1), which is what InsectSound's 10 string class names need.
LABEL_MAPS = {
    # Severe imbalance across classes 1-5; aggregate to normal vs abnormal.
    "ECG5000": {"1": 0, "2": 1, "3": 1, "4": 1, "5": 1},
    # Labels are -1 / 1; shift to 0 / 1 so they index logits directly.
    "Wafer": {"-1": 0, "1": 1},
}

# Human-readable names for the *encoded* classes, where the aggregation makes
# the raw labels misleading. Datasets not listed derive names from the mapping.
CLASS_NAMES = {
    "ECG5000": ["normal (1)", "abnormal (2-5)"],
    # Raw -1 is the fault class, raw 1 is normal — so encoded 0 is the abnormal one.
    "Wafer": ["abnormal (-1)", "normal (1)"],
}


def canon_label(label):
    """Canonical string form of a raw label, so maps are dtype-agnostic.

    `.ts` files yield strings, `.txt` files floats, and the same class may
    appear as `1`, `1.0` or `"1"`. Numeric-looking labels collapse to their
    integer form; everything else (`"Aedes_female"`) is passed through stripped.
    """
    s = str(label).strip()
    try:
        f = float(s)
    except ValueError:
        return s
    return str(int(f)) if f.is_integer() else s


def build_label_map(labels, dataset_name=None):
    """Return the canonical-label -> int mapping to use for a dataset.

    Uses the curated `LABEL_MAPS` entry when one exists, otherwise derives a
    stable map from the sorted unique labels. Always build this from the TRAIN
    split and pass it to TEST, so both splits share one encoding.
    """
    if dataset_name in LABEL_MAPS:
        return dict(LABEL_MAPS[dataset_name])
    return {lab: i for i, lab in enumerate(sorted({canon_label(x) for x in labels}))}


def encode_labels(labels, label_map):
    """Apply `label_map` to a sequence of raw labels -> int64 numpy array."""
    out = np.empty(len(labels), dtype=np.int64)
    for i, raw in enumerate(labels):
        key = canon_label(raw)
        if key not in label_map:
            raise KeyError(
                f"Label {raw!r} (canonical {key!r}) is not in the label map "
                f"{sorted(label_map)}. If this is a TEST-only class, the map was "
                f"built from TRAIN and cannot encode it."
            )
        out[i] = label_map[key]
    return out


def derive_class_names(label_map, dataset_name=None):
    """Readable name per encoded class, for confusion-matrix axes and legends."""
    if dataset_name in CLASS_NAMES:
        return list(CLASS_NAMES[dataset_name])
    n = max(label_map.values()) + 1
    groups = [[] for _ in range(n)]
    for raw, idx in sorted(label_map.items()):
        groups[idx].append(raw)
    return ["/".join(g) for g in groups]


# --------------------------------------------------------------------------
# Dataset / DataLoader
# --------------------------------------------------------------------------


class UCRDataset(Dataset):
    """A UCR split held entirely in RAM as two tensors.

    `x` is `(N, T)` float32, `y` is `(N,)` int64. Even InsectSound — the largest
    of the three — is only ~60 MB per split as float32, and surrogate-gradient
    BPTT sweeps the same tensor for many epochs, so eager loading is strictly
    better than per-item disk reads. `__getitem__` is a pure index op, which is
    why `num_workers=0` is the right DataLoader setting here: worker processes
    would add pickling overhead and buy nothing.

    Normalization stats are *not* fit per split. Build TRAIN first, then pass its
    `.stats` into TEST so the test split is scaled by train statistics.
    """

    def __init__(self, df, label_map, normalize="global", stats=None,
                 dataset_name=None, split=None):
        self.dataset_name = dataset_name
        self.split = split
        self.label_map = dict(label_map)
        self.class_names = derive_class_names(self.label_map, dataset_name)
        self.normalize = normalize

        # Keep the pre-encoding labels so plots/confusion matrices can show
        # "Aedes_female" rather than 7.
        self.raw_labels = df.iloc[:, 0].astype(str).reset_index(drop=True)
        self.y = torch.from_numpy(encode_labels(df.iloc[:, 0].tolist(), self.label_map))

        # .copy() because a cache-backed frame can hand back a read-only array,
        # which torch refuses to wrap without warning.
        x = torch.from_numpy(df.iloc[:, 1:].to_numpy(dtype=np.float32).copy())

        if normalize == "global":
            if stats is None:
                mean, std = x.mean(), x.std()
            else:
                mean, std = (torch.as_tensor(s, dtype=torch.float32) for s in stats)
            std = torch.clamp(std, min=1e-8)  # guard a constant-valued split
            self.x = (x - mean) / std
            self.stats = (mean, std)
        elif normalize == "per_sample":
            # Each series standardized against itself — removes per-recording
            # amplitude offsets; carries no train->test leakage by construction.
            mean = x.mean(dim=1, keepdim=True)
            std = torch.clamp(x.std(dim=1, keepdim=True), min=1e-8)
            self.x = (x - mean) / std
            self.stats = None
        elif normalize in (None, "none"):
            self.x = x
            self.stats = None
        else:
            raise ValueError(
                f"Unknown normalize={normalize!r}; expected 'global', 'per_sample' or None."
            )

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    @property
    def num_classes(self):
        return len(self.class_names)

    @property
    def seq_len(self):
        """Sequence length T — the ceiling for the truncation parameter `k`."""
        return self.x.shape[1]

    def class_counts(self):
        """Encoded-class -> instance count, as a DataFrame for reporting."""
        counts = torch.bincount(self.y, minlength=self.num_classes).tolist()
        return pd.DataFrame({
            "class": range(self.num_classes),
            "name": self.class_names,
            "count": counts,
            "proportion": [c / len(self) for c in counts],
        })

    def to_frame(self, denormalize=False):
        """The transformed data in `parse_ucr_file`'s shape, for `plotting.py`.

        Lets the EDA plotters run unchanged on the *encoded* data, so a
        before/after figure pair compares the raw parse against the exact
        tensors the model will train on. Set `denormalize=True` when the figure
        is about waveform morphology — after z-scoring the y-axis is in standard
        deviations, which reads oddly as "what an ECG beat looks like".
        """
        x = self.x
        if denormalize and self.stats is not None:
            mean, std = self.stats
            x = x * std + mean
        return _arrays_to_df(self.y.numpy(), x.numpy())


def make_datasets(data_dir, dataset_name, normalize="global", download=True,
                  use_cache=True, label_map=None):
    """Build the (train, test) `UCRDataset` pair for a dataset.

    The label map is built from TRAIN and reused for TEST, and TEST inherits
    TRAIN's normalization stats — both on purpose, so neither encoding drift nor
    test-set leakage is possible by construction. Returns `(None, None)` if the
    data cannot be obtained.
    """
    df_train, df_test = load_ucr_data(data_dir, dataset_name,
                                      download=download, use_cache=use_cache)
    if df_train is None or df_test is None:
        return None, None

    label_map = label_map or build_label_map(df_train.iloc[:, 0].tolist(), dataset_name)

    train_ds = UCRDataset(df_train, label_map, normalize=normalize,
                          dataset_name=dataset_name, split="train")
    test_ds = UCRDataset(df_test, label_map, normalize=normalize,
                         stats=train_ds.stats, dataset_name=dataset_name, split="test")
    return train_ds, test_ds


def make_dataloaders(data_dir, dataset_name, batch_size=64, normalize="global",
                     shuffle=True, num_workers=0, download=True, use_cache=True,
                     label_map=None):
    """`make_datasets` plus DataLoaders. Returns `(train_loader, test_loader, train_ds, test_ds)`.

    `num_workers=0` by default — the tensors are already resident, so
    `__getitem__` is pure indexing and worker processes would only add overhead.
    """
    train_ds, test_ds = make_datasets(data_dir, dataset_name, normalize=normalize,
                                      download=download, use_cache=use_cache,
                                      label_map=label_map)
    if train_ds is None:
        return None, None, None, None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=False)
    return train_loader, test_loader, train_ds, test_ds
