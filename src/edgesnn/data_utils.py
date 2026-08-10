"""UCR dataset parsing and loading.

The parser normalizes both whitespace `.txt`/`.tsv` and sktime `.ts` files into a
single in-memory shape, so the rest of the codebase never has to care about the
on-disk format: **column 0 is the class label, columns 1+ are the time series**.
Because of this, the raw files are kept exactly as downloaded — no `.ts`→`.txt`
conversion is needed.

Two tiers:
    parse_ucr_file(filepath) low-level, one file -> DataFrame
    load_ucr_data(data_dir, name) convenience, resolves TRAIN/TEST paths
"""

import os
import shutil
import tempfile
import urllib.request
import zipfile

import pandas as pd

# The raw datasets are NOT committed to git (InsectSound alone is >160 MB per
# split, over GitHub's 100 MB limit). They are downloaded on demand from the
# official UEA/UCR archive. `ensure_dataset` fetches + extracts them into
# `data_dir`, and `load_ucr_data` calls it automatically, so a fresh clone works
# without any manual data setup.
UCR_EXTENSIONS = ('.txt', '.tsv', '.ts')

DATASET_URLS = {
    "ECG5000": "https://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip",
    "Wafer": "https://www.timeseriesclassification.com/aeon-toolkit/Wafer.zip",
    "InsectSound": "https://www.timeseriesclassification.com/aeon-toolkit/InsectSound.zip",
}


def _resolve_split_paths(data_dir, dataset_name):
    """Locate the TRAIN and TEST files for a dataset, or (None, None).

    Searches `data_dir` flat, then a nested `data_dir/dataset_name/` folder,
    across the supported UCR extensions.
    """
    for base in (data_dir, os.path.join(data_dir, dataset_name)):
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


def ensure_dataset(dataset_name, data_dir="Data", url=None):
    """Ensure a dataset's TRAIN/TEST files exist locally, downloading if needed.

    If the files are already present in `data_dir`, does nothing. Otherwise
    downloads the official archive zip, extracts the `_TRAIN`/`_TEST` files (in
    the formats our parser reads — `.txt`/`.tsv`/`.ts`, ignoring `.arff`), and
    copies them into `data_dir`. Returns `(train_path, test_path)`.
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

    os.makedirs(data_dir, exist_ok=True)
    print(f"[data] {dataset_name} not found in {data_dir} — downloading from {url}")

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

        # Copy the split files we can parse into data_dir (archives are flat or
        # nested; walk to be safe, and skip formats we can't read like .arff).
        wanted = {f"{dataset_name}_TRAIN", f"{dataset_name}_TEST"}
        copied = 0
        for root, _dirs, files in os.walk(extract_dir):
            for fname in files:
                stem, ext = os.path.splitext(fname)
                if stem in wanted and ext in UCR_EXTENSIONS:
                    shutil.copy(os.path.join(root, fname), os.path.join(data_dir, fname))
                    copied += 1

    train_path, test_path = _resolve_split_paths(data_dir, dataset_name)
    if not (train_path and test_path):
        raise RuntimeError(
            f"Downloaded and extracted {dataset_name} but could not find TRAIN/TEST "
            f"files in the archive (copied {copied} file(s))."
        )
    print(f"[data] {dataset_name} ready in {data_dir}")
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


def load_ucr_data(data_dir, dataset_name, download=True):
    """Load the TRAIN and TEST splits for a dataset by name.

    Looks for the files in `data_dir` (and a nested `data_dir/dataset_name/`
    folder) across the supported UCR extensions. When `download=True` (default),
    missing datasets are fetched automatically via `ensure_dataset`, so this
    works on a fresh clone with no data committed. Returns `(df_train, df_test)`,
    or `(None, None)` if the files are missing and cannot be obtained.
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

    df_train = parse_ucr_file(train_path)
    df_test = parse_ucr_file(test_path)

    return df_train, df_test
