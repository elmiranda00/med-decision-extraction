from pathlib import Path
from collections import defaultdict
import random
import sys


def build_splits(meddec_dir, seed=42, train_frac=0.8, val_frac=0.1):
    """Split MedDec JSON filenames 80/10/10 (default) by SUBJECT_ID to prevent leakage.

    Returns (train_names, val_names, test_names) as sorted lists of filenames.
    """
    meddec_dir = Path(meddec_dir)
    splits_dir = meddec_dir / "splits"
    splits_dir.mkdir(exist_ok=True)

    json_files = list((meddec_dir / "data").glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files in {meddec_dir / 'data'}")

    # Group filenames by SUBJECT_ID to avoid leakage across splits
    by_subject = defaultdict(list)
    for f in json_files:
        sid = f.stem.split("_")[0]
        by_subject[sid].append(f.name)

    subjects = sorted(by_subject.keys())
    rng = random.Random(seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train_subs = subjects[:n_train]
    val_subs = subjects[n_train : n_train + n_val]
    test_subs = subjects[n_train + n_val :]

    def collect(subject_list):
        names = sorted(name for s in subject_list for name in by_subject[s])
        return names

    train_names = collect(train_subs)
    val_names = collect(val_subs)
    test_names = collect(test_subs)

    for split_name, names in [("train", train_names), ("val", val_names), ("test", test_names)]:
        (splits_dir / f"{split_name}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

    print(f"Train: {len(train_names)} files ({len(train_subs)} subjects)")
    print(f"Val:   {len(val_names)} files ({len(val_subs)} subjects)")
    print(f"Test:  {len(test_names)} files ({len(test_subs)} subjects)")
    return train_names, val_names, test_names


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python build_splits.py <meddec_dir>")
        sys.exit(1)
    build_splits(sys.argv[1])
