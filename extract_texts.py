from pathlib import Path
import pandas as pd
import sys


def extract_texts(data_dir, notes_path):
    """Extract raw note texts for all MedDec JSON files
    Reads NOTEEVENTS.csv in chunks until all MedDec notes are found within it
    """
    data_dir = Path(data_dir)
    notes_path = Path(notes_path)
    text_dir = data_dir / "raw_text"
    text_dir.mkdir(exist_ok=True)

    json_files = list((data_dir / "data").glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {data_dir / 'data'}")

    # Build lookup: (SUBJECT_ID, HADM_ID, ROW_ID) -> stem
    needed = {}
    for fn in json_files:
        sid, hadm, rid = map(int, fn.stem.split("_"))
        needed[(sid, hadm, rid)] = fn.stem

    # Scan NOTEEVENTS in chunks; stop once every needed note is found
    found = {}
    for chunk in pd.read_csv(notes_path, usecols=["SUBJECT_ID", "HADM_ID", "ROW_ID", "TEXT"], chunksize=10000):
        for row in chunk.itertuples(index=False):
            key = (row.SUBJECT_ID, row.HADM_ID, row.ROW_ID)
            if key in needed and key not in found:
                found[key] = row.TEXT
        if len(found) == len(needed):
            break

    missing = 0
    for key, stem in needed.items():
        if key in found:
            out_path = text_dir / f"{stem}.txt"
            out_path.write_text(found[key], encoding="utf-8")
        else:
            print(f"WARNING: Note not found: {stem}")
            missing += 1

    print(f"Extracted {len(found)} notes to {text_dir}  ({missing} missing)")
    return len(found), missing


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_texts.py <data_dir> <notes_path>")
        sys.exit(1)
    extract_texts(sys.argv[1], sys.argv[2])
