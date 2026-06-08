from pathlib import Path
import pandas as pd
import sys


def build_stats(data_dir):
    """Join PATIENTS + ADMISSIONS, filtered to MedDec subject/admission IDs.

    Output columns: SUBJECT_ID, HADM_ID, GENDER, ETHNICITY, LANGUAGE
    """
    data_dir = Path(data_dir)
    meddec_dir = data_dir / "meddec-mimic-iii"
    admissions_path = data_dir / "ADMISSIONS.csv" / "ADMISSIONS.csv"
    patients_path = data_dir / "PATIENTS.csv" / "PATIENTS.csv"
    output_path = meddec_dir / "stats.csv"

    # Collect (SUBJECT_ID, HADM_ID) pairs present in MedDec
    json_files = list((meddec_dir / "data").glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {meddec_dir / 'data'}")

    meddec_ids = pd.DataFrame([tuple(map(int, f.stem.split("_")[:2])) for f in json_files],
                              columns=["SUBJECT_ID", "HADM_ID"]).drop_duplicates()

    patients = pd.read_csv(patients_path, usecols=["SUBJECT_ID", "GENDER"])
    admissions = pd.read_csv(admissions_path, usecols=["SUBJECT_ID", "HADM_ID", "ETHNICITY", "LANGUAGE"])

    merged = admissions.merge(patients, on="SUBJECT_ID")
    result = merged.merge(meddec_ids, on=["SUBJECT_ID", "HADM_ID"])
    result = result[["SUBJECT_ID", "HADM_ID", "GENDER", "ETHNICITY", "LANGUAGE"]]

    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} rows to {output_path}")
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python build_stats.py <data_dir>")
        sys.exit(1)
    build_stats(sys.argv[1])
