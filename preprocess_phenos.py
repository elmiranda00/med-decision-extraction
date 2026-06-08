# Aggregate per-note phenotype annotations into a single label.
# Priority order: DAG/PAT > JTW/JF > ETM/JW
# Label is '?' when the selected annotator marked UNSURE, or when two
# selected annotators disagree and one marked NONE.

import pandas as pd
import sys
from pathlib import Path


PHENOTYPE_COLUMNS = [
    "ADVANCED.CANCER", "ADVANCED.HEART.DISEASE", "ADVANCED.LUNG.DISEASE",
    "ALCOHOL.ABUSE", "CHRONIC.NEUROLOGICAL.DYSTROPHIES", "CHRONIC.PAIN.FIBROMYALGIA",
    "DEPRESSION", "OBESITY", "OTHER.SUBSTANCE.ABUSE", "PSYCHIATRIC.DISORDERS",
    "NONE", "UNSURE",
]

# Operator tier list (Lower number = higher priority)
_OPERATOR_TIER = {"DAG": 0, "PAT": 0, "JTW": 1, "JF": 1, "ETM": 2, "JW": 2}


def _select_priority_rows(group):
    """Return only the rows from the highest-priority annotator(s) in group."""
    best_tier = group["OPERATOR"].map(_OPERATOR_TIER).min()
    return group[group["OPERATOR"].map(_OPERATOR_TIER) == best_tier]


def _rows_to_label(selected):
    """Convert selected annotation rows into a phenotype label string."""
    # Drop bookkeeping columns and deduplicate identical annotations
    pheno_cols = [c for c in selected.columns if c not in ("BATCH.ID", "OPERATOR")]
    deduped = selected[pheno_cols].drop_duplicates()
    summed = deduped[PHENOTYPE_COLUMNS].sum()

    # Conflicting NONE among selected annotators → uncertain
    if len(deduped) > 1 and summed["NONE"] > 0:
        return "?"

    # Selected annotator(s) explicitly marked UNSURE → uncertain
    if summed["UNSURE"] > 0:
        return "?"

    active = [col for col in PHENOTYPE_COLUMNS if col not in ("UNSURE",) and summed[col] > 0]
    return ",".join(sorted(active)) if active else "?"


def aggregate_annotations(df):
    
    def process_group(group):
        selected = _select_priority_rows(group)
        return pd.Series({"phenotype_label": _rows_to_label(selected), 
                          "OPERATOR": ",".join(selected["OPERATOR"].unique()),})

    return df.groupby(["SUBJECT_ID", "HADM_ID", "ROW_ID"]).apply(process_group).reset_index()


def preprocess_phenos(input_file, output_file=None):
    input_file = Path(input_file)
    if output_file is None:
        output_file = input_file.parent / "phenos.csv"

    df = pd.read_csv(input_file)
    df["PSYCHIATRIC.DISORDERS"] = (
        df["DEMENTIA"]
        | df["DEVELOPMENTAL.DELAY.RETARDATION"]
        | df["SCHIZOPHRENIA.AND.OTHER.PSYCHIATRIC.DISORDERS"]
    )
    df = df[["SUBJECT_ID", "HADM_ID", "ROW_ID"] + PHENOTYPE_COLUMNS + ["OPERATOR", "BATCH.ID"]]

    result_df = aggregate_annotations(df)
    result_df.to_csv(output_file, index=False)
    print(f"Saved {len(result_df)} rows to {output_file}")
    return result_df


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preprocess_phenos.py <input_file>")
        sys.exit(1)
    preprocess_phenos(sys.argv[1])
