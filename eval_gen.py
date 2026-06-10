import argparse
import json
from pathlib import Path

from gen_span_detection import _parse_cat_number

CATEGORY_NAMES = [
    "Contact-related",
    "Gathering information",
    "Defining problem",
    "Treatment goal",
    "Drug",
    "Therapeutic procedure",
    "Evaluating test result",
    "Deferment",
    "Advice and precaution",
]


# MATCHING

def is_match(pred: str, gold: str, method: str) -> bool:
    """
    Return True if predicted span matches gold i.e. human annotated span (Using the methods below)
    Methods:
    - em: exact string equality after whitespace normalisation
    - approx-m: one is a substring of the other AND word-count gap <= 10
    """
    pred = pred.strip()
    gold = gold.strip()
    if method == "em":
        return pred == gold
    if method == "approx-m":
        substring_match = (pred in gold) or (gold in pred)
        word_gap = abs(len(pred.split()) - len(gold.split()))
        return substring_match and word_gap <= 10
    raise ValueError(f"Unknown method: {method}. Choose 'em' or 'approx-m'.")


# LOAD DATA

def load_gold(meddec_dir: Path, split_file: Path) -> set:
    """
    Build the gold spans set from the original MedDec JSON annotation files.

    Returns : (filename_stem, category_indexed, decision_string)
    """
    meddec_dir = Path(meddec_dir)
    filenames  = [
        ln.strip()
        for ln in Path(split_file).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]

    gold = set()
    for fname in filenames:
        stem     = Path(fname).stem # filename without extension
        json_path = meddec_dir / "data" / fname
        if not json_path.exists():
            continue
        ann_data    = json.loads(json_path.read_text(encoding="utf-8"))
        annotations = ann_data.get("annotations", [])
        for ann in annotations:
            cat = _parse_cat_number(ann.get("category", ""))
            if cat is None or cat > 9:
                continue
            gold.add((stem, cat, ann["decision"].strip()))

    return gold


def load_predictions(pred_dir: Path) -> set:
    """
    Load all prediction files from output directory of gen_span_detection.py.

    Returns : set of (filename_stem, category_indexed, decision_string) triples
    """
    pred_dir = Path(pred_dir)
    preds    = set()

    for json_file in pred_dir.glob("*.json"):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        stem = Path(data.get("file_name", json_file.stem)).stem
        for cat_str, decisions in data.get("predictions", {}).items():
            cat = int(cat_str)
            for dec in decisions:
                dec = dec.strip()
                if dec:
                    preds.add((stem, cat, dec))

    return preds


# EVALUATION FUNCTIONS

def _compute_f1_sets(gold: set, pred: set, method: str) -> dict:
    
    matched_gold = set()
    tp = 0
    for p_stem, p_cat, p_dec in pred:
        for g_key in list(gold - matched_gold):
            g_stem, g_cat, g_dec = g_key
            if p_stem == g_stem and p_cat == g_cat and is_match(p_dec, g_dec, method):
                tp += 1
                matched_gold.add(g_key)
                break

    fp = len(pred) - tp
    fn = len(gold) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "n_gold": len(gold), "n_pred": len(pred),
    }


def evaluate_predictions(meddec_dir, split_file, pred_dir, method = "em",) -> dict:
    """
    Full evaluation: overall + per-category F1.

    Args:
        meddec_dir:  path to meddec-mimic-iii/
        split_file:  path to splits/test.txt (or val.txt)
        pred_dir:    output directory from run_pipeline()
        method:      "em" or "approx-m"

    Returns:
        Results in dictionary format
    """
    gold = load_gold(Path(meddec_dir), Path(split_file))
    pred = load_predictions(Path(pred_dir))

    print(f"Gold triples : {len(gold)}")
    print(f"Pred triples : {len(pred)}")

    overall = _compute_f1_sets(gold, pred, method)

    per_cat = {}
    for cat in range(1, 10):
        cat_gold = {t for t in gold if t[1] == cat}
        cat_pred = {t for t in pred if t[1] == cat}
        per_cat[cat] = _compute_f1_sets(cat_gold, cat_pred, method)

    return {"overall": overall, "per_cat": per_cat, "method": method}


# PRINT RESULTS

def print_results(results: dict, label: str = "") -> None:
    method = results.get("method", "")
    ov     = results["overall"]
    header = f"  LLM Span F1 ({method})"
    if label:
        header += f" — {label}"
    print(f"\n{'='*60}")
    print(header)
    print(f"{'='*60}")
    print(f"  Gold triples : {ov['n_gold']:5d}")
    print(f"  Pred triples : {ov['n_pred']:5d}")
    print(f"  TP={ov['tp']}  FP={ov['fp']}  FN={ov['fn']}")
    print(f"  Precision  : {ov['precision']:.4f}")
    print(f"  Recall     : {ov['recall']:.4f}")
    print(f"  F1         : {ov['f1']:.4f}  ← main metric")
    print(f"\n  {'Cat':>3}  {'Name':25s}  {'F1':>6}  {'P':>6}  {'R':>6}  {'Gold':>5}  {'Pred':>5}")
    print(f"  {'-'*62}")
    for cat in range(1, 10):
        m = results["per_cat"][cat]
        print(
            f"  {cat:>3}  {CATEGORY_NAMES[cat-1]:25s}  "
            f"{m['f1']:>6.3f}  {m['precision']:>6.3f}  {m['recall']:>6.3f}  "
            f"{m['n_gold']:>5d}  {m['n_pred']:>5d}"
        )
    print(f"{'='*60}\n")


# CLI

def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate LLM decision span extraction")
    p.add_argument("--meddec_dir", required=True)
    p.add_argument("--splits_dir", required=True)
    p.add_argument("--pred_dir",   required=True, help="Output dir from gen_span_detection.py")
    p.add_argument("--split",      default="test", choices=["train", "val", "test"])
    p.add_argument("--method",     default="em",   choices=["em", "approx-m"])
    return p.parse_args()


if __name__ == "__main__":
    args    = _parse_args()
    results = evaluate_predictions(meddec_dir = args.meddec_dir, split_file = Path(args.splits_dir) / f"{args.split}.txt", pred_dir = args.pred_dir, method = args.method,)
    print_results(results)
