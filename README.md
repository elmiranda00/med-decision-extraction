# Medical Decision Extraction from ICU Discharge Summaries

A partial replication of Elgaar et al. (2024), "MedDec: A Dataset for Extracting Medical Decisions from Discharge Summaries" (NeurIPS 2024).
Paper: https://arxiv.org/abs/2408.12980 — Dataset: https://physionet.org/content/meddec-mimic-iii

The goal is to extract and classify spans of text from ICU clinical notes (MIMIC-III discharge summaries) that represent a medical decision, and assign each span to one of 9 decision categories from the DICTUM taxonomy (e.g. Drug, Therapeutic Procedure, Defining Problem).

---

## Methodology

Two approaches are implemented and compared.

**Approach 1 — ELECTRA fine-tuning**

ELECTRA-base is fine-tuned for BIO token classification, i.e. each token in a clinical note is labelled as the beginning (B), inside (I), or outside (O) of a decision span for a given category. Evaluation uses exact span-level F1: a predicted span is a true positive only if the category, start token, and end token all match a gold (i.e. human) annotation exactly.

**Approach 2 — LLM prompting**

A pretrained language model (FLAN-T5) is prompted once per category per note to list all decision substrings belonging to that category. Two settings are tested: zero-shot (no examples given) and one-shot (one in-context example drawn from the same note's annotations). Evaluation uses string-level F1 with exact match and approximate match (one string is a substring of the other and the word count difference is at most 10).

---

## Files

### Data preparation (run once before training)

| File | What it does |
|---|---|
| extract_texts.py | Extracts raw note text for each annotated note from NOTEEVENTS.csv |
| build_stats.py | Joins PATIENTS + ADMISSIONS to produce demographic stats per note |
| preprocess_phenos.py | Processes phenotype annotations into one label per note |
| build_splits.py | Creates 80/10/10 train/val/test splits by subject to avoid data leakage |

### Phase 2 — ELECTRA

| File | What it does |
|---|---|
| dataset.py | PyTorch dataset: tokenises notes, maps character offsets to token-level BIO labels |
| model.py | ELECTRA-base encoder with a linear token classification head (19 output classes) |
| train.py | Training loop: AdamW optimiser, linear warmup, gradient accumulation, MLflow logging |
| evaluate.py | Chunked inference, BIO decoding, and span-level F1 computation |

### Phase 3 — LLM prompting

| File | What it does |
|---|---|
| gen_span_detection.py | Prompt construction, model loading, generation, output parsing, and pipeline runner |
| eval_gen.py | String-level F1 evaluation (exact match and approximate match) |

### Notebooks

| Notebook | What it does |
|---|---|
| workingfolder/train_electra.ipynb | Full ELECTRA train, evaluate, MLflow summary, and checkpoint save workflow for Colab |
| workingfolder/train_flan.ipynb | Full FLAN-T5 zero-shot and one-shot pipeline with evaluation and README figures for Colab |

---

## Running the Code

**Data access**: This project requires MIMIC-III access via PhysioNet (credentialed). The MedDec annotation files are available at the dataset link above. Data preparation scripts must be run before training.

**Recommended: Google Colab (free tier, T4 GPU)**

1. Upload the data folder to Google Drive under `MyDrive/AI4H-project-rework/02 Data/`
2. Upload the code folder to `MyDrive/AI4H-project-rework/04 Code/04 Code/med-decision-extraction/`
3. Open `workingfolder/train_electra.ipynb` in Colab for ELECTRA training. 5 epochs takes around 3 minutes on a T4.
4. Open `workingfolder/train_flan.ipynb` in Colab for LLM prompting. The full test set takes around 22 minutes per mode with flan-t5-xl.


Install dependencies:

```
pip install -r requirement.txt
```

---

## Results

### Approach 1 - ELECTRA (5 epochs, 324 training notes, 41 test notes)

| Category | Precision | Recall | F1 |
|---|---|---|---|
| Contact-related | 0.052 | 0.089 | 0.065 |
| Gathering information | 0.000 | 0.000 | 0.000 |
| Defining problem | 0.136 | 0.130 | 0.133 |
| Treatment goal | 0.000 | 0.000 | 0.000 |
| Drug | 0.239 | 0.247 | 0.243 |
| Therapeutic procedure | 0.103 | 0.135 | 0.117 |
| Evaluating test result | 0.044 | 0.055 | 0.049 |
| Deferment | 0.000 | 0.000 | 0.000 |
| Advice and precaution | 0.024 | 0.031 | 0.027 |
| **Overall** | **0.129** | **0.141** | **0.135** |

### Approach 2 - FLAN-T5-xl string-level F1 (41 test notes)

| Setting | Exact match F1 | Approximate match F1 |
|---|---|---|
| Zero-shot | 0.000 | 0.002 |
| One-shot | 0.000 | 0.000 |

### Paper results for comparison (Elgaar et al., 2024)

| Model | Span F1 |
|---|---|
| ELECTRA-base (full dataset, full training) | 0.347 |
| RoBERTa-base (best in paper) | 0.348 |
| Llama-3-8B zero-shot (10 notes) | 0.038 EM / 0.104 fuzzy |
| Llama-3-8B one-shot (10 notes) | 0.048 EM / 0.179 fuzzy |

**Why our results are lower**: Due to computing limitations, this implementation uses only 5 trianing epochs and doesn't train on the full 451-note dataset (some were inaccessible in the MIMIC-III extract). The paper also used Llama-3-8B for its LLM baseline, which is a significantly stronger instruction-following model than the seq2seq FLAN-T5-xl with 3B parameters.

---

## Reference

Elgaar, M., Cheng, J., Vakil, N., Amiri, H., & Celi, L. A. (2024). MedDec: A Dataset for Extracting Medical Decisions from Discharge Summaries. NeurIPS 2024. https://arxiv.org/abs/2408.12980
