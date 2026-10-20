# Calibration-Aware Extractive Reading Comprehension QA on SQuAD 2.0

This project implements a modular extractive QA system with:

- one shared `bert-base-uncased` encoder
- a span prediction head (`start_logits`, `end_logits`)
- an answerability head (binary prediction from `[CLS]`)
- a calibration-aware inference rule to decide between span vs **No Answer**

The codebase is designed for graduate NLP project work: clear architecture, manual training loop, explicit preprocessing, and reproducible outputs.

## Project Goal

Evaluate whether adding answerability-aware calibration improves reliability for SQuAD 2.0 extractive QA, especially no-answer handling.

## Architecture

- **Encoder**: Hugging Face `AutoModel` (default `bert-base-uncased`)
- **Span Head**: linear layer over token embeddings for start/end logits
- **Answerability Head**: linear classifier over `[CLS]`
- **Training Objective**:
  - `total_loss = start_loss + end_loss + lambda * answerability_loss`
  - default `lambda = 0.5`
- **Calibration Rule**:
  - `calibrated_span_score = best_span_score * answerability_prob`
  - `calibrated_null_score = null_score * (1 - answerability_prob)`
  - if `(calibrated_null_score - calibrated_span_score) > threshold`, return `""`

## Project Structure

```text
qa_project/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_error_analysis.ipynb
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   ├── plots/
│   └── predictions/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── data.py
│   ├── preprocess.py
│   ├── model.py
│   ├── calibrate.py
│   ├── evaluate.py
│   ├── baselines.py
│   ├── train.py
│   ├── analysis.py
│   ├── calibration_metrics.py
│   ├── stats_utils.py
│   ├── publication_tables.py
│   └── paper_report.py
├── run_train.py
├── run_eval.py
├── run_experiment.py
├── run_paper_report.py
├── requirements.txt
├── EXPERIMENT_CHECKLIST.md
└── README.md
```

## Prerequisites

- Python 3.10+
- Pip
- (Recommended) virtual environment

## Installation

### Windows (PowerShell)

```powershell
cd C:\path\to\qa_project
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux (bash)

```bash
cd /path/to/qa_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### macOS (zsh/bash)

```bash
cd /path/to/qa_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Usage

### Option A: Hosted SQuAD 2.0 (default)

No local dataset paths needed.

```bash
python run_train.py --output-dir outputs
```

### Option B: Local SQuAD JSON paths

Replace placeholders with your real paths.

- Windows example:
  - `C:\\Users\\YourName\\data\\train-v2.0.json`
  - `C:\\Users\\YourName\\data\\dev-v2.0.json`
- Linux example:
  - `/home/yourname/data/train-v2.0.json`
  - `/home/yourname/data/dev-v2.0.json`
- macOS example:
  - `/Users/yourname/data/train-v2.0.json`
  - `/Users/yourname/data/dev-v2.0.json`

```bash
python run_train.py \
  --train-file /Users/yourname/data/train-v2.0.json \
  --validation-file /Users/yourname/data/dev-v2.0.json \
  --output-dir outputs
```

## Train Command (Example)

```bash
python run_train.py \
  --output-dir outputs \
  --model-name bert-base-uncased \
  --epochs 3 \
  --train-batch-size 8 \
  --eval-batch-size 16 \
  --learning-rate 3e-5
```

## Evaluate Command (Example)

```bash
python run_eval.py \
  --checkpoint outputs/checkpoints/epoch_3 \
  --output-json outputs/final_eval.json
```

Optional: bootstrap confidence intervals on calibrated per-example scores (set `--bootstrap-samples 0` to skip):

```bash
python run_eval.py \
  --checkpoint outputs/checkpoints/epoch_3 \
  --output-json outputs/final_eval.json \
  --bootstrap-samples 10000 \
  --seed 42
```

## Multi-seed runs (mean ± std for papers)

```bash
python run_experiment.py --seeds 42 43 44 --output-dir outputs/multi_seed
```

Aggregated scalars are written to `outputs/multi_seed/multi_seed_summary.json`.

## Publication tables and paired comparison

After you have `epoch_*_per_example.json` (from training) or `*_per_example.json` (from `run_eval`):

```bash
python run_paper_report.py \
  --per-example outputs/predictions/epoch_3_per_example.json \
  --out-dir outputs/paper \
  --label-a Calibrated
```

Paired bootstrap (same validation set, two checkpoints; per-example rows must align—same data order):

```bash
python run_paper_report.py \
  --per-example outputs/run_a/epoch_3_per_example.json \
  --compare-per-example outputs/run_b/epoch_3_per_example.json \
  --label-a Ours \
  --label-b Baseline \
  --out-dir outputs/paper_paired
```

Outputs: `paper_report.json` (bootstrap CIs and optional paired diffs), `metrics_table.csv`, `metrics_table.tex`.

See `EXPERIMENT_CHECKLIST.md` for a reproducibility checklist aimed at camera-ready reporting.

## Expected Outputs

After training, you should see:

- `outputs/checkpoints/epoch_*/model.pt`
- `outputs/logs/train.log`
- `outputs/plots/loss_curve.png`
- `outputs/plots/val_em_curve.png`
- `outputs/plots/val_f1_curve.png`
- `outputs/predictions/epoch_*_predictions.json`
- `outputs/predictions/epoch_*_metadata.json`
- `outputs/predictions/epoch_*_threshold_sweep.json`
- `outputs/predictions/epoch_*_per_example.json` (per-example scores for bootstrap)
- `outputs/training_summary.json` (includes `calibration_metrics` per epoch: ECE, Brier, AURC)

## Evaluation Details

- EM and token-level F1 are implemented manually.
- Text normalization includes lowercase, punctuation removal, article removal, and whitespace normalization.
- Multiple gold answers are supported using max-over-golds scoring.
- Answerability accuracy is reported from no-answer vs answerable decision.
- **Calibration metrics** (validation): expected calibration error (ECE) and Brier score for answerability probabilities vs gold answerable labels; **AURC** is the area under the risk–coverage curve using answerability probability to order examples and EM correctness as risk (see `src/calibration_metrics.py`).

## Baselines

- `first_three_words`: returns first 3 words of context
- `lexical_overlap_sentence`: picks context sentence with highest overlap with question tokens

## Troubleshooting

- **Tokenizer/model download fails**:
  - Retry with stable internet.
  - Optionally set `--cache-dir /your/cache/path`.
- **CUDA not used**:
  - Current config defaults to CPU for portability.
  - You can update `TrainingConfig(device="cuda")` in code if your environment supports it.
- **Out-of-memory**:
  - Reduce `--train-batch-size` and `--eval-batch-size`.
- **Local JSON not loading**:
  - Ensure your files are true SQuAD v2 format and paths are correct.
- **No predictions generated**:
  - Check `outputs/logs/train.log` for exceptions.

## Notes for Course Report / Viva

- Preprocessing in `src/preprocess.py` is intentionally explicit and commented for explainability.
- Model heads and loss composition are fully manual in `src/model.py`.
- Calibration and threshold sweep are separated in `src/calibrate.py`.
- Evaluation/post-processing are explicit in `src/evaluate.py`.

