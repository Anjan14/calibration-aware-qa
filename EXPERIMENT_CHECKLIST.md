# Reproducibility checklist - generated suggestion

## Environment

- [ ] Python version recorded (e.g. `python --version`).
- [ ] Fixed dependency set: `pip freeze > requirements_frozen.txt` after a clean install.
- [ ] CUDA / device noted in the paper appendix (`TrainingConfig.device`, GPU model if any).
- [ ] Random seeds listed: data order shuffling uses `TrainingConfig.seed`; multi-seed runs use `run_experiment.py`.

## Data

- [ ] Dataset: SQuAD 2.0 via `datasets.load_dataset("squad_v2")` or local paths documented.
- [ ] Train/validation split unchanged from official SQuAD 2.0 dev for reporting dev scores.

## Model and training

- [ ] Encoder name (default `bert-base-uncased`) and `lambda_answerability` reported.
- [ ] Hyperparameters match `README.md` defaults or deviations are stated (LR, batch sizes, epochs, max length, stride).
- [ ] Checkpoint path for each reported row (e.g. `outputs/checkpoints/epoch_3`).

## Metrics reported in the paper

- [ ] **EM / F1** (normalized as in `src/evaluate.py`).
- [ ] **Answerability accuracy** (predicted empty vs non-empty vs gold).
- [ ] **ECE, Brier, AURC** from `calibration_metrics` in `training_summary.json` or eval JSON (see below).
- [ ] **Bootstrap CIs**: from `run_eval.py` (`bootstrap_calibrated`) or `run_paper_report.py` (`paper_report.json`).
- [ ] **Multi-seed mean ± std**: from `outputs/multi_seed/multi_seed_summary.json` after `run_experiment.py`.

## Files to archive per run

- [ ] `training_summary.json` (per run).
- [ ] `epoch_*_per_example.json` (for bootstrap and calibration).
- [ ] `eval_results.json` + `*_per_example.json` from `run_eval.py` if eval-only.
- [ ] `paper_report.json`, `metrics_table.csv`, `metrics_table.tex` from `run_paper_report.py`.

## Statistical comparison

- [ ] For two systems on the **same** validation examples, use paired bootstrap (`run_paper_report.py --compare-per-example`).
- [ ] Confirm both `*_per_example.json` files align by construction (same dataset, same order).

