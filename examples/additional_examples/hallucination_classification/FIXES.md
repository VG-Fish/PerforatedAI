# FIXES.md — hallucination_classification

## 1. No PerforatedAI integration

Neither notebook imports `perforatedai`, and no models are perforated. Therefore, we can't determine whether perforating the two models improves performance.

## 2. Discrepancy between README and script.

The README says train on `pqa_artificial` (~9k), val/test on `pqa_labeled` (~1k).
The BERT notebook instead loads only `pqa_labeled`, subsamples 500 rows, and
splits 80/20 into train/val — no held-out test set at all.