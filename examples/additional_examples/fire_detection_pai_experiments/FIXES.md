# FIXES.md — fire_detection_pai_experiments

## 1. Missing `.npz` file

`fire_dendrite_vs_baseline.ipynb` line 17 loads
`../data/processed/modis_firms_train_val_test_dataset.npz`, but this file doesn't exist in the repo. `../data/processed/` also doesn't seem to be a valid path.