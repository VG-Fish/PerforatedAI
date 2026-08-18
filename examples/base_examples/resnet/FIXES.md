# ResNet + Knowledge Distillation (PerforatedAI) — Fixes
 
## 1. `NameError: name 'UPB' is not defined`
**Error:** Occurs inside `perforatedai/utils_perforatedai.py`, in
`convert_network`, when calling `UPA.perforate_model(model)`.

## 2. Missing `requirements.txt` 
This project has no committed `requirements.txt`. The actual third-party dependency list is
short; (`presets`, `utils`, `sampler`, `transforms`,
`resnet_double`) are local project files, not pip packages:
 
```
torch
torchvision
wandb
```