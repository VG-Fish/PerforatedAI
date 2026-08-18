# Human Segmentation (UNet Resolution Experiment) — Fixes
 
## 1. whole `requirements.txt` fails to build
**Error:** `RuntimeError: Python version 2.7 or 3.4+ is required` during `setup.py` build.
**Cause:** The pinned version predates modern Python and has no prebuilt wheel for
Python 3.10+.
**Fix:** Drop the exact pins in `requirements.txt` and make consistent with other models:
```
absl-py
numpy
opencv-python
Pillow
matplotlib
tensorboardX
torchsummary
six
```