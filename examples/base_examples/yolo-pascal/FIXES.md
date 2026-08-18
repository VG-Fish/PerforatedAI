# PAI-YOLO (PASCAL VOC Data Efficiency) — Fixes
 
## 1. `perforatedai` package missing from requirements.txt
## 2. `requirements.txt` includes unrelated packages
**Fix** Condensed requirements.txt and removed exact pins
'''
numpy
opencv-python
pillow
matplotlib
PyYAML
wandb
tqdm
ultralytics
'''