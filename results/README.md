# Training Results — YOLOv8m Fire & Smoke Detection

Fine-tuned on a custom fire/smoke dataset for 50 epochs on NVIDIA T4 GPU.
**Final Performance: 91.4% mAP@0.5**

| File | Description |
|---|---|
| `results.png` | "Box loss (CIoU), Distribution Focal Loss (DFL), and classification loss (BCE) across 50 epochs for train/val sets. Smooth convergence with no divergence confirms augmentation prevented overfitting. |
| `confusion_matrix.png` | True vs predicted labels for FIRE, SMOKE, and background. Strong true-positive diagonal with minimal class confusion. |
| `PR_curve.png` | Precision-recall curve across confidence thresholds, validating the 91.4% mAP@0.5 target. |
