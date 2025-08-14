# retinal-vessel-segmentation
U-Net model for retinal blood vessel segmentation using DRIVE dataset and Gradio interface.
# Retinal Blood Vessel Segmentation - Model Comparison

This project segments retinal blood vessels using a U-Net based model. Below is a comparison of different models on the **DRIVE dataset**.

---

## Table: Performance Comparison

| Model                   | Accuracy (Acc) | Sensitivity (Se) | Specificity (Sp) | AUC    | F1-Score |
|-------------------------|----------------|-----------------|-----------------|--------|----------|
| U-Net                   | 0.9739         | 0.7512          | 0.9879          | 0.9851 | 0.7933   |
| Sine-Net                | 0.9712         | 0.8788          | 0.9803          | 0.9910 | N/A      |
| RFARN (ref 30)          | 0.9712         | 0.8453          | N/A             | N/A    | 0.8453   |
| BLCB-CNN                | 0.9622         | 0.8157          | 0.9765          | 0.9823 | N/A      |
| Dense-U-Net (ref 31)    | 0.9559         | N/A             | N/A             | 0.9793 | N/A      |
| Ensemble Meta-Model (3) | 0.9778         | 0.7790          | 0.9920          | 0.9912 | 0.8231   |
| Proposed Model          | 0.9601         | 0.9188          | 0.9625          | 0.9406 | 0.7097   |

---

## Visual Comparison (Example)

We can plot **Accuracy, Sensitivity, and F1-Score** for a quick comparison:

```python
import matplotlib.pyplot as plt
import numpy as np

models = ["U-Net", "Sine-Net", "RFARN", "BLCB-CNN", "Dense-U-Net", "Ensemble", "Proposed"]
accuracy = [0.9739, 0.9712, 0.9712, 0.9622, 0.9559, 0.9778, 0.9601]
sensitivity = [0.7512, 0.8788, 0.8453, 0.8157, 0, 0.7790, 0.9188]  # use 0 for N/A
f1_score = [0.7933, 0, 0.8453, 0, 0, 0.8231, 0.7097]  # use 0 for N/A

x = np.arange(len(models))
width = 0.25

plt.bar(x - width, accuracy, width, label='Accuracy')
plt.bar(x, sensitivity, width, label='Sensitivity')
plt.bar(x + width, f1_score, width, label='F1-Score')

plt.xticks(x, models, rotation=30)
plt.ylabel("Score")
plt.title("Model Performance Comparison on DRIVE Dataset")
plt.legend()
plt.tight_layout()
plt.savefig("examples/model_comparison.png")
plt.show()
