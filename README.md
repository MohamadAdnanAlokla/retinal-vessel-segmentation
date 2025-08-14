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

![Uploading image.png…]()
