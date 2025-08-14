# retinal-vessel-segmentation
Proposed model for retinal blood vessel segmentation using DRIVE dataset and Gradio interface and depend on U-Net.
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
<img width="1058" height="647" alt="image" src="https://github.com/user-attachments/assets/fb3fae85-2222-4254-b622-5a1a4f4b4cfb" />

## Dataset

The retinal images dataset is publicly available on Kaggle:

[Retina Blood Vessel Dataset](https://www.kaggle.com/datasets/abdallahwagih/retina-blood-vessel)  

You can download it and use it to train or test the model.

---

## Demo

<img width="1918" height="923" alt="Result" src="https://github.com/user-attachments/assets/a17feee4-1c84-406d-9574-1dc0e412d35c" />


---

## Requirements

Python **3.11** must be installed on your system.

All required packages are listed in `requirements.txt`:

```text
torch
torchvision
numpy
opencv-python
Pillow
gradio

##How to Run Locally
###Clone or download the repository:
git clone https://github.com/MohamadAdnanAlokla/retinal-vessel-segmentation.git
cd retinal-vessel-segmentation

###Create a virtual environment:
python -m venv venv

###Activate the environment:

Windows PowerShell:.\venv\Scripts\Activate.ps1

###Install required packages

pip install -r requirements.txt

###Run the Gradio app:
python app.py

