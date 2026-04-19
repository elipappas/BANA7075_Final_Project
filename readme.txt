# Pneumonia Detection via CNN — BANA 7075 Final Project

A computer vision system that analyzes pediatric chest X-rays to detect pneumonia using a Convolutional Neural Network (CNN). Built as part of BANA 7075 (Spring 2026) by Leah Domos, Arshia Ghasemi, Owen Montgomery, Eli Pappas, and Emily Sullivan.

## Motivation

Pneumonia is a leading cause of death in children under 5 years old worldwide. Human error rates in chest X-ray interpretation can reach 10–20%, and delays caused by manual review slow down treatment. This project aims to:

- **Expedite diagnosis** — give radiologists faster, automated preliminary results.
- **Minimize human error** — reduce misdiagnoses, with a particular focus on lowering false negatives.

## Dataset

**Chest X-Ray Images (Pneumonia)** from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), originally sourced from Guangzhou Women and Children’s Medical Center.

- 5,863 chest X-ray images from children ages 1–5
- Binary labels: `NORMAL` / `PNEUMONIA`
- Pre-split into `train/`, `val/`, and `test/` directories

**Setup Instructions:**
1. Download the dataset from Kaggle.
2. Place the downloaded data under a `data/` directory at the project root:

```bash
data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/

Model Architecture
A sequential CNN built with TensorFlow/Keras:
Layer
Details
Conv2D
32 filters, 3×3 kernel, ReLU, input 224×224×3
MaxPooling2D
Pool size 2, stride 2
Flatten
—
Dense
128 units, ReLU
Dropout
0.5 rate
Dense (Output)
1 unit, Sigmoid

Loss: Binary Cross-Entropy
Optimizer: Adam
Input size: 224×224 RGB

Training Augmentation
Images are augmented during training with random rotation (±10°), zoom (10%), and horizontal flips, then normalized to [0, 1].
Project Structure
.
├── Image_Preparation.py      # Robust data loading with configurable DATA_DIR and clear error messages
├── utils.py                  # Shared preprocessing for inference (consistent with training)
├── CNN_training.py           # Builds, trains, and saves the CNN model
├── CNN_testing.py            # Evaluates the model on the test set
├── ui.py                     # Streamlit web app for live predictions
├── full_run.sh               # Shell script to train then test in one command
├── xray_cnn_model.keras      # Trained model (generated after training)
├── notebooks/                # Exploratory notebooks
└── data/                     # Dataset folder (not included in repo)

Installation
Requirements: Python 3.9+
pip install tensorflow streamlit pillow numpy mlflow scikit-learn matplotlib

Usage
Train the model
python CNN_training.py

Evaluate on the test set
python CNN_testing.py

Train and test in one step
bash full_run.sh

Run the Streamlit UI
streamlit run ui.py

Upload a chest X-ray image (JPG or PNG). The app will display the model's prediction: Normal or Pneumonia, along with confidence score.
Results
Adam optimizer achieved superior performance with:
	•	Accuracy: ~0.86
	•	Strong Pneumonia recall (critical for medical screening)
MLflow was used to track experiments and register the best model.
Roadmap
	•	Track full suite of metrics (F1, Recall, Precision) across runs
	•	Experiment with additional model architectures (e.g., transfer learning)
	•	Improve data validation and versioning
	•	Enhance UI for better clinical usability

Team
Name

Leah Domos

Arshia Ghasemi

Owen Montgomery

Eli Pappas

Emily Sullivan




