# Pneumonia Detection via CNN — BANA 7075 Final Project

A computer vision system that analyzes pediatric chest X-rays to detect pneumonia using a Convolutional Neural Network (CNN). Built as part of BANA 7075 (Spring 2026) by Leah Domos, Arshia Ghasemi, Owen Montgomery, Eli Pappas, and Emily Sullivan.

---

## Motivation

Pneumonia is a leading cause of death in children under 5 years old worldwide. Human error rates in chest X-ray interpretation can reach 10–20%, and delays caused by manual review slow down treatment. This project aims to:

- **Expedite diagnosis** — give radiologists faster, automated preliminary results.
- **Minimize human error** — reduce misdiagnoses, with a particular focus on lowering false negatives.

---

## Dataset

**Chest X-Ray Images (Pneumonia)** from [Kaggle], originally sourced from Guangzhou Women and Children's Medical Center.

- 8,530 chest X-ray images from children ages 1–5
- Binary labels: `NORMAL` / `PNEUMONIA`
- Pre-split into `train/`, `val/`, and `test/` directories

Place the downloaded data under a `data/` directory at the project root:

```
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
```

---

## Model Architecture

A sequential CNN built with TensorFlow/Keras:

| Layer | Details |
|---|---|
| Conv2D | 32 filters, 3×3 kernel, ReLU, input 224×224×3 |
| MaxPooling2D | Pool size 2, stride 2 |
| Flatten | — |
| Dense | 128 units, ReLU |
| Dropout | 0.5 rate |
| Dense (output) | 1 unit, Sigmoid |

- **Loss:** Binary Cross-Entropy  
- **Optimizer:** Adam (primary) / SGD (comparison)  
- **Input size:** 224×224 RGB

### Training Augmentation

Images are augmented during training with random rotation (±10°), zoom (10%), and horizontal flips, then normalized to [0, 1].

---

## Results

Two optimizers were compared:

| Optimizer | Accuracy | Normal F1 | Pneumonia F1 | Pneumonia Recall |
|---|---|---|---|---|
| **Adam** | **0.86** | 0.86 | 0.87 | **0.91** |
| SGD | 0.80 | 0.77 | 0.83 | 0.95 |

Adam was selected as the primary optimizer for its superior overall accuracy. MLflow was used to track, compare, and version model runs, automatically aliasing the best-performing model as `@champion`.

---

## Project Structure

```
.
├── CNN_training.py        # Trains the CNN and saves the model
├── CNN_testing.py         # Evaluates the saved model on the test set
├── Image_Preparation.py   # Data generators for train/val/test splits
├── utils.py               # Image preprocessing utility for inference
├── ui.py                  # Streamlit web app for live predictions
├── full_run.sh            # Shell script to train then test in one command
├── notebooks/
│   ├── CNN_v1.ipynb                   # Initial prototype notebook
│   ├── cnn_mlflow_versioning.ipynb    # MLflow experiment tracking & model versioning
│   └── testing.ipynb                  # Exploratory testing notebook
└── data/                  # Dataset directory (not included — see Dataset section)
```

---

## Installation

**Requirements:** Python 3.9+

```bash
pip install tensorflow streamlit pillow numpy mlflow scikit-learn matplotlib
```

---

## Usage

### Train the model

```bash
python3 CNN_training.py
```

This trains the CNN for 5 epochs and saves the model to `xray_cnn_model.keras`.

### Evaluate on the test set

```bash
python3 CNN_testing.py
```

### Train and test in one step

```bash
bash full_run.sh
```

### Run the Streamlit UI

```bash
python -m streamlit run ui.py
```

Upload a chest X-ray image (JPG or PNG) and the app will display the model's prediction: **Normal** or **Pneumonia**.

### MLflow experiment tracking

Open `notebooks/cnn_mlflow_versioning.ipynb` to run both Adam and SGD experiments, log metrics, and register the best model automatically.

---

## Roadmap

- [ ] Track full suite of metrics (F1, Recall, Precision) across runs
- [ ] Experiment with additional model architectures (e.g., transfer learning)
- [ ] More robust data validation and versioning
- [ ] UI enhancements for clinical usability
- [ ] Integration with hospital imaging systems for automated inference on upload

---

## Team

| Name | Role |
|---|---|
| Leah Domos | — |
| Arshia Ghasemi | — |
| Owen Montgomery | — |
| Eli Pappas | — |
| Emily Sullivan | — |

*BANA 7075 — Spring 2026*
