# CodeAlpha_DiseasePrediction

## Task 4 — Disease Prediction from Medical Data
### CodeAlpha Machine Learning Internship

---

## Overview
This project predicts the likelihood of diseases based on patient medical data using four classification algorithms. Built for the **CodeAlpha ML Internship — Task 4**.

## Algorithms Used
| Algorithm | Library | Notes |
|-----------|---------|-------|
| Logistic Regression | scikit-learn | Baseline classifier |
| SVM (RBF Kernel) | scikit-learn | Good for high-dim data |
| Random Forest | scikit-learn | Ensemble, feature importance |
| XGBoost | xgboost | Gradient boosting, best overall |

## Datasets
| Dataset | Source | Records | Features |
|---------|--------|---------|----------|
| Heart Disease | UCI ML Repository (Cleveland) | 303 | 13 |
| Diabetes | Pima Indians / UCI | 768 | 8 |
| Breast Cancer | Wisconsin / sklearn | 569 | 30 |

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- 5-Fold Cross-Validation

## Setup & Run

```bash
# 1. Clone this repo
git clone https://github.com/rupikashreecg-byte/CodeAlpha_DiseasePrediction.git
cd CodeAlpha_DiseasePrediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py
```

The Gradio app will open in your browser automatically. A public share link is also printed in the terminal.

## Project Structure
```
CodeAlpha_DiseasePrediction/
├── app.py              # Main application (models + Gradio UI)
├── requirements.txt    # Dependencies
└── README.md           # This file
```

## Tech Stack
- Python 3.8+
- scikit-learn
- XGBoost
- Gradio (interactive UI)
- Pandas, NumPy
- Matplotlib

---
*Built by Rupika Shree | CodeAlpha ML Internship*
