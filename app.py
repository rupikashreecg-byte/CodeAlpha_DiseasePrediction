```python
"""
CodeAlpha ML Internship — Task 4: Disease Prediction from Medical Data
Author: Rupika Shree
GitHub Repo: CodeAlpha_DiseasePrediction

Algorithms Used:
- Logistic Regression
- SVM
- Random Forest
- XGBoost

Datasets:
- Heart Disease (UCI)
- Diabetes (Pima Indians)
- Breast Cancer (Wisconsin)

Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
"""

import io
import random
import warnings

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

matplotlib.use("Agg")

np.random.seed(42)
random.seed(42)


# ─────────────────────────────────────────────
# SECTION 1: LOAD DATASETS
# ─────────────────────────────────────────────

def load_heart_disease():
    """Load UCI Heart Disease dataset."""
    
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "heart-disease/processed.cleveland.data"
    )

    cols = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak",
        "slope", "ca", "thal", "target"
    ]

    df = pd.read_csv(url, names=cols, na_values="?")

    df.dropna(inplace=True)

    # Convert to binary classification
    df["target"] = (df["target"] > 0).astype(int)

    X = df.drop("target", axis=1)
    y = df["target"]

    return X, y, cols[:-1]


def load_diabetes():
    """Load Pima Indians Diabetes dataset."""

    url = (
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
        "pima-indians-diabetes.data.csv"
    )

    cols = [
        "pregnancies", "glucose", "bloodpressure",
        "skinthickness", "insulin", "bmi",
        "dpf", "age", "target"
    ]

    df = pd.read_csv(url, names=cols)

    # Replace invalid zero values with median values
    zero_cols = [
        "glucose",
        "bloodpressure",
        "skinthickness",
        "insulin",
        "bmi",
    ]

    for col in zero_cols:
        df[col] = df[col].replace(0, df[col].median())

    X = df.drop("target", axis=1)
    y = df["target"]

    return X, y, cols[:-1]


def load_breast_cancer_data():
    """Load Wisconsin Breast Cancer dataset."""

    data = load_breast_cancer()

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    return X, y, list(data.feature_names)


# ─────────────────────────────────────────────
# SECTION 2: BUILD MODELS
# ─────────────────────────────────────────────

def build_models():
    """Create all classification models."""

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=42,
        ),

        "SVM (RBF Kernel)": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=42,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        ),
    }


# ─────────────────────────────────────────────
# SECTION 3: TRAIN & EVALUATE
# ─────────────────────────────────────────────

def evaluate_models(X, y):
    """Train and evaluate all models."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = build_models()

    records = []
    trained_models = {}

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    for name, model in models.items():

        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        cv_accuracy = cross_val_score(
            model,
            X_train_scaled,
            y_train,
            cv=cv,
            scoring="accuracy",
        ).mean()

        records.append({
            "Model": name,
            "Accuracy": round(
                accuracy_score(y_test, y_pred) * 100,
                2,
            ),
            "Precision": round(
                precision_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                ) * 100,
                2,
            ),
            "Recall": round(
                recall_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                ) * 100,
                2,
            ),
            "F1-Score": round(
                f1_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                ) * 100,
                2,
            ),
            "ROC-AUC": round(
                roc_auc_score(y_test, y_prob),
                4,
            ),
            "CV-Accuracy": round(
                cv_accuracy * 100,
                2,
            ),
        })

        trained_models[name] = (model, scaler)

    results_df = pd.DataFrame(records).set_index("Model")

    return results_df, trained_models


# ─────────────────────────────────────────────
# SECTION 4: VISUALIZATIONS
# ─────────────────────────────────────────────

def fig_to_pil(fig):
    """Convert matplotlib figure to PIL image."""

    buffer = io.BytesIO()

    fig.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )

    buffer.seek(0)

    plt.close(fig)

    return Image.open(buffer)


def plot_metrics(results_df):
    """Create model comparison charts."""

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
    ]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 5),
        facecolor="#0d1520",
    )

    colors = [
        "#00d4ff",
        "#00ff9d",
        "#ffb700",
        "#c084fc",
    ]

    for ax, metric in zip(axes, metrics):

        values = results_df[metric].values

        bars = ax.bar(
            range(len(results_df)),
            values,
            color=colors,
            width=0.6,
        )

        ax.set_facecolor("#1a2332")

        ax.set_ylim(0, 105)

        ax.set_title(
            metric,
            color="#b8cfe0",
            fontsize=11,
            fontweight="bold",
        )

        ax.set_xticks(range(len(results_df)))

        ax.set_xticklabels(
            ["LR", "SVM", "RF", "XGB"],
            color="#8fa3bc",
        )

        ax.tick_params(colors="#8fa3bc")

        for bar, value in zip(bars, values):

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{value:.1f}",
                ha="center",
                color="white",
                fontsize=8,
            )

    fig.suptitle(
        "Model Comparison",
        color="#00d4ff",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    return fig_to_pil(fig)


def plot_feature_importance(trained_models, feature_names):
    """Plot Random Forest feature importance."""

    rf_model, _ = trained_models["Random Forest"]

    importances = rf_model.feature_importances_

    indices = np.argsort(importances)[::-1][:10]

    fig, ax = plt.subplots(
        figsize=(10, 5),
        facecolor="#0d1520",
    )

    ax.set_facecolor("#1a2332")

    ax.barh(
        range(len(indices)),
        importances[indices][::-1],
        color="#00d4ff",
    )

    ax.set_yticks(range(len(indices)))

    ax.set_yticklabels(
        [feature_names[i] for i in indices[::-1]],
        color="#b8cfe0",
    )

    ax.set_title(
        "Top Feature Importances",
        color="#00d4ff",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()

    return fig_to_pil(fig)


# ─────────────────────────────────────────────
# SECTION 5: TRAIN MODELS
# ─────────────────────────────────────────────

print("Training models...\n")

DATASETS = {
    "Heart Disease (UCI Cleveland)": load_heart_disease,
    "Diabetes (Pima Indians)": load_diabetes,
    "Breast Cancer (Wisconsin)": load_breast_cancer_data,
}

TRAINED = {}
METRIC_IMAGES = {}
FEATURE_IMAGES = {}

for dataset_name, loader_function in DATASETS.items():

    print(f"Training on {dataset_name}...")

    X, y, feature_names = loader_function()

    results_df, trained_models = evaluate_models(X, y)

    TRAINED[dataset_name] = (
        results_df,
        trained_models,
        feature_names,
    )

    METRIC_IMAGES[dataset_name] = plot_metrics(results_df)

    FEATURE_IMAGES[dataset_name] = plot_feature_importance(
        trained_models,
        feature_names,
    )

print("\nAll models trained successfully.\n")


# ─────────────────────────────────────────────
# SECTION 6: INPUT FIELDS
# ─────────────────────────────────────────────

HEART_FIELDS = [
    ("Age", 55, 20, 80),
    ("Sex (1=Male, 0=Female)", 1, 0, 1),
    ("Chest Pain Type", 2, 0, 3),
    ("Resting Blood Pressure", 130, 80, 200),
    ("Cholesterol", 240, 100, 600),
    ("Fasting Blood Sugar", 0, 0, 1),
    ("Resting ECG", 1, 0, 2),
    ("Maximum Heart Rate", 150, 70, 210),
    ("Exercise Angina", 0, 0, 1),
    ("ST Depression", 1.0, 0.0, 6.2),
    ("Slope", 1, 0, 2),
    ("Major Vessels", 0, 0, 3),
    ("Thal (1=normal, 2=fixed, 3=reversible)", 2, 1, 3),
]

DIABETES_FIELDS = [
    ("Pregnancies", 2, 0, 17),
    ("Glucose", 120, 40, 200),
    ("Blood Pressure", 70, 0, 130),
    ("Skin Thickness", 25, 0, 100),
    ("Insulin", 80, 0, 900),
    ("BMI", 28.5, 10.0, 70.0),
    ("Diabetes Pedigree Function", 0.45, 0.07, 2.5),
    ("Age", 35, 21, 81),
]

CANCER_FIELDS = [
    ("Mean Radius", 14.5, 5.0, 30.0),
    ("Mean Texture", 19.0, 8.0, 40.0),
    ("Mean Perimeter", 95.0, 40.0, 200.0),
    ("Mean Area", 650.0, 100.0, 2500.0),
    ("Mean Smoothness", 0.10, 0.05, 0.20),
]

DATASET_FIELDS = {
    "Heart Disease (UCI Cleveland)": HEART_FIELDS,
    "Diabetes (Pima Indians)": DIABETES_FIELDS,
    "Breast Cancer (Wisconsin)": CANCER_FIELDS,
}

MAX_INPUTS = 30


# ─────────────────────────────────────────────
# SECTION 7: PREDICTION FUNCTION
# ─────────────────────────────────────────────

def predict(dataset, *inputs):

    results_df, trained_models, feature_names = TRAINED[dataset]

    values = list(inputs[:len(feature_names)])

    X_input = np.array(values).reshape(1, -1)

    rows = []

    for model_name, (model, scaler) in trained_models.items():

        X_scaled = scaler.transform(X_input)

        prediction = model.predict(X_scaled)[0]

        probability = model.predict_proba(X_scaled)[0][1]

        label = (
            "⚠ HIGH RISK"
            if prediction == 1
            else "✅ LOW RISK"
        )

        rows.append({
            "Algorithm": model_name,
            "Prediction": label,
            "Confidence": f"{probability * 100:.1f}%",
        })

    prediction_df = pd.DataFrame(rows)

    metrics_df = results_df.reset_index()

    return (
        prediction_df,
        METRIC_IMAGES[dataset],
        FEATURE_IMAGES[dataset],
        metrics_df,
    )


# ─────────────────────────────────────────────
# SECTION 8: DYNAMIC INPUTS
# ─────────────────────────────────────────────

def update_inputs(dataset):

    fields = DATASET_FIELDS[dataset]

    updates = []

    for i in range(MAX_INPUTS):

        if i < len(fields):

            label, default, minimum, maximum = fields[i]

            updates.append(
                gr.Number(
                    value=default,
                    label=label,
                    visible=True,
                    minimum=minimum,
                    maximum=maximum,
                )
            )

        else:
            updates.append(gr.Number(visible=False))

    return updates


# ─────────────────────────────────────────────
# SECTION 9: GRADIO UI
# ─────────────────────────────────────────────

with gr.Blocks(
    title="MedAI — Disease Prediction",
    theme=gr.themes.Base(
        primary_hue="cyan",
        secondary_hue="slate",
    ),
) as demo:

    gr.Markdown("""
    # 🩺 MedAI — Disease Prediction System

    ### CodeAlpha ML Internship — Task 4

    Compare multiple machine learning models for disease prediction using:
    - Heart Disease dataset
    - Diabetes dataset
    - Breast Cancer dataset
    """)

    dataset_dropdown = gr.Dropdown(
        choices=list(DATASETS.keys()),
        value="Heart Disease (UCI Cleveland)",
        label="Select Dataset",
    )

    input_components = []

    with gr.Row():

        with gr.Column():

            gr.Markdown("### Patient Parameters")

            for i in range(MAX_INPUTS):

                if i < len(HEART_FIELDS):

                    label, default, minimum, maximum = HEART_FIELDS[i]

                    component = gr.Number(
                        value=default,
                        label=label,
                        minimum=minimum,
                        maximum=maximum,
                    )

                else:
                    component = gr.Number(visible=False)

                input_components.append(component)

        with gr.Column():

            run_button = gr.Button(
                "Run Prediction",
                variant="primary",
            )

            prediction_table = gr.DataFrame(
                label="Prediction Results",
                interactive=False,
            )

            metrics_table = gr.DataFrame(
                label="Model Evaluation Metrics",
                interactive=False,
            )

    with gr.Row():

        metrics_plot = gr.Image(
            label="Metrics Comparison",
            type="pil",
        )

        feature_plot = gr.Image(
            label="Feature Importance",
            type="pil",
        )

    dataset_dropdown.change(
        fn=update_inputs,
        inputs=[dataset_dropdown],
        outputs=input_components,
    )

    run_button.click(
        fn=predict,
        inputs=[dataset_dropdown] + input_components,
        outputs=[
            prediction_table,
            metrics_plot,
            feature_plot,
            metrics_table,
        ],
    )

    gr.Markdown("""
    ---
    ⚠ This project is for educational purposes only and should not
    replace professional medical diagnosis.
    """)

if __name__ == "__main__":
    demo.launch()
```
