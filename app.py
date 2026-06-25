"""
CodeAlpha ML Internship — Task 4: Disease Prediction from Medical Data
Author: [Your Name]
GitHub Repo: CodeAlpha_DiseasePrediction

Algorithms Used: Logistic Regression, SVM, Random Forest, XGBoost
Datasets: Heart Disease (UCI), Diabetes (Pima Indians), Breast Cancer (Wisconsin)
Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report,
                              confusion_matrix)
from xgboost import XGBClassifier
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
from PIL import Image

# ─────────────────────────────────────────────
#  SECTION 1: Load & Prepare Datasets
# ─────────────────────────────────────────────

def load_heart_disease():
    """Load UCI Heart Disease (Cleveland) dataset."""
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "heart-disease/processed.cleveland.data")
    cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal","target"]
    df = pd.read_csv(url, names=cols, na_values="?")
    df.dropna(inplace=True)
    df["target"] = (df["target"] > 0).astype(int)   # 0=no disease, 1=disease
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y, cols[:-1]

def load_diabetes():
    """Load Pima Indians Diabetes dataset."""
    url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
           "pima-indians-diabetes.data.csv")
    cols = ["pregnancies","glucose","bloodpressure","skinthickness",
            "insulin","bmi","dpf","age","target"]
    df = pd.read_csv(url, names=cols)
    # Replace biological zeros with column median (standard preprocessing)
    zero_cols = ["glucose","bloodpressure","skinthickness","insulin","bmi"]
    for col in zero_cols:
        df[col] = df[col].replace(0, df[col].median())
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y, cols[:-1]

def load_breast_cancer_data():
    """Load Wisconsin Breast Cancer dataset from sklearn."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)   # 0=malignant, 1=benign
    return X, y, list(data.feature_names)


# ─────────────────────────────────────────────
#  SECTION 2: Train & Evaluate All Models
# ─────────────────────────────────────────────

def build_models():
    """Return dict of all four classifiers with tuned hyperparameters."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0,
                                                   solver="lbfgs",
                                                   random_state=42),
        "SVM (RBF Kernel)":    SVC(kernel="rbf", C=1.0, gamma="scale",
                                   probability=True, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100,
                                                      max_depth=None,
                                                      random_state=42),
        "XGBoost":             XGBClassifier(n_estimators=100, learning_rate=0.1,
                                             use_label_encoder=False,
                                             eval_metric="logloss",
                                             random_state=42)
    }


def evaluate_models(X, y):
    """
    Train/test split + 5-fold cross-validation for every model.
    Returns a results DataFrame and trained model objects.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models  = build_models()
    records = []
    trained = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        y_pred  = model.predict(X_test_sc)
        y_proba = model.predict_proba(X_test_sc)[:, 1]

        cv_acc = cross_val_score(model, X_train_sc, y_train,
                                 cv=cv, scoring="accuracy").mean()

        records.append({
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
            "Recall":    round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
            "F1-Score":  round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
            "ROC-AUC":   round(roc_auc_score(y_test, y_proba), 4),
            "CV-Acc":    round(cv_acc * 100, 2),
        })
        trained[name] = (model, scaler)

    results_df = pd.DataFrame(records).set_index("Model")
    return results_df, trained, X_test_sc, y_test


# ─────────────────────────────────────────────
#  SECTION 3: Visualisations
# ─────────────────────────────────────────────

def plot_metrics(results_df):
    """Bar chart comparing all models across key metrics."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), facecolor="#0d1520")
    colors = ["#00d4ff", "#00ff9d", "#ffb700", "#c084fc"]

    for ax, metric in zip(axes, metrics):
        vals = results_df[metric].values
        bars = ax.bar(range(len(results_df)), vals, color=colors, width=0.6)
        ax.set_facecolor("#1a2332")
        ax.set_ylim(0, 105)
        ax.set_title(metric, color="#b8cfe0", fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(results_df)))
        ax.set_xticklabels(["LR", "SVM", "RF", "XGB"],
                           color="#8fa3bc", fontsize=9)
        ax.set_ylabel("%", color="#8fa3bc", fontsize=9)
        ax.tick_params(colors="#8fa3bc")
        for spine in ax.spines.values():
            spine.set_edgecolor("#243040")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom",
                    color="white", fontsize=8)
        ax.yaxis.label.set_color("#8fa3bc")

    fig.suptitle("Model Comparison — All Metrics", color="#00d4ff",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    return _fig_to_pil(fig)


def plot_feature_importance(trained, feature_names):
    """Feature importance from Random Forest."""
    rf_model, _ = trained["Random Forest"]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]   # top-10

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0d1520")
    ax.set_facecolor("#1a2332")
    bars = ax.barh(range(len(indices)),
                   importances[indices][::-1],
                   color="#00d4ff", alpha=0.85)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]],
                       color="#b8cfe0", fontsize=9)
    ax.set_xlabel("Importance Score", color="#8fa3bc")
    ax.set_title("Top-10 Feature Importances (Random Forest)",
                 color="#00d4ff", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#8fa3bc")
    for spine in ax.spines.values():
        spine.set_edgecolor("#243040")
    plt.tight_layout()
    return _fig_to_pil(fig)


def _fig_to_pil(fig):
    """Convert matplotlib figure to PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)


# ─────────────────────────────────────────────
#  SECTION 4: Gradio Interface
# ─────────────────────────────────────────────

print("Loading datasets and training models — please wait...\n")

# Train all models at startup
DATASETS = {
    "Heart Disease (UCI Cleveland)": load_heart_disease,
    "Diabetes (Pima Indians)":        load_diabetes,
    "Breast Cancer (Wisconsin)":      load_breast_cancer_data,
}

TRAINED   = {}   # {dataset_name: (results_df, trained_models, X_test, y_test, feature_names)}
METRIC_IMGS = {}
FEAT_IMGS   = {}

for ds_name, loader_fn in DATASETS.items():
    print(f"  Training on {ds_name}...")
    X, y, feat_names = loader_fn()
    results_df, trained, X_test, y_test = evaluate_models(X, y)
    TRAINED[ds_name]    = (results_df, trained, X_test, y_test, feat_names)
    METRIC_IMGS[ds_name] = plot_metrics(results_df)
    FEAT_IMGS[ds_name]   = plot_feature_importance(trained, feat_names)
    print(results_df.to_string())
    print()

print("All models trained!\n")


# ── Gradio callback ──────────────────────────

def predict(dataset, *inputs):
    """Run all four models on user-provided inputs and return results."""
    results_df, trained, _, _, feat_names = TRAINED[dataset]

    # Build input array
    values = list(inputs[:len(feat_names)])
    X_input = np.array(values, dtype=float).reshape(1, -1)

    rows = []
    for model_name, (model, scaler) in trained.items():
        X_sc   = scaler.transform(X_input)
        pred   = model.predict(X_sc)[0]
        prob   = model.predict_proba(X_sc)[0][1]
        label  = "⚠ HIGH RISK" if pred == 1 else "✅ LOW RISK"
        rows.append({
            "Algorithm":   model_name,
            "Prediction":  label,
            "Confidence":  f"{prob*100:.1f}%",
        })

    pred_df = pd.DataFrame(rows)
    metrics = results_df[["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]]
    return (pred_df,
            METRIC_IMGS[dataset],
            FEAT_IMGS[dataset],
            metrics.reset_index())


# ── Dynamic input fields (per dataset) ───────

HEART_FIELDS = [
    ("Age (years)", 55, 20, 80),
    ("Sex (1=Male, 0=Female)", 1, 0, 1),
    ("Chest Pain Type (0–3)", 2, 0, 3),
    ("Resting BP (mmHg)", 130, 80, 200),
    ("Cholesterol (mg/dl)", 240, 100, 600),
    ("Fasting Blood Sugar >120 (1/0)", 0, 0, 1),
    ("Resting ECG (0–2)", 1, 0, 2),
    ("Max Heart Rate", 150, 70, 210),
    ("Exercise Angina (1/0)", 0, 0, 1),
    ("ST Depression (oldpeak)", 1.0, 0.0, 6.2),
    ("Slope (0–2)", 1, 0, 2),
    ("Major Vessels (0–3)", 0, 0, 3),
    ("Thal (1=normal,2=fixed,3=reversable)", 2, 1, 3),
]
DIAB_FIELDS = [
    ("Pregnancies", 2, 0, 17),
    ("Glucose (mg/dl)", 120, 40, 200),
    ("Blood Pressure (mmHg)", 70, 0, 130),
    ("Skin Thickness (mm)", 25, 0, 100),
    ("Insulin (mu U/ml)", 80, 0, 900),
    ("BMI", 28.5, 10.0, 70.0),
    ("Diabetes Pedigree Function", 0.45, 0.07, 2.5),
    ("Age (years)", 35, 21, 81),
]
CANCER_FIELDS = [
    ("Mean Radius", 14.5, 5.0, 30.0),
    ("Mean Texture", 19.0, 8.0, 40.0),
    ("Mean Perimeter", 95.0, 40.0, 200.0),
    ("Mean Area", 650.0, 100.0, 2500.0),
    ("Mean Smoothness", 0.10, 0.05, 0.20),
    ("Mean Compactness", 0.12, 0.01, 0.35),
    ("Mean Concavity", 0.10, 0.00, 0.50),
    ("Mean Concave Points", 0.06, 0.00, 0.20),
    ("Mean Symmetry", 0.18, 0.10, 0.30),
    ("Mean Fractal Dimension", 0.063, 0.040, 0.100),
    ("SE Radius", 0.4, 0.1, 3.0),
    ("SE Texture", 1.2, 0.3, 5.0),
    ("SE Perimeter", 2.8, 0.7, 22.0),
    ("SE Area", 40.0, 5.0, 550.0),
    ("SE Smoothness", 0.007, 0.001, 0.03),
    ("SE Compactness", 0.025, 0.002, 0.14),
    ("SE Concavity", 0.030, 0.000, 0.40),
    ("SE Concave Points", 0.012, 0.000, 0.05),
    ("SE Symmetry", 0.020, 0.008, 0.08),
    ("SE Fractal Dimension", 0.004, 0.001, 0.03),
    ("Worst Radius", 17.0, 7.0, 37.0),
    ("Worst Texture", 25.0, 12.0, 50.0),
    ("Worst Perimeter", 110.0, 50.0, 252.0),
    ("Worst Area", 900.0, 150.0, 4250.0),
    ("Worst Smoothness", 0.13, 0.07, 0.23),
    ("Worst Compactness", 0.25, 0.03, 1.10),
    ("Worst Concavity", 0.27, 0.00, 1.25),
    ("Worst Concave Points", 0.11, 0.00, 0.29),
    ("Worst Symmetry", 0.29, 0.16, 0.66),
    ("Worst Fractal Dimension", 0.08, 0.05, 0.21),
]

ALL_FIELDS = HEART_FIELDS + DIAB_FIELDS + CANCER_FIELDS   # 51 total (padded in callback)
MAX_INPUTS = max(len(HEART_FIELDS), len(DIAB_FIELDS), len(CANCER_FIELDS))  # 30 (cancer)

DS_FIELDS = {
    "Heart Disease (UCI Cleveland)": HEART_FIELDS,
    "Diabetes (Pima Indians)":        DIAB_FIELDS,
    "Breast Cancer (Wisconsin)":      CANCER_FIELDS,
}


def update_inputs(dataset):
    """Return updated Gradio Number components based on selected dataset."""
    fields = DS_FIELDS[dataset]
    updates = []
    for i in range(MAX_INPUTS):
        if i < len(fields):
            label, default, mn, mx = fields[i]
            updates.append(gr.Number(value=default, label=label, visible=True,
                                     minimum=mn, maximum=mx))
        else:
            updates.append(gr.Number(visible=False))
    return updates


with gr.Blocks(
    title="MedAI — Disease Prediction | CodeAlpha Task 4",
    theme=gr.themes.Base(
        primary_hue="cyan", secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Exo 2"), "sans-serif"]
    ),
    css="""
        body { background: #0d1520; }
        .gradio-container { background: #0d1520 !important; }
        h1, h2, h3 { color: #00d4ff !important; letter-spacing: 2px; }
        .label-wrap span { color: #8fa3bc !important; font-size: 12px; }
        footer { display: none !important; }
    """
) as demo:

    gr.Markdown("""
    # 🩺 MedAI — Disease Prediction System
    ### CodeAlpha Machine Learning Internship — Task 4
    **Algorithms:** Logistic Regression · SVM · Random Forest · XGBoost  
    **Datasets:** Heart Disease (UCI) · Diabetes (Pima) · Breast Cancer (Wisconsin)  
    *Adjust patient parameters and click **Run Prediction** to see all four models in action.*
    """)

    with gr.Row():
        dataset_dd = gr.Dropdown(
            choices=list(DATASETS.keys()),
            value="Heart Disease (UCI Cleveland)",
            label="Select Disease Dataset",
            scale=2
        )

    # Dynamic input sliders — up to 30 (Breast Cancer has most features)
    input_components = []
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Patient Parameters")
            for i in range(MAX_INPUTS):
                if i < len(HEART_FIELDS):          # default: heart disease
                    label, default, mn, mx = HEART_FIELDS[i]
                    comp = gr.Number(value=default, label=label, visible=True,
                                     minimum=mn, maximum=mx)
                else:
                    comp = gr.Number(visible=False)
                input_components.append(comp)

        with gr.Column():
            run_btn = gr.Button("⚡ Run Prediction", variant="primary", size="lg")

            pred_table = gr.DataFrame(label="Prediction Results (All Models)",
                                      interactive=False)
            metrics_table = gr.DataFrame(label="Model Evaluation Metrics",
                                         interactive=False)

    with gr.Row():
        metrics_plot = gr.Image(label="Metric Comparison Chart", type="pil")
        feat_plot    = gr.Image(label="Feature Importance (Random Forest)", type="pil")

    # Wire up dataset selector
    dataset_dd.change(fn=update_inputs, inputs=[dataset_dd],
                      outputs=input_components)

    # Wire up prediction button
    run_btn.click(
        fn=predict,
        inputs=[dataset_dd] + input_components,
        outputs=[pred_table, metrics_plot, feat_plot, metrics_table]
    )

    gr.Markdown("""
    ---
    > ⚠ **Disclaimer:** This tool is for educational purposes only.
    > Predictions are not a substitute for professional medical diagnosis.
    > Always consult a qualified healthcare provider.
    """)

if __name__ == "__main__":
    demo.launch(share=True)
