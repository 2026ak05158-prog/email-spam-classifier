import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve
)

# Page configuration
st.set_page_config(
    page_title="Email Spam Classifier | ML Assignment 2",
    page_icon="📧",
    layout="wide"
)

# Application Header
st.title("📧 Email Spam Classification & Model Evaluation")
st.caption("M.Tech (AIML) - Machine Learning Assignment 2 | BITS Pilani WILP")

# Brief description
st.markdown("""
This interactive web application evaluates five distinct classification models on the **UCI Spambase Dataset** (57 continuous features).
The application loads pre-trained models and computes evaluation metrics on uploaded or bundled test data without retraining.
""")

# Cache model loading to optimize performance
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "model")
    
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    models = {
        "Logistic Regression": {
            "model": joblib.load(os.path.join(model_dir, "logistic_regression.joblib")),
            "scaled": True,
            "desc": "Linear classification model using L2 regularization."
        },
        "Decision Tree Classifier": {
            "model": joblib.load(os.path.join(model_dir, "decision_tree.joblib")),
            "scaled": False,
            "desc": "Non-linear tree-based classifier with max_depth=6."
        },
        "K-Nearest Neighbor (KNN)": {
            "model": joblib.load(os.path.join(model_dir, "knn.joblib")),
            "scaled": True,
            "desc": "Instance-based non-parametric classifier with k=5."
        },
        "Naive Bayes Classifier": {
            "model": joblib.load(os.path.join(model_dir, "naive_bayes.joblib")),
            "scaled": False,
            "desc": "Probabilistic Gaussian Naive Bayes classifier."
        },
        "Support Vector Machine (SVM)": {
            "model": joblib.load(os.path.join(model_dir, "svm.joblib")),
            "scaled": True,
            "desc": "Kernel SVM with RBF kernel and probability calibration."
        }
    }
    return scaler, models

scaler, models = load_artifacts()

# Sidebar for controls
st.sidebar.header("⚙️ Configuration & Controls")

# 1. Dataset Selection / Upload
st.sidebar.subheader("1. Test Dataset")
data_source = st.sidebar.radio(
    "Choose data source:",
    ["Use Bundled Test Data (test_data.csv)", "Upload Custom CSV"]
)

test_df = None
if data_source == "Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload test CSV (must include features + target 'class'):", type=["csv"])
    if uploaded_file is not None:
        try:
            test_df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Uploaded successfully! ({len(test_df)} rows)")
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")
    else:
        st.sidebar.info("Awaiting file upload. Using bundled test_data.csv in the meantime.")
        if os.path.exists("test_data.csv"):
            test_df = pd.read_csv("test_data.csv")
else:
    if os.path.exists("test_data.csv"):
        test_df = pd.read_csv("test_data.csv")
    else:
        st.sidebar.error("test_data.csv not found in repository root.")

# Download button for sample test data
if os.path.exists("test_data.csv"):
    with open("test_data.csv", "rb") as f:
        st.sidebar.download_button(
            label="📥 Download Sample test_data.csv",
            data=f,
            file_name="test_data.csv",
            mime="text/csv"
        )

# 2. Model Selection Dropdown
st.sidebar.subheader("2. Model Selection")
model_options = list(models.keys()) + ["Compare All 5 Models"]
selected_option = st.sidebar.selectbox("Select ML Model:", model_options)

# Validate dataset
if test_df is None:
    st.warning("Please provide a valid test dataset to evaluate models.")
    st.stop()

# Check for target column
target_col = None
for candidate in ["class", "target", "label", "Class", "Target"]:
    if candidate in test_df.columns:
        target_col = candidate
        break

if target_col is None:
    st.error("Error: Target column ('class') not found in uploaded CSV. Please ensure the CSV contains features and the ground-truth label.")
    st.stop()

X_eval = test_df.drop(columns=[target_col])
y_eval = test_df[target_col].astype(int)

# Normalize column names in uploaded data
rename_map = {';': 'semicolon', '(': 'round_bracket', '[': 'square_bracket', '!': 'exclamation', '$': 'dollar', '#': 'hash'}
norm_cols = []
for c in X_eval.columns:
    col = str(c)
    for k, v in rename_map.items():
        col = col.replace(k, v)
    norm_cols.append(col)
X_eval.columns = norm_cols

# Check feature count
if X_eval.shape[1] < 12:
    st.error(f"The dataset must contain at least 12 features. Detected {X_eval.shape[1]} features.")
    st.stop()

# Scale features
try:
    X_eval_scaled = scaler.transform(X_eval)
except Exception as e:
    # If column alignment differs, warn user
    st.error(f"Feature alignment error with trained scaler: {e}")
    st.stop()

# Helper function to compute metrics
def compute_metrics(model_dict, X_raw, X_sc, y_true):
    clf = model_dict["model"]
    use_sc = model_dict["scaled"]
    features = X_sc if use_sc else X_raw
    
    y_pred = clf.predict(features)
    
    # Probability for AUC
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(features)[:, 1]
    elif hasattr(clf, "decision_function"):
        y_prob = clf.decision_function(features)
    else:
        y_prob = y_pred

    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = np.nan
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    return {
        "acc": acc,
        "auc": auc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "mcc": mcc,
        "y_pred": y_pred,
        "y_prob": y_prob
    }

# Main display logic
if selected_option != "Compare All 5 Models":
    model_info = models[selected_option]
    m = compute_metrics(model_info, X_eval, X_eval_scaled, y_eval)
    
    st.subheader(f"Model: {selected_option}")
    st.write(f"*{model_info['desc']}*")
    
    # Metric cards display
    st.markdown("### 📊 Evaluation Metrics")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Accuracy", f"{m['acc']:.4f}")
    col2.metric("AUC Score", f"{m['auc']:.4f}" if not np.isnan(m['auc']) else "N/A")
    col3.metric("Precision", f"{m['prec']:.4f}")
    col4.metric("Recall", f"{m['rec']:.4f}")
    col5.metric("F1 Score", f"{m['f1']:.4f}")
    col6.metric("MCC Score", f"{m['mcc']:.4f}")
    
    st.divider()
    
    # Visualizations: Confusion Matrix & Classification Report
    viz_col1, viz_col2 = st.columns([1, 1])
    
    with viz_col1:
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_eval, m["y_pred"])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Non-Spam (0)", "Spam (1)"],
            yticklabels=["Non-Spam (0)", "Spam (1)"],
            ax=ax
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix: {selected_option}")
        st.pyplot(fig)
        plt.close(fig)
        
    with viz_col2:
        st.markdown("#### Classification Report")
        cr_dict = classification_report(y_eval, m["y_pred"], target_names=["Non-Spam (0)", "Spam (1)"], output_dict=True)
        cr_df = pd.DataFrame(cr_dict).transpose().round(4)
        st.dataframe(cr_df, use_container_width=True)
        
        # ROC Curve if AUC is valid
        if not np.isnan(m["auc"]):
            st.markdown("#### ROC Curve")
            fpr, tpr, _ = roc_curve(y_eval, m["y_prob"])
            fig_roc, ax_roc = plt.subplots(figsize=(5, 3))
            ax_roc.plot(fpr, tpr, label=f"AUC = {m['auc']:.4f}", color="darkblue", lw=2)
            ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("Receiver Operating Characteristic (ROC)")
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)
            plt.close(fig_roc)

else:
    # Compare all 5 models view
    st.subheader("📊 Comparison of All 5 Machine Learning Models")
    
    comparison_data = []
    for name, m_info in models.items():
        res = compute_metrics(m_info, X_eval, X_eval_scaled, y_eval)
        comparison_data.append({
            "ML Model Name": name,
            "Accuracy": round(res["acc"], 4),
            "AUC": round(res["auc"], 4),
            "Precision": round(res["prec"], 4),
            "Recall": round(res["rec"], 4),
            "F1": round(res["f1"], 4),
            "MCC": round(res["mcc"], 4)
        })
    
    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df, use_container_width=True)
    
    # Bar Chart for metrics comparison
    st.markdown("#### Graphical Metric Comparison")
    melted = comp_df.melt(id_vars=["ML Model Name"], var_name="Metric", value_name="Score")
    
    fig_comp, ax_comp = plt.subplots(figsize=(10, 5))
    sns.barplot(data=melted, x="Metric", y="Score", hue="ML Model Name", palette="Set2", ax=ax_comp)
    ax_comp.set_ylim(0.65, 1.0)
    ax_comp.set_ylabel("Score")
    ax_comp.set_title("Cross-Model Performance Across All 6 Evaluation Metrics")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    st.pyplot(fig_comp)
    plt.close(fig_comp)

# Dataset Summary Section at bottom
with st.expander("ℹ️ Inspect Dataset Details"):
    st.write(f"**Total Samples Evaluated:** {len(test_df)}")
    st.write(f"**Number of Input Features:** {X_eval.shape[1]}")
    st.write(f"**Ground-Truth Class Breakdown:**")
    st.write(y_eval.value_counts().rename({0: "Non-Spam (0)", 1: "Spam (1)"}))
    st.dataframe(test_df.head(10), use_container_width=True)
