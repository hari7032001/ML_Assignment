import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNBx
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from model import preprocess_data, train_model, evaluate_model


# ----------------------------------------------------
# Page Config
# ----------------------------------------------------
st.set_page_config(
    page_title="Bank Marketing ML Dashboard",
    layout="wide"
)

st.title("📊 Bank Marketing Classification Dashboard")
st.markdown("Train and evaluate ML models for term deposit prediction.")

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------
st.sidebar.header("⚙️ Model Selection")

uploaded_file = st.sidebar.file_uploader(
    "Upload Bank Marketing CSV",
    type=["csv"]
)

model_option = st.sidebar.selectbox(
    "Choose Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

# ----------------------------------------------------
# Main App Logic
# ----------------------------------------------------
if uploaded_file is not None:

    data = pd.read_csv(uploaded_file, sep=";")

    st.subheader("📂 Dataset Preview")
    st.dataframe(data.head())

    # ---------------- Preprocessing ----------------
    X, y = preprocess_data(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ---------------- Model Selection ----------------
    if model_option == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)

    elif model_option == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)

    elif model_option == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)

    elif model_option == "Naive Bayes":
        model = GaussianNB()

    elif model_option == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    elif model_option == "XGBoost":
        model = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        )

    # ---------------- Train ----------------
    model = train_model(model, X_train, y_train)

    # ---------------- Evaluate ----------------
    metrics = evaluate_model(model, X_test, y_test)

    # ---------------- Display Metrics ----------------
    st.subheader("📈 Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    col1.metric("Precision", f"{metrics['Precision']:.4f}")

    col2.metric("Recall", f"{metrics['Recall']:.4f}")
    col2.metric("F1 Score", f"{metrics['F1']:.4f}")

    col3.metric("AUC", f"{metrics['AUC']:.4f}" if metrics['AUC'] else "N/A")
    col3.metric("MCC", f"{metrics['MCC']:.4f}")

    # ---------------- Confusion Matrix ----------------
    st.subheader("🧩 Confusion Matrix")

    fig, ax = plt.subplots()
    sns.heatmap(
        metrics["Confusion Matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"]
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

else:
    st.info("👈 Upload dataset from sidebar to begin.")
