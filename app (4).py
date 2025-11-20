import os
import streamlit as st
import pandas as pd
import numpy as np

# ------------------------------
# FORCE-INSTALL packages FIRST
# ------------------------------
st.write("Installing required libraries... please wait a moment.")

os.system("pip install matplotlib seaborn fairlearn scikit-learn pandas numpy > /dev/null 2>&1")

# ------------------------------
# IMPORT PACKAGES *AFTER* INSTALL
# ------------------------------
import importlib

plt = importlib.import_module("matplotlib.pyplot")
sns = importlib.import_module("seaborn")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.linear_model import LogisticRegression

# ------------------------------
# CONFIG
# ------------------------------
TARGET = "heart_disease"
SENSITIVE = "sex"

st.set_page_config(layout="wide")
st.title("Fairness Analysis and Mitigation App")

# ------------------------------
# FILE UPLOAD
# ------------------------------
st.header("1. Upload a Medical Dataset")
uploaded_file = st.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("Dataset Loaded Successfully")
        st.dataframe(df.head())

        df_original = df.copy()

        # ------------------------------
        # PREPROCESSING
        # ------------------------------
        st.header("2. Preprocessing")
        le = LabelEncoder()

        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = le.fit_transform(df[col])

        if TARGET not in df.columns:
            st.error(f"Target column '{TARGET}' not found.")
            st.stop()

        if SENSITIVE not in df.columns:
            st.error(f"Sensitive column '{SENSITIVE}' not found.")
            st.stop()

        X = df.drop(columns=[TARGET])
        y = df[TARGET]
        A = df[SENSITIVE]

        X_train, X_test, y_train, y_test, A_train, A_test =
        train_test_split(X, y, A, test_size=0.2, random_state=42)

        # ------------------------------
        # BASELINE MODEL
        # ------------------------------
        st.header("3. Baseline Logistic Regression")

        model = LogisticRegression(max_iter=2000, solver="liblinear")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        st.write(f"Accuracy: {acc:.4f}")

        mf = MetricFrame(
            metrics={"Accuracy": accuracy_score,
                     "TPR": true_positive_rate,
                     "FPR": false_positive_rate,
                     "Selection Rate": selection_rate},
            y_true=y_test,
            y_pred=preds,
            sensitive_features=A_test
        )

        st.subheader("Fairness Before Mitigation")
        st.dataframe(mf.by_group)

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        mf.by_group.plot(kind="bar", ax=ax1)
        st.pyplot(fig1)

        # ------------------------------
        # MITIGATION
        # ------------------------------
        st.header("4. Fairness Mitigation")

        mitigator = ExponentiatedGradient(
            LogisticRegression(max_iter=2000, solver="liblinear"),
            DemographicParity()
        )

        mitigator.fit(X_train, y_train, sensitive_features=A_train)
        preds_fair = mitigator.predict(X_test)

        mf_fair = MetricFrame(
            metrics={"Accuracy": accuracy_score,
                     "TPR": true_positive_rate,
                     "FPR": false_positive_rate,
                     "Selection Rate": selection_rate},

               
