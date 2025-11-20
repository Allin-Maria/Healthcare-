import os
# Auto-install missing dependencies (works on Streamlit Cloud / Replit / Render)
os.system("pip install matplotlib seaborn fairlearn scikit-learn pandas numpy")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

from sklearn.linear_model import LogisticRegression

# Hardcoded TARGET and SENSITIVE variables
TARGET = 'heart_disease'
SENSITIVE = 'sex'

st.set_page_config(layout="wide")

st.title("Fairness Analysis and Mitigation App")

st.markdown(f"""
Upload a dataset, analyze bias, and apply fairness mitigation.

**Target Column:** `{TARGET}`  
**Sensitive Attribute:** `{SENSITIVE}`
""")

# ------------------------------
# 1. File Upload
# ------------------------------
st.header("1. Upload Dataset")
uploaded_file = st.file_uploader("Choose CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Read dataset
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"Loaded dataset: {uploaded_file.name}")
        st.dataframe(df.head())

        df_original = df.copy()

        # ------------------------------
        # 2. Preprocessing
        # ------------------------------
        st.header("2. Preprocessing")
        st.write("Encoding categorical columns automatically...")

        le = LabelEncoder()
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = le.fit_transform(df[col])

        st.write("Columns detected:")
        st.write(df.columns.tolist())

        # Check mandatory columns
        if TARGET not in df.columns:
            st.error(f"ERROR: The dataset does not contain the target column `{TARGET}`.")
            st.stop()

        if SENSITIVE not in df.columns:
            st.error(f"ERROR: The dataset does not contain the sensitive attribute `{SENSITIVE}`.")
            st.stop()

        X = df.drop(columns=[TARGET])
        y = df[TARGET]
        A = df[SENSITIVE]

        X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
            X, y, A, test_size=0.2, random_state=42
        )

        # ------------------------------
        # 3. Baseline Model
        # ------------------------------
        st.header("3. Baseline Logistic Regression Model")

        model = LogisticRegression(max_iter=2000, solver="liblinear")
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        st.write(f"**Baseline Accuracy:** {acc:.4f}")

        # Fairness metrics
        mf = MetricFrame(
            metrics={
                "Accuracy": accuracy_score,
                "TPR": true_positive_rate,
                "FPR": false_positive_rate,
                "Selection Rate": selection_rate
            },
            y_true=y_test,
            y_pred=preds,
            sensitive_features=A_test
        )

        st.subheader("Fairness Metrics (Before Mitigation)")
        st.dataframe(mf.by_group)

        # Plot fairness metrics
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        mf.by_group.plot(kind="bar", ax=ax1)
        ax1.set_title("Fairness Metrics - Baseline")
        st.pyplot(fig1)

        # ------------------------------
        # 4. Fairness Mitigation
        # ------------------------------
        st.header("4. Fairness Mitigation (Demographic Parity)")

        mitigator = ExponentiatedGradient(
            LogisticRegression(max_iter=2000, solver="liblinear"),
            DemographicParity()
        )

        mitigator.fit(X_train, y_train, sensitive_features=A_train)
        preds_fair = mitigator.predict(X_test)

        mf_fair = MetricFrame(
            metrics={
                "Accuracy": accuracy_score,
                "TPR": true_positive_rate,
                "FPR": false_positive_rate,
                "Selection Rate": selection_rate
            },
            y_true=y_test,
            y_pred=preds_fair,
            sensitive_features=A_test
        )

        st.subheader("Fairness Metrics (After Mitigation)")
        st.dataframe(mf_fair.by_group)

        # Plot fairness after mitigation
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        mf_fair.by_group.plot(kind="bar", ax=ax2)
        ax2.set_title("Fairness Metrics - After Mitigation")
        st.pyplot(fig2)

        # ------------------------------
        # 5. Summary
        # ------------------------------
        st.header("5. Summary")

        st.write("**Dataset shape:**", df_original.shape)
        st.write("**Baseline Accuracy:**", acc)
        st.write("**Sensitive Attribute:**", SENSITIVE)

        st.subheader("Comparison Table: Before vs After Fairness Mitigation")
        combined = pd.concat(
            [mf.by_group.add_suffix(" (Before)"),
             mf_fair.by_group.add_suffix(" (After)")],
            axis=1
        )
        st.dataframe(combined)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Upload a dataset to start.")

               
