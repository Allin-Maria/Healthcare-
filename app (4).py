import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

from sklearn.linear_model import LogisticRegression

# Hardcoded TARGET and SENSITIVE variables based on the notebook context
TARGET = 'heart_disease'
SENSITIVE = 'sex'

st.set_page_config(layout="wide")

st.title("Fairness Analysis and Mitigation App")

st.markdown("""
This application allows you to upload a dataset, train a baseline logistic regression model,
analyze its fairness with respect to a sensitive attribute, and apply fairness mitigation
techniques using Fairlearn's ExponentiatedGradient algorithm.

**Hardcoded Target Variable:** `{}`
**Hardcoded Sensitive Attribute:** `{}`
""".format(TARGET, SENSITIVE))

st.header("1. Upload your Dataset")
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        
        st.success(f"Successfully loaded: {uploaded_file.name}")
        st.write("Dataset Preview:")
        st.dataframe(df.head())

        # Copy before encoding
        df_original = df.copy()

        st.header("2. Data Preprocessing")
        st.write(f"Automatically encoding categorical variables (if any) and preparing data for modeling.")

        # Automatically encode categorical variables
        le = LabelEncoder()
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = le.fit_transform(df[col])
        
        st.write("Columns in dataset:")
        st.write(df.columns.tolist())

        if TARGET not in df.columns:
            st.error(f"Error: Target column '{TARGET}' not found in the dataset. Please ensure your dataset has a column named '{TARGET}'.")
        elif SENSITIVE not in df.columns:
            st.error(f"Error: Sensitive attribute column '{SENSITIVE}' not found in the dataset. Please ensure your dataset has a column named '{SENSITIVE}'.")
        else:
            X = df.drop(columns=[TARGET])
            y = df[TARGET]
            A = df[SENSITIVE]  # sensitive attribute

            X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
                X, y, A, test_size=0.2, random_state=42
            )
            st.success("Data split into training and testing sets.")

            st.header("3. Baseline Model Training and Evaluation")
            st.write("Training a Logistic Regression model as a baseline.")

            model = LogisticRegression(max_iter=2000, solver='liblinear')
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            st.write(f"**Baseline Model Accuracy:** {acc:.4f}")

            st.subheader("Fairness Metrics (Baseline Model - Before Mitigation)")
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

            st.dataframe(mf.by_group)

            st.subheader("Fairness Metrics Visualization (Baseline)")
            fig_baseline, ax_baseline = plt.subplots(figsize=(10, 6))
            mf.by_group.plot(kind="bar", ax=ax_baseline)
            ax_baseline.set_title("Fairness Metrics by Sensitive Attribute (Baseline)")
            ax_baseline.set_ylabel("Metric Value")
            ax_baseline.set_xlabel(SENSITIVE)
            st.pyplot(fig_baseline)

            st.header("4. Fairness Mitigation with ExponentiatedGradient")
            st.write(f"Applying Demographic Parity constraint to mitigate fairness issues concerning '{SENSITIVE}'.")

            constraint = DemographicParity()
            mitigator = ExponentiatedGradient(
                LogisticRegression(max_iter=2000, solver='liblinear'),
                constraint
            )
            
            mitigator.fit(X_train, y_train, sensitive_features=A_train)
            preds_fair = mitigator.predict(X_test)

            st.subheader("Fairness Metrics (Mitigated Model - After Mitigation)")
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

            st.dataframe(mf_fair.by_group)

            st.subheader("Comparison: Before vs. After Mitigation")
            combined_metrics = pd.concat(
                [mf.by_group.add_suffix(" (Before)"),
                 mf_fair.by_group.add_suffix(" (After)")], axis=1
            )
            st.dataframe(combined_metrics)

            st.subheader("Fairness Metrics Visualization (After Mitigation)")
            fig_fair, ax_fair = plt.subplots(figsize=(10, 6))
            mf_fair.by_group.plot(kind="bar", ax=ax_fair)
            ax_fair.set_title("Fairness Metrics by Sensitive Attribute (After Mitigation)")
            ax_fair.set_ylabel("Metric Value")
            ax_fair.set_xlabel(SENSITIVE)
            st.pyplot(fig_fair)
            
            st.header("5. Project Summary")
            st.write(f"**Dataset shape:** {df_original.shape}")
            st.write(f"**Target variable:** '{TARGET}'")
            st.write(f"**Sensitive attribute:** '{SENSITIVE}'")
            st.write(f"**Baseline Model Accuracy:** {acc:.4f}")

            st.subheader("Detailed Fairness Comparison")
            st.markdown("Metrics for baseline model:")
            st.dataframe(mf.by_group)
            st.markdown("Metrics for fairness-mitigated model:")
            st.dataframe(mf_fair.by_group)


    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.write("Please ensure your file is correctly formatted and contains the specified 'TARGET' and 'SENSITIVE' columns.")
else:
    st.info("Please upload a CSV or Excel file to get started.")
