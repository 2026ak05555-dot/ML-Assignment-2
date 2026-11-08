
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="ML Classification Model Comparison",
    page_icon="",
    layout="wide"
)

st.title("Machine Learning Classification Dashboard")
st.write("Comparison of five classification algorithms using the Dry Bean dataset.")

MODEL_FILES = {
    "Logistic Regression": "models/logistic_regression.joblib",
    "Decision Tree": "models/decision_tree.joblib",
    "K-Nearest Neighbors": "models/knn.joblib",
    "Naive Bayes": "models/naive_bayes.joblib",
    "Support Vector Machine": "models/support_vector_machine.joblib"
}

st.sidebar.header("Model Configuration")

selected_model = st.sidebar.selectbox(
    "Select Classification Model",
    list(MODEL_FILES.keys())
)

st.header("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload test CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.write("Dataset shape:", df.shape)
    st.dataframe(df.head())

    if "Class" not in df.columns:
        st.error("Target column 'Class' is missing from the uploaded CSV.")

    else:

        model_path = MODEL_FILES[selected_model]

        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")

        else:

            model = joblib.load(model_path)

            X_test = df.drop(columns=["Class"])
            y_test = df["Class"]

            expected_features = list(model.feature_names_in_)

            missing_features = set(expected_features) - set(X_test.columns)
            extra_features = set(X_test.columns) - set(expected_features)

            if missing_features:
                st.error(f"Missing input features: {sorted(missing_features)}")

            elif extra_features:
                st.error(f"Unexpected input features: {sorted(extra_features)}")

            elif len(df) == 0:
                st.error("Uploaded CSV contains no records.")

            else:
                X_test = X_test[expected_features]

                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)

                classes = model.classes_

                accuracy = accuracy_score(y_test, y_pred)

                precision = precision_score(
                    y_test, y_pred,
                    average="weighted",
                    zero_division=0
                )

                recall = recall_score(
                    y_test, y_pred,
                    average="weighted",
                    zero_division=0
                )

                f1 = f1_score(
                    y_test, y_pred,
                    average="weighted",
                    zero_division=0
                )

                mcc = matthews_corrcoef(y_test, y_pred)

                try:
                    auc = roc_auc_score(
                        y_test,
                        y_prob,
                        multi_class="ovr",
                        average="weighted",
                        labels=classes
                    )
                except ValueError:
                    auc = None

                st.success(f"Evaluation completed: {selected_model}")

                st.header("Model Evaluation Metrics")

                col1, col2, col3 = st.columns(3)

                col1.metric("Accuracy", f"{accuracy:.4f}")
                col2.metric("AUC Score", f"{auc:.4f}" if auc is not None else "N/A")
                col3.metric("Precision", f"{precision:.4f}")

                col4, col5, col6 = st.columns(3)

                col4.metric("Recall", f"{recall:.4f}")
                col5.metric("F1 Score", f"{f1:.4f}")
                col6.metric("MCC Score", f"{mcc:.4f}")

                st.header("Confusion Matrix")

                cm = confusion_matrix(
                    y_test,
                    y_pred,
                    labels=classes
                )

                cm_df = pd.DataFrame(
                    cm,
                    index=classes,
                    columns=classes
                )

                st.dataframe(cm_df)

                st.header("Classification Report")

                report = classification_report(
                    y_test,
                    y_pred,
                    zero_division=0,
                    output_dict=True
                )

                st.dataframe(pd.DataFrame(report).transpose())

                st.header("Prediction Results")

                predictions = X_test.copy()
                predictions["Actual Class"] = y_test.values
                predictions["Predicted Class"] = y_pred

                st.dataframe(predictions)

                st.download_button(
                    "Download Prediction Results",
                    predictions.to_csv(index=False),
                    "prediction_results.csv",
                    "text/csv"
                )

else:
    st.info("Please upload test_data.csv to evaluate the selected model.")

    st.write("Available classification models:")

    for model_name in MODEL_FILES:
        st.write("- " + model_name)
