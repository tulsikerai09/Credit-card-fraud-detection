
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

FEATURE_COLUMNS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    with open(
        MODEL_DIR / "best_fraud_detection_model.pkl", "rb"
    ) as file:
        model = pickle.load(file)

    with open(
        MODEL_DIR / "fraud_threshold.pkl", "rb"
    ) as file:
        threshold = pickle.load(file)

    return model, threshold


st.title("💳 Credit Card Fraud Detection")
# st.write(
#     "An educational machine learning dashboard that "
#     "predicts potential fraudulent transactions."
# )

try:
    model, threshold = load_model()
except FileNotFoundError:
    st.error(
        "Model files not found. Check that "
        "best_fraud_detection_model.pkl and "
        "fraud_threshold.pkl are inside the models folder."
    )
    st.stop()

st.caption(f"Fraud classification threshold: {threshold:.2f}")

# -----------------------------
# Upload transaction data
# -----------------------------
st.subheader("Upload transaction data")

st.write(
    "Upload a CSV containing the same feature columns used "
    "during model training: Time, V1–V28, and Amount."
)

uploaded_file = st.file_uploader(
    "Choose a transaction CSV file",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        transactions = pd.read_csv(uploaded_file)

        # Allow an optional actual-label column, but do not
        # use it as a model input.
        actual_labels = None
        if "Class" in transactions.columns:
            actual_labels = transactions["Class"]

        missing_columns = [
            col for col in FEATURE_COLUMNS
            if col not in transactions.columns
        ]

        if missing_columns:
            st.error(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )
            st.stop()

        X = transactions[FEATURE_COLUMNS].copy()

        # Ensure feature values are numeric.
        X = X.apply(pd.to_numeric, errors="coerce")

        if X.isna().any().any():
            st.error(
                "The input contains missing or non-numeric "
                "feature values. Please clean the CSV first."
            )
            st.stop()

        if X.empty:
            st.warning("The uploaded CSV contains no transactions.")
            st.stop()

        # Predict fraud probabilities and labels.
        probabilities = model.predict_proba(X)[:, 1]
        predictions = (probabilities >= threshold).astype(int)

                # -----------------------------
        # Model Evaluation
        # -----------------------------
        st.subheader("Model Evaluation")

        if actual_labels is not None:
            valid_labels = (
                pd.to_numeric(actual_labels, errors="coerce")
                .isin([0, 1]).all()
            )

            if valid_labels:
                y_true = pd.to_numeric(actual_labels).astype(int)

                accuracy = accuracy_score(y_true, predictions)
                precision = precision_score(
                    y_true, predictions, zero_division=0
                )
                recall = recall_score(
                    y_true, predictions, zero_division=0
                )
                f1 = f1_score(
                    y_true, predictions, zero_division=0
                )

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{accuracy:.2%}")
                c2.metric("Precision", f"{precision:.2%}")
                c3.metric("Recall", f"{recall:.2%}")
                c4.metric("F1 Score", f"{f1:.2%}")

                cm = confusion_matrix(
                    y_true, predictions, labels=[0, 1]
                )

                fig, ax = plt.subplots(figsize=(5, 4))
                display = ConfusionMatrixDisplay(
                    confusion_matrix=cm,
                    display_labels=[
                        "Legitimate (0)", "Fraud (1)"
                    ],
                )
                display.plot(
                    ax=ax,
                    values_format="d",
                    colorbar=False,
                )
                ax.set_title("Confusion Matrix")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            else:
                st.warning(
                    "The Class column must contain only 0 and 1 "
                    "to calculate evaluation metrics."
                )
        else:
            st.info(
                "For evaluation metrics, upload a labeled CSV "
                "that includes a Class column (0 = legitimate, "
                "1 = fraud). The Class column is not used as "
                "a model input."
            )

        results = transactions.copy()
        results["Fraud Probability (%)"] = (
            probabilities * 100
        ).round(2)
        results["Prediction"] = [
            "Potential Fraud" if p == 1 else "Likely Legitimate"
            for p in predictions
        ]

        # Summary metrics.
        total = len(results)
        flagged = int(predictions.sum())
        legitimate = total - flagged

        col1, col2, col3 = st.columns(3)
        col1.metric("Transactions analyzed", f"{total:,}")
        col2.metric("Potential fraud flagged", f"{flagged:,}")
        col3.metric("Likely legitimate", f"{legitimate:,}")


        
        # -----------------------------
        # Model Comparison
        # -----------------------------
        st.subheader("Machine Learning Model Comparison")

        comparison_data = pd.DataFrame({
            "Model": ["Logistic Regression", "Random Forest"],
            "Accuracy": [0.9756, 0.9995],
            "Precision": [0.0570, 0.9250],
            "Recall": [0.8737, 0.7789],
            "F1 Score": [0.1070, 0.8457],
        })

        st.dataframe(
            comparison_data.style.format({
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1 Score": "{:.2%}",
            }),
            use_container_width=True
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        comparison_data.set_index("Model").plot(
            kind="bar",
            ax=ax
        )

        ax.set_title("Performance Comparison of ML Models")
        ax.set_ylabel("Score")
        ax.set_xlabel("Machine Learning Model")
        ax.set_ylim(0, 1)
        ax.legend(loc="lower right")
        ax.tick_params(axis="x", rotation=0)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


        

        # -----------------------------
        # Improved Visualizations
        # -----------------------------
        import matplotlib.pyplot as plt

        st.subheader("Transaction Analysis")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.write("**Prediction Distribution**")

            counts = pd.Series({
                "Likely Legitimate": legitimate,
                "Potential Fraud": flagged,
            })

            fig1, ax1 = plt.subplots(figsize=(6, 4))

            counts.plot(kind="bar", ax=ax1, rot=0)

            # Log scale makes the smaller fraud count visible.
            ax1.set_yscale("log")
            ax1.set_ylabel("Transactions (log scale)")
            ax1.set_xlabel("Predicted Category")
            ax1.set_title("Legitimate vs Potential Fraud")

            for i, value in enumerate(counts):
                ax1.annotate(
                    f"{value:,}",
                    (i, value),
                    ha="center",
                    xytext=(0, 5),
                    textcoords="offset points",
                )

            fig1.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)

        with chart_col2:
            st.write("**Fraud Probabilities of Flagged Transactions**")

            flagged_probabilities = probabilities[
                predictions == 1
            ]

            fig2, ax2 = plt.subplots(figsize=(6, 4))

            if len(flagged_probabilities) > 0:
                ax2.hist(
                    flagged_probabilities,
                    bins=20,
                )
                ax2.axvline(
                    threshold,
                    linestyle="--",
                    label=f"Threshold = {threshold:.2f}",
                )
                ax2.legend()
                ax2.set_xlabel("Predicted Fraud Probability")
                ax2.set_ylabel("Number of Transactions")
                ax2.set_title("Flagged Transactions Only")
            else:
                ax2.text(
                    0.5, 0.5,
                    "No transactions were flagged",
                    ha="center",
                    va="center",
                )
                ax2.set_axis_off()

            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)



        st.subheader("Prediction results")
        st.dataframe(results, use_container_width=True)

        # Download results.
        csv_data = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download prediction results",
            data=csv_data,
            file_name="fraud_predictions.csv",
            mime="text/csv",
        )

        if actual_labels is not None:
            st.info(
                "The uploaded CSV contains a Class column. "
                "It is displayed as part of the original data "
                "but is not used as a prediction input."
            )

    except (ValueError, pd.errors.ParserError) as error:
        st.error(f"Could not process the CSV: {error}")

else:
    st.info(
        "Upload a CSV to analyze transactions. "
        "Use the same feature format as the training dataset."
    )

st.divider()
# st.caption(
#     "Academic demonstration only. Predictions are not a "
#     "guarantee of fraud and should not be used alone to "
#     "make real financial decisions."
# )
