import json
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st
import requests
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Exoplanet Detection System", layout="wide")

API_URL = "http://localhost:8000"


def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_model_metrics():
    try:
        response = requests.get(f"{API_URL}/model/metrics")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def predict_single(features):
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": features}
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Prediction failed: {e}")
    return None


st.title("🌍 Exoplanet Detection System")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Overview", "Single Prediction", "Batch Prediction", "Retrain Model"])

    st.divider()

    api_healthy = check_api_health()
    if api_healthy:
        st.success("✓ API Connected")
    else:
        st.error("✗ API Offline")
        st.info("Start API: `uvicorn src.api:app`")

if page == "Overview":
    st.header("System Overview")

    st.markdown("""
    Automated exoplanet identification system trained on NASA's Kepler, K2, and TESS datasets.
    Uses LightGBM classifier with SHAP explainability.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Features")
        st.markdown("""
        - Multi-mission dataset integration
        - Advanced feature engineering
        - SMOTE class balancing
        - Cross-validated training
        - SHAP-based explanations
        """)

    with col2:
        st.subheader("Model Classes")
        st.markdown("""
        - **Confirmed**: Validated exoplanets
        - **Candidate**: Potential exoplanets
        - **False Positive**: Non-planetary signals
        """)

    st.divider()

    metrics = get_model_metrics()
    if metrics:
        st.subheader("Current Model Performance")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Precision", f"{metrics['precision']:.3f}")
        col2.metric("Recall", f"{metrics['recall']:.3f}")
        col3.metric("F1 Score", f"{metrics['f1']:.3f}")
        col4.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

        with st.expander("Detailed Classification Report"):
            st.json(metrics["classification_report"])

        with st.expander("Confusion Matrix"):
            cm = np.array(metrics["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)
    else:
        st.warning("No model metrics available. Train a model first.")


elif page == "Single Prediction":
    st.header("Single Sample Prediction")

    st.info("Enter exoplanet candidate features for classification")

    col1, col2 = st.columns(2)

    with col1:
        period = st.number_input("Orbital Period (days)", value=10.0, min_value=0.0)
        duration = st.number_input("Transit Duration (hours)", value=3.0, min_value=0.0)
        depth = st.number_input("Transit Depth (ppm)", value=500.0, min_value=0.0)

    with col2:
        radius = st.number_input("Planet Radius (Earth radii)", value=2.0, min_value=0.0)
        impact = st.number_input("Impact Parameter", value=0.5, min_value=0.0, max_value=1.0)
        snr = st.number_input("Signal-to-Noise Ratio", value=15.0, min_value=0.0)

    if st.button("Predict", type="primary"):
        features = {
            "koi_period": period,
            "koi_duration": duration,
            "koi_depth": depth,
            "koi_prad": radius,
            "koi_impact": impact,
            "koi_model_snr": snr,
        }

        result = predict_single(features)

        if result:
            st.success(f"Predicted Class: **{result['predicted_class']}**")

            st.subheader("Class Probabilities")
            probs_df = pd.DataFrame(result["probabilities"].items(), columns=["Class", "Probability"])
            st.bar_chart(probs_df.set_index("Class"))

            st.subheader("Top Contributing Features (SHAP)")
            shap_df = pd.DataFrame(result["top_features"].items(), columns=["Feature", "SHAP Value"])
            st.dataframe(shap_df, use_container_width=True)


elif page == "Batch Prediction":
    st.header("Batch CSV Prediction")

    st.markdown("Upload a CSV file with exoplanet candidate features for batch classification.")

    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:", df.head())

        if st.button("Run Batch Prediction", type="primary"):
            with st.spinner("Processing..."):
                try:
                    files = {"file": ("input.csv", uploaded_file.getvalue(), "text/csv")}
                    response = requests.post(f"{API_URL}/predict_batch", files=files)

                    if response.status_code == 200:
                        result_df = pd.read_csv(pd.io.common.BytesIO(response.content))
                        st.success(f"Processed {len(result_df)} records")
                        st.dataframe(result_df.head(20))

                        csv = result_df.to_csv(index=False).encode()
                        st.download_button(
                            "Download Results",
                            csv,
                            "predictions.csv",
                            "text/csv"
                        )
                    else:
                        st.error(f"Prediction failed: {response.text}")
                except Exception as e:
                    st.error(f"Error: {e}")


elif page == "Retrain Model":
    st.header("Model Retraining")

    st.warning("⚠️ Retraining will replace the current model. Maximum 50,000 rows.")

    retrain_file = st.file_uploader("Upload labeled training CSV", type=["csv"])

    if retrain_file:
        df = pd.read_csv(retrain_file)
        st.write(f"Uploaded: {len(df)} rows")
        st.write("Preview:", df.head())

        if len(df) > 50000:
            st.error("Dataset exceeds 50,000 row limit")
        elif "target" not in df.columns:
            st.error("CSV must contain 'target' column with labels")
        else:
            st.info("Ready to retrain. This may take several minutes.")

            if st.button("🚀 Confirm and Retrain", type="primary"):
                output_path = f"data/processed/retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                Path("data/processed").mkdir(parents=True, exist_ok=True)
                df.to_csv(output_path, index=False)

                with st.spinner("Training in progress..."):
                    try:
                        result = subprocess.run(
                            ["python", "src/train.py"],
                            capture_output=True,
                            text=True,
                            timeout=600
                        )

                        if result.returncode == 0:
                            st.success("✓ Retraining complete!")
                            st.code(result.stdout)

                            st.info("Restart API to load new model")
                        else:
                            st.error("Training failed")
                            st.code(result.stderr)
                    except subprocess.TimeoutExpired:
                        st.error("Training timeout (10 min limit)")
                    except Exception as e:
                        st.error(f"Error: {e}")
