from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "customer_cluster_classifier.joblib"
DATA_PATH = APP_DIR / "CC GENERAL.csv"
PROFILE_PATH = APP_DIR / "customer_cluster_profiles.csv"

st.set_page_config(page_title="Customer Segmentation", page_icon="👥", layout="wide")
st.title("Customer Segmentation")
st.markdown(
    "Explore customer behavior and assign each customer to a segment using the saved model. "
    "This project performs customer segmentation; it does not detect fraud."
)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH.name}. Place it beside app.py.")
    st.stop()

with st.sidebar:
    st.header("Data")
    uploaded_file = st.file_uploader("Upload a customer CSV", type=["csv"])
    use_sample = st.checkbox(
        "Use the project dataset when no file is uploaded",
        value=DATA_PATH.exists() and uploaded_file is None,
        disabled=uploaded_file is not None or not DATA_PATH.exists(),
    )

try:
    if uploaded_file is not None:
        customer_data = load_csv(uploaded_file)
        source_name = uploaded_file.name
    elif use_sample and DATA_PATH.exists():
        customer_data = load_csv(DATA_PATH)
        source_name = DATA_PATH.name
    else:
        st.info("Upload a CSV file, or select the project dataset in the sidebar.")
        st.stop()
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

model = load_model(MODEL_PATH)
preprocessor = model.named_steps["preprocessor"]
feature_columns = list(preprocessor.feature_names_in_)
missing_columns = [column for column in feature_columns if column not in customer_data.columns]
if missing_columns:
    st.error("The CSV is missing model features: " + ", ".join(missing_columns))
    st.stop()

st.subheader("Dataset overview")
col1, col2, col3 = st.columns(3)
col1.metric("Customers", f"{len(customer_data):,}")
col2.metric("Columns", f"{customer_data.shape[1]:,}")
col3.metric("Source", source_name)
st.dataframe(customer_data.head(10), use_container_width=True)

st.subheader("Customer segments")
if st.button("Assign customer segments", type="primary"):
    with st.spinner("Assigning segments..."):
        features = customer_data[feature_columns].copy()
        predictions = model.predict(features)
        results = customer_data.copy()
        results["Cluster"] = predictions
        st.session_state["segmentation_results"] = results

results = st.session_state.get("segmentation_results")
if results is not None:
    counts = results["Cluster"].value_counts().sort_index()
    metric_columns = st.columns(len(counts))
    for column, (cluster, count) in zip(metric_columns, counts.items()):
        column.metric(f"Segment {cluster}", f"{count:,} customers")

    st.bar_chart(counts.rename("Customers"))
    st.dataframe(results.head(20), use_container_width=True)
    st.download_button(
        "Download customer segments as CSV",
        data=results.to_csv(index=False).encode("utf-8-sig"),
        file_name="customer_segments.csv",
        mime="text/csv",
    )

    if PROFILE_PATH.exists():
        st.subheader("Training-data segment profiles")
        st.caption("Average feature values for each segment in the training dataset.")
        st.dataframe(load_csv(PROFILE_PATH), use_container_width=True)
