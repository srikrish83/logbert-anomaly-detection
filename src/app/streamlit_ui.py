import streamlit as st
from kafka import KafkaConsumer
from utils.config_loader import load_config
from src.rca.rca_pipeline import start_rca_pipeline, rca_queue


config = load_config()

st.set_page_config(page_title="LogBERT Real-Time RCA", layout="wide")
st.title("📊 LogBERT Real-Time Anomaly RCA ")

placeholder = st.empty()

start_rca_pipeline()  # Start RCA background listener

while True:
    if not rca_queue.empty():
        data = rca_queue.get()
        anomaly = data.get("anomaly", "No anomaly text")
        rca = data.get("rca", "No RCA generated")

        with placeholder.container():
            st.error("🚨 New Anomaly Detected!")
            st.code(anomaly, language="text")
            st.success("💡 Root Cause Analysis")
            st.write(rca)

            st.toast("🔔 RCA generated for anomaly!", icon="🚨")
