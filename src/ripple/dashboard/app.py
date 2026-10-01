"""Streamlit dashboard: live signals, stress test (Module B) and index weights (Module A)."""

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Ripple", layout="wide")
st.title("Ripple")
st.caption("Real-time NLP risk signals → portfolio stress testing")

try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    signals = requests.get(f"{API_URL}/signals", params={"limit": 100}, timeout=5).json()
except requests.RequestException as e:
    st.error(f"API not reachable at {API_URL}: {e}")
    st.stop()

st.sidebar.write("API", health)

if not signals:
    st.info("No signals yet. Start the ingestor and engine to see live data.")
else:
    df = pd.DataFrame(signals)
    st.dataframe(df[["published_at", "entity", "sentiment_score", "event_type", "impact_score", "headline"]])
