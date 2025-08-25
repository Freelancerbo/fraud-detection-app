

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# Load model
try:
    model = joblib.load('fraud_detection_model.pkl')
except Exception as e:
    st.error("❌ Model not loaded. Make sure 'fraud_detection_model.pkl' is in the folder.")
    st.stop()

# Page config
st.set_page_config(page_title="🛡️ FraudGuard AI", page_icon="🛡️", layout="centered")

# --- Animated CSS Styling for "Built by Tafseer Haider" ---
st.markdown("""
<style>
@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0; }
}
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.1); }
    100% { transform: scale(1); }
}
.animated-text {
    font-family: 'Courier New', monospace;
    font-size: 18px;
    text-align: center;
    margin: 20px 0;
    animation: pulse 2s infinite, blink 1.5s infinite;
    color: #FF4081;
    font-weight: bold;
    text-shadow: 1px 1px 2px #00000020;
}
</style>
<div class="animated-text">
✨ Built by Tafseer Haider ✨
</div>
""", unsafe_allow_html=True)

# Title
st.title("🛡️ AI-Powered Fraud Detection System")
st.markdown("🔍 Detect fraudulent transactions in real-time with high accuracy.")

# Sidebar
st.sidebar.header("⚡ Quick Actions")
if st.sidebar.button("🎯 Load Random Fraud Values"):
    fraud_sample = [-2.3, 1.5, -1.8, 3.2, -0.5, 2000.0]
    for i in range(1, 6):
        st.session_state[f"v{i}"] = fraud_sample[i-1]
    st.session_state.amount = fraud_sample[5]

if st.sidebar.button("✅ Load Normal Transaction"):
    normal_sample = [0.0, 0.1, -0.2, 0.3, 0.0, 50.0]
    for i in range(1, 6):
        st.session_state[f"v{i}"] = normal_sample[i-1]
    st.session_state.amount = normal_sample[5]

if st.sidebar.button("🗑️ Clear All Inputs"):
    for i in range(1, 6):
        st.session_state[f"v{i}"] = 0.0
    st.session_state.amount = 0.0

# Current time
st.sidebar.markdown("---")
st.sidebar.write("🕒 Current Time:")
st.sidebar.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Input fields
st.subheader("📋 Enter Transaction Details")

cols = st.columns(3)
for i in range(1, 6):
    key = f"v{i}"
    if key not in st.session_state:
        st.session_state[key] = 0.0
    with cols[(i-1) % 3]:
        st.number_input(f"V{i}", value=st.session_state[key], format="%.6f", step=0.01, key=key)

st.subheader("💰 Transaction Amount")
if "amount" not in st.session_state:
    st.session_state.amount = 0.0
st.number_input("", value=st.session_state.amount, format="%.2f", step=1.0, key="amount")

# Predict
st.markdown("---")
if st.button("🔮 Predict Fraud"):
    input_data = [st.session_state[f"v{i}"] for i in range(1, 6)]
    input_data.append(st.session_state.amount)
    input_array = np.array([input_data])

    pred = model.predict(input_array)[0]
    prob = model.predict_proba(input_array)[0]

    st.markdown("### 📊 Prediction Result")
    if pred == 1:
        st.error("🚨 **FRAUDULENT TRANSACTION DETECTED!**", icon="🚨")
    else:
        st.success("✅ **LEGITIMATE TRANSACTION**", icon="✅")

    fig, ax = plt.subplots(figsize=(6, 3))
    categories = ['Not Fraud', 'Fraud']
    colors = ['#4CAF50', '#F44336']
    bars = ax.bar(categories, prob, color=colors, alpha=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    for bar, prob_val in zip(bars, prob):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{prob_val:.4f}", ha='center', fontsize=12)
    st.pyplot(fig)

    st.write(f"🔢 **Not Fraud:** {prob[0]:.4f} | **Fraud:** {prob[1]:.4f}")

else:
    st.info("👉 Enter values or use Quick Actions to load samples.")

# Footer with animated glow effect
st.markdown("""
<style>
.footer-glow {
    font-size: 16px;
    text-align: center;
    margin-top: 30px;
    color: #BB86FC;
    font-weight: bold;
    animation: blink 2s infinite;
}
</style>
<div class="footer-glow">
🌈 Designed with ❤️ by Tafseer Haider
</div>
""", unsafe_allow_html=True)
