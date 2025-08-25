
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load('fraud_detection_model.pkl')

# Page config
st.set_page_config(page_title="💳 FraudGuard AI", page_icon="💳", layout="centered")

# Title & Description
st.title("🛡️ FraudGuard AI")
st.markdown("""
> **AI-powered real-time fraud detection system**  
> Enter transaction details to check if it's fraudulent.
""")
st.markdown("---")

# Sidebar
st.sidebar.header("About")
st.sidebar.info("""
This app uses a **Random Forest model** trained on real credit card transactions.
Detects fraud with high precision using ML.
""")

# Input Section
st.header("📋 Enter Transaction Details")

input_data = []
cols = st.columns(3)
for i in range(1, 6):
    with cols[(i-1) % 3]:
        val = st.number_input(f"V{i}", value=0.0, format="%.6f", step=0.01)
        input_data.append(val)

# Amount
st.markdown("### 💰 Transaction Amount")
amount = st.number_input("", min_value=0.0, value=50.0, format="%.2f")
input_data.append(amount)

# Predict Button
st.markdown("---")
if st.button("🔍 Predict Fraud", key="predict", help="Click to run fraud detection"):
    with st.spinner("Analyzing..."):
        input_array = np.array([input_data])
        pred = model.predict(input_array)[0]
        prob = model.predict_proba(input_array)[0]

    # Result
    st.markdown("### 📊 Result")
    if pred == 1:
        st.error("🚨 **FRAUDULENT TRANSACTION DETECTED!**", icon="🚨")
    else:
        st.success("✅ **LEGITIMATE TRANSACTION**", icon="✅")

    # Probability bar chart
    st.markdown("#### Prediction Confidence")
    fig, ax = plt.subplots(figsize=(6, 2))
    categories = ['Not Fraud', 'Fraud']
    colors = ['#4CAF50', '#F44336']
    ax.bar(categories, prob, color=colors, alpha=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    for i, v in enumerate(prob):
        ax.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=12)
    st.pyplot(fig)

    # Raw values
    st.write(f"**Probability:** Not Fraud = {prob[0]:.4f}, Fraud = {prob[1]:.4f}")

else:
    st.info("Enter values and click **Predict** to analyze.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>🔐 Powered by Machine Learning | FraudGuard AI System</p>", unsafe_allow_html=True)
