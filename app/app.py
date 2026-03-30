import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# 1. Load the saved AI assets
try:
    model = joblib.load('liver_model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('encoder.pkl')
    feature_names = joblib.load('feature_names.pkl')
except FileNotFoundError:
    st.error("Error: Model files not found. Please run 'python train.py' first!")
    st.stop()

# Page Setup
st.set_page_config(page_title="Liver Disease Detector", page_icon="🔬")
st.title("🔬 Liver Disease Detector")
st.write("Enter patient laboratory details to check for signs of liver disease.")

# 2. Sidebar Input Layout
st.sidebar.header("Patient Lab Results")
def user_input_features():
    age = st.sidebar.slider("Age", 1, 90, 35)
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    total_bilirubin = st.sidebar.number_input("Total Bilirubin", 0.1, 30.0, 1.0)
    direct_bilirubin = st.sidebar.number_input("Direct Bilirubin", 0.1, 15.0, 0.3)
    alkphos = st.sidebar.number_input("Alkaline Phosphotase", 50, 2000, 180)
    sgpt = st.sidebar.number_input("SGPT (ALT)", 10, 1000, 40)
    sgot = st.sidebar.number_input("SGOT (AST)", 10, 1000, 40)
    total_proteins = st.sidebar.number_input("Total Proteins", 1.0, 10.0, 6.0)
    albumin = st.sidebar.number_input("Albumin", 0.1, 6.0, 3.0)
    ag_ratio = st.sidebar.number_input("A/G Ratio", 0.1, 3.0, 1.0)
    
    # Encode Gender
    gender_encoded = encoder.transform([gender])[0]
    
    # Match the order of the training data
    data = [[age, gender_encoded, total_bilirubin, direct_bilirubin, alkphos, 
             sgpt, sgot, total_proteins, albumin, ag_ratio]]
    return np.array(data)

input_data = user_input_features()

# 3. Prediction Button Logic
if st.button("Run AI Analysis"):
    # Preprocess
    scaled_data = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(scaled_data)
    prediction_proba = model.predict_proba(scaled_data)

    st.subheader("Detection Result")
    if prediction[0] == 1:
        st.error(f"⚠️ **Liver Disease Detected** (Probability: {prediction_proba[0][1]:.2%})")
    else:
        st.success(f"✅ **No Disease Detected** (Probability: {prediction_proba[0][0]:.2%})")

    # 4. SHAP Explanation (Visualizing the Decision)
    st.markdown("---")
    st.subheader("AI Decision Analysis (SHAP)")
    st.write("This chart explains which features most influenced the AI's decision.")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(scaled_data)

    fig, ax = plt.subplots()

    # Robust logic to prevent IndexError across different SHAP/Sklearn versions
    try:
        if isinstance(shap_values, list):
            # List format: pick index 1 for the 'Disease' class
            values_to_plot = shap_values[1][0]
        else:
            # Single array format
            if len(shap_values.shape) == 3:
                values_to_plot = shap_values[0, :, 1]
            else:
                values_to_plot = shap_values[0]
    except Exception:
        # Emergency fallback
        values_to_plot = shap_values[0] if isinstance(shap_values, list) else shap_values

    # Plot the results
    shap.bar_plot(values_to_plot, feature_names=feature_names, show=False)
    st.pyplot(fig)