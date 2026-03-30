import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# 1. Load the saved AI assets
# The app expects these .pkl files to be in the same folder
try:
    model = joblib.load('liver_model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('encoder.pkl')
    feature_names = joblib.load('feature_names.pkl')
except FileNotFoundError:
    st.error("⚠️ Model files not found! Please run 'python train.py' first to generate liver_model.pkl, scaler.pkl, etc.")
    st.stop()

# Page Configuration
st.set_page_config(page_title="Liver Disease Detector", page_icon="🔬", layout="wide")

st.title("🔬 Liver Disease Detector")
st.write("This application uses Machine Learning to predict the likelihood of liver disease based on clinical laboratory results.")

# 2. Sidebar Input Layout
st.sidebar.header("Patient Clinical Data")
def user_input_features():
    age = st.sidebar.slider("Age", 1, 90, 35)
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    total_bilirubin = st.sidebar.number_input("Total Bilirubin (mg/dL)", 0.1, 30.0, 1.0)
    direct_bilirubin = st.sidebar.number_input("Direct Bilirubin (mg/dL)", 0.1, 15.0, 0.3)
    alkphos = st.sidebar.number_input("Alkaline Phosphotase (IU/L)", 50, 2000, 180)
    sgpt = st.sidebar.number_input("SGPT (ALT) (U/L)", 10, 1000, 40)
    sgot = st.sidebar.number_input("SGOT (AST) (U/L)", 10, 1000, 40)
    total_proteins = st.sidebar.number_input("Total Proteins (g/dL)", 1.0, 10.0, 6.0)
    albumin = st.sidebar.number_input("Albumin (g/dL)", 0.1, 6.0, 3.0)
    ag_ratio = st.sidebar.number_input("A/G Ratio", 0.1, 3.0, 1.0)
    
    # Encode Gender using the saved LabelEncoder
    gender_encoded = encoder.transform([gender])[0]
    
    # Create array in the exact order the model expects
    data = [[age, gender_encoded, total_bilirubin, direct_bilirubin, alkphos, 
             sgpt, sgot, total_proteins, albumin, ag_ratio]]
    return np.array(data)

input_data = user_input_features()

# 3. Prediction and Analysis Logic
if st.button("Run Diagnostic Analysis"):
    # Preprocess the input data
    scaled_data = scaler.transform(input_data)
    
    # Perform Prediction
    prediction = model.predict(scaled_data)
    prediction_proba = model.predict_proba(scaled_data)

    # Display result in a clear box
    st.markdown("---")
    st.subheader("Diagnostic Result")
    
    if prediction[0] == 1:
        st.error(f"### Result: High Risk of Liver Disease Detected")
        st.write(f"The model is **{prediction_proba[0][1]:.2%}** confident in this detection.")
    else:
        st.success(f"### Result: Low Risk / Healthy")
        st.write(f"The model is **{prediction_proba[0][0]:.2%}** confident that the patient is healthy.")

    # 4. Explainability with SHAP
    st.markdown("---")
    st.subheader("AI Decision Analysis (SHAP)")
    st.write("This chart visualizes which specific laboratory values most influenced the AI's prediction for this patient.")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(scaled_data)

    fig, ax = plt.subplots()

    # Logic to handle different SHAP output formats (List vs Array)
    try:
        if isinstance(shap_values, list):
            # For typical binary Random Forest, index 1 is 'Disease'
            values_to_plot = shap_values[1][0]
        else:
            # Handle 3D arrays or single arrays from newer SHAP versions
            if len(shap_values.shape) == 3:
                values_to_plot = shap_values[0, :, 1]
            else:
                values_to_plot = shap_values[0]
    except Exception:
        # Fallback to avoid crashing the UI
        values_to_plot = shap_values[0] if isinstance(shap_values, list) else shap_values

    # Generate and display the plot
    shap.bar_plot(values_to_plot, feature_names=feature_names, show=False)
    st.pyplot(fig)
    
    st.info("💡 **How to read this:** Features extending to the right pushed the model toward a 'Disease' prediction, while features to the left pushed it toward a 'Healthy' prediction.")
