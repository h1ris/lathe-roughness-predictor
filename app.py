import streamlit as st
import pandas as pd
import joblib

# Set page configuration
st.set_page_config(page_title="Lathe Machining Predictor", layout="centered")

st.title("⚙️ Lathe Machining Predictor")
st.write("Predict Surface Roughness and Cutting Temperature for turning operations.")

# 1. Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('lathe_predictive_model.joblib')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 2. Create the User Interface layout
st.subheader("Input Cutting Parameters")

col1, col2 = st.columns(2)

with col1:
    # Dropdown for Material
    material = st.selectbox("Workpiece Material", ["Mild Steel", "Aluminum"])
    
    # Dropdown for the 5 RPM settings
    rpm = st.selectbox("Spindle Speed (RPM)", [255, 385, 585, 710, 900])

with col2:
    # Dropdown for the 12 standard Feed Rates
    # Note: If your machine's exact gear increments differ slightly, 
    # you can edit these 12 numbers directly in the brackets!
    feed = st.selectbox(
        "Feed Rate (mm/rev)", 
        [0.054, 0.104, 0.154, 0.204, 0.254, 0.304, 0.354, 0.404, 0.454, 0.504, 0.554, 0.606]
    )
    
    # Typing box for Depth of Cut (only number input as requested)
    depth = st.number_input("Depth of Cut (mm)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

# 3. Predict Button and Output Display
if st.button("Predict Outcomes", type="primary"):
    
    # Format the input exactly as the Kaggle model expects it
    input_data = pd.DataFrame({
        'Material': [material],
        'RPM': [rpm],
        'Feed': [feed],
        'Depth': [depth]
    })
    
    # Run the prediction
    prediction = model.predict(input_data)
    predicted_temp = prediction[0][0]
    predicted_roughness = prediction[0][1]
    
    st.divider()
    st.subheader("Predicted Results")
    
    # Display beautiful metric cards
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric(label="🌡️ Estimated Temperature", value=f"{predicted_temp:.1f} °C")
    with res_col2:
        st.metric(label="📏 Surface Roughness (Ra)", value=f"{predicted_roughness:.3f} µm")

    if material == "Aluminum" and rpm < 600:
        st.caption("⚠️ *Note: Low RPM on Aluminum may cause Built-Up Edge (BUE), leading to unpredictable surface finishes.*")
