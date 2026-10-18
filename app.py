import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(
    page_title="Lathe Machining Predictor",
    page_icon="⚙️",
    layout="centered"
)

# 2. Header & Title
st.title("⚙️ Lathe Machining Predictor")
st.write("Predict Surface Roughness ($R_a$) and Cutting Temperature ($T$) for turning operations.")

# 3. Load Model Pipeline
@st.cache_resource
def load_model():
    return joblib.load('lathe_predictive_model.joblib')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model file: {e}")
    st.stop()

# 4. Input Controls
st.subheader("Input Cutting Parameters")

col_a, col_b = st.columns(2)

with col_a:
    material = st.selectbox(
        "Workpiece Material",
        options=["Mild Steel", "Aluminum", "Stainless Steel", "Brass"],
        index=0
    )
    rpm = st.number_input(
        "Spindle Speed (RPM)",
        min_value=100,
        max_value=3000,
        value=255,
        step=10
    )

with col_b:
    feed = st.number_input(
        "Feed Rate (mm/rev)",
        min_value=0.01,
        max_value=1.00,
        value=0.054,
        step=0.005,
        format="%.3f"
    )
    depth = st.number_input(
        "Depth of Cut (mm)",
        min_value=0.1,
        max_value=5.0,
        value=0.50,
        step=0.1,
        format="%.2f"
    )

st.write("")

# 5. Prediction Logic & Visual Analytics
if st.button("Predict Outcomes", type="primary"):
    # Create input DataFrame matching model features
    input_data = pd.DataFrame({
        'Material': [material],
        'RPM': [rpm],
        'Feed': [feed],
        'Depth': [depth]
    })
    
    # Run prediction
    prediction = model.predict(input_data)
    temp_pred = prediction[0][0]
    rough_pred = prediction[0][1]
    
    st.divider()
    
    # Styled Metric Cards
    st.subheader("🎯 Prediction Results")
    col1, col2 = st.columns(2)
    col1.metric(
        label="Cutting Temperature", 
        value=f"{temp_pred:.2f} °C", 
        delta="High Heat" if temp_pred > 40 else "Normal", 
        delta_color="inverse"
    )
    col2.metric(
        label="Surface Roughness", 
        value=f"{rough_pred:.3f} µm", 
        delta="Optimal Finish" if rough_pred < 4.0 else "Rough Finish", 
        delta_color="normal"
    )
    
    st.divider()
    
    # Dynamic Parameter Sensitivity Chart
    st.subheader("📊 Machining Analytics: Feed Rate vs. Roughness")
    st.caption("Visualizing how varying Feed Rate alters surface roughness under your selected Material, Speed, and Depth.")
    
    feed_range = np.linspace(0.02, 0.25, 20)
    
    simulated_data = pd.DataFrame({
        'Material': [material] * 20,
        'RPM': [rpm] * 20,
        'Feed': feed_range,
        'Depth': [depth] * 20
    })
    
    simulated_preds = model.predict(simulated_data)
    simulated_roughness = simulated_preds[:, 1]
    
    chart_data = pd.DataFrame({
        "Feed Rate (mm/rev)": feed_range,
        "Roughness (µm)": simulated_roughness
    }).set_index("Feed Rate (mm/rev)")
    
    st.line_chart(chart_data, y="Roughness (µm)", color="#ff4b4b")
