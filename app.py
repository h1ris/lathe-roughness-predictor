import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(
    page_title="Lathe Machining Predictor",
    page_icon="⚙️",
    layout="wide"
)

# 2. Header & Title
st.title("Lathe Machining Predictor")
st.write("Quantitative estimation of Surface Roughness ($R_a$) and Cutting Temperature ($T$) for turning operations.")

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
st.subheader("Machining Parameters")

col_a, col_b = st.columns(2)

with col_a:
    material = st.selectbox(
        "Workpiece Material",
        options=["Mild Steel", "Aluminum"],
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

# 5. Prediction Logic & Academic Visualization
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
    
    # Clean Metric Output
    st.subheader("Predicted Outcomes")
    col1, col2 = st.columns(2)
    col1.metric(label="Cutting Temperature", value=f"{temp_pred:.2f} °C")
    col2.metric(label="Surface Roughness (Ra)", value=f"{rough_pred:.3f} µm")
    
    st.divider()
    
    # ---------------------------------------------------------
    # NEW GRAPH: PARAMETRIC SENSITIVITY ANALYSIS (ACADEMIC STYLE)
    # ---------------------------------------------------------
    st.subheader("Parametric Sensitivity Analysis")
    st.write(
        "These charts isolate the effects of Spindle Speed and Feed Rate. "
        "The **solid red line** tracks Temperature, while the **dashed blue line** tracks Roughness. "
        "The large markers indicate your current operating parameters."
    )
    
    # Generate data sweeps
    rpm_sweep = np.linspace(100, 3000, 50)
    feed_sweep = np.linspace(0.01, 0.50, 50)
    
    # DataFrames for sweeps holding other variables constant
    df_rpm = pd.DataFrame({'Material': [material]*50, 'RPM': rpm_sweep, 'Feed': [feed]*50, 'Depth': [depth]*50})
    df_feed = pd.DataFrame({'Material': [material]*50, 'RPM': [rpm]*50, 'Feed': feed_sweep, 'Depth': [depth]*50})
    
    # Predict over sweeps
    pred_rpm = model.predict(df_rpm)
    pred_feed = model.predict(df_feed)
    
    # Setup Matplotlib Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # --- PLOT 1: EFFECT OF SPINDLE SPEED ---
    ax1_temp = ax1
    ax1_rough = ax1.twinx() # Create dual Y-axis
    
    # Plot lines
    ax1_temp.plot(rpm_sweep, pred_rpm[:, 0], color='firebrick', linewidth=2.5, label='Temperature (°C)')
    ax1_rough.plot(rpm_sweep, pred_rpm[:, 1], color='navy', linewidth=2.5, linestyle='--', label='Roughness (µm)')
    
    # Plot current operating point
    ax1_temp.plot(rpm, temp_pred, marker='o', markersize=10, color='firebrick', markeredgecolor='black')
    ax1_rough.plot(rpm, rough_pred, marker='s', markersize=9, color='navy', markeredgecolor='black')
    
    # Format axes
    ax1.set_xlabel('Spindle Speed (RPM)', fontsize=11, fontweight='bold')
    ax1_temp.set_ylabel('Cutting Temperature (°C)', color='firebrick', fontsize=11, fontweight='bold')
    ax1_rough.set_ylabel('Surface Roughness (µm)', color='navy', fontsize=11, fontweight='bold')
    ax1.set_title('Effect of Spindle Speed\n(Constant Feed & Depth)', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    # --- PLOT 2: EFFECT OF FEED RATE ---
    ax2_temp = ax2
    ax2_rough = ax2.twinx() # Create dual Y-axis
    
    # Plot lines
    ax2_temp.plot(feed_sweep, pred_feed[:, 0], color='firebrick', linewidth=2.5)
    ax2_rough.plot(feed_sweep, pred_feed[:, 1], color='navy', linewidth=2.5, linestyle='--')
    
    # Plot current operating point
    ax2_temp.plot(feed, temp_pred, marker='o', markersize=10, color='firebrick', markeredgecolor='black')
    ax2_rough.plot(feed, rough_pred, marker='s', markersize=9, color='navy', markeredgecolor='black')
    
    # Format axes
    ax2.set_xlabel('Feed Rate (mm/rev)', fontsize=11, fontweight='bold')
    ax2_temp.set_ylabel('Cutting Temperature (°C)', color='firebrick', fontsize=11, fontweight='bold')
    ax2_rough.set_ylabel('Surface Roughness (µm)', color='navy', fontsize=11, fontweight='bold')
    ax2.set_title('Effect of Feed Rate\n(Constant RPM & Depth)', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.7)
    
    # Universal Legend
    lines_1, labels_1 = ax1_temp.get_legend_handles_labels()
    lines_2, labels_2 = ax1_rough.get_legend_handles_labels()
    fig.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2, frameon=False, fontsize=11)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18) # Make room for legend
    st.pyplot(fig)
