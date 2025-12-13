import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

st.title("⚡ Minimal Agricultural Load Forecaster")

# Load your actual data
@st.cache_data
def load_actual_data():
    try:
        df = pd.read_csv("datos_modelo.csv")
        st.info(f"✅ Loaded {len(df)} rows from datos_modelo.csv")
        st.write(f"Columns: {list(df.columns)}")
        
        # Show first few rows
        with st.expander("View Data Sample"):
            st.dataframe(df.head())
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("LightGBM_joblib")
        st.success("✅ LightGBM model loaded")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load data and model
df = load_actual_data()
model = load_model()

if df is not None and model is not None:
    # Simple prediction interface
    st.header("Make Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        hour = st.slider("Hour", 0, 23, 12)
        temperature = st.slider("Temperature (°C)", 0, 40, 25)
    
    with col2:
        humidity = st.slider("Humidity (%)", 0, 100, 60)
        month = st.slider("Month", 1, 12, 3)
    
    if st.button("Predict Load"):
        # Prepare features (simplified)
        features = {
            'temperature_c': temperature,
            'humidity_percent': humidity,
            'hour': hour,
            'day_of_week': datetime.now().weekday(),
            'month': month,
            'agricultural_season': 1 if month in [11, 12, 1, 2, 3, 4] else 0,
            'farming_activity': 2 if month in [3, 4, 5] else 1,
            'is_weekend': 1 if datetime.now().weekday() >= 5 else 0,
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'month_sin': np.sin(2 * np.pi * (month - 1) / 12),
            'month_cos': np.cos(2 * np.pi * (month - 1) / 12)
        }
        
        # Convert to array
        feature_array = np.array([[features[col] for col in [
            'temperature_c', 'humidity_percent', 'hour', 'day_of_week', 
            'month', 'agricultural_season', 'farming_activity',
            'is_weekend', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
        ]]])
        
        # Predict
        prediction = model.predict(feature_array)[0]
        
        st.metric("Predicted Load", f"{prediction:.2f} MW")