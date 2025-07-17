
import streamlit as st
import pandas as pd
import joblib
from src.data_loader import load_data

st.set_page_config(page_title="🏠 House Price Predictor", layout="centered")

st.title("🏠 House Price Predictor")
st.markdown("An interactive tool to predict house prices using a trained ML model.")

# Load the model
@st.cache_resource
def load_model():
    return joblib.load("models/trained_model.pkl")

model = load_model()

# Load data to get feature names
df = load_data()
features = df.drop("MEDV", axis=1).columns.tolist()

# Sidebar for input
st.sidebar.header("📋 Input Features")
input_data = {}
for feature in features:
    min_val = float(df[feature].min())
    max_val = float(df[feature].max())
    mean_val = float(df[feature].mean())
    input_data[feature] = st.sidebar.slider(
        label=feature,
        min_value=min_val,
        max_value=max_val,
        value=mean_val
    )

# Predict button
if st.button("Predict Price"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated House Price: ${prediction:.2f}K")
