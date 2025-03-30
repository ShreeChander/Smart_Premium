import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model, encoder, and scaler
with open('XGBoost.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoder.pkl', 'rb') as f:
    ordinal_encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Updated expected features (without Month/Year if they weren't in training)
expected_features = [
    "Age", "Annual Income", "Number of Dependents", "Health Score",
    "Previous Claims", "Vehicle Age", "Credit Score", "Insurance Duration",
    "Marital Status_Divorced", "Marital Status_Married", "Marital Status_Single",
    "Occupation_Employed", "Occupation_Self-Employed", "Occupation_Unemployed",
    "Location_Rural", "Location_Suburban", "Location_Urban",
    "Property Type_Apartment", "Property Type_Condo", "Property Type_House",
    "Gender_Female", "Gender_Male",
    "Education Level", "Policy Type", "Customer Feedback", 
    "Smoking Status_Non-Smoker", "Smoking Status_Smoker",
    "Exercise Frequency"
]

# Streamlit UI
st.title("🔮 Insurance Claim Prediction")

def user_input():
    # Numeric inputs
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Annual Income ($)", min_value=1000, max_value=500000, value=50000)
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1)
    health_score = st.slider("Health Score", min_value=0, max_value=100, value=70)
    previous_claims = st.number_input("Previous Claims", min_value=0, max_value=50, value=0)
    vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=30, value=5)
    credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=700)
    insurance_duration = st.number_input("Insurance Duration (years)", min_value=0, max_value=50, value=5)

    # Categorical inputs
    gender = st.selectbox("Gender", ['Male', 'Female'])
    marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
    education = st.selectbox("Education Level", ['High School', "Bachelor's", "Master's", 'PhD'])
    occupation = st.selectbox("Occupation", ['Unemployed', 'Self-Employed', 'Employed'])
    location = st.selectbox("Location", ['Urban', 'Rural', 'Suburban'])
    policy_type = st.selectbox("Policy Type", ['Basic', 'Standard', 'Premium'])
    feedback = st.selectbox("Customer Feedback", ['Poor', 'Average', 'Good'])
    smoking_status = st.selectbox("Smoking Status", ['Smoker', 'Non-Smoker'])
    exercise_frequency = st.selectbox("Exercise Frequency", ['Rarely', 'Monthly', 'Weekly', 'Daily'])
    property_type = st.selectbox("Property Type", ['Apartment', 'House', 'Condo'])

    # Create DataFrame (without Month/Year)
    data = pd.DataFrame({
        "Age": [age],
        "Annual Income": [income],
        "Number of Dependents": [dependents],
        "Health Score": [health_score],
        "Previous Claims": [previous_claims],
        "Vehicle Age": [vehicle_age],
        "Credit Score": [credit_score],
        "Insurance Duration": [insurance_duration],
        "Gender": [gender],
        "Marital Status": [marital_status],
        "Education Level": [education],
        "Occupation": [occupation],
        "Location": [location],
        "Policy Type": [policy_type],
        "Customer Feedback": [feedback],
        "Smoking Status": [smoking_status],
        "Exercise Frequency": [exercise_frequency],
        "Property Type": [property_type]
    })

    return data

# Get user input
data = user_input()

# Define which columns should be ordinal encoded vs one-hot encoded
ordinal_encode_cols = ["Education Level", "Policy Type", "Customer Feedback", "Exercise Frequency"]
one_hot_encode_cols = ["Gender", "Marital Status", "Occupation", "Location", 
                      "Smoking Status", "Property Type"]

try:
    # Ordinal encode specified columns
    data[ordinal_encode_cols] = ordinal_encoder.transform(data[ordinal_encode_cols])
    
    # One-hot encode remaining categorical columns
    data = pd.get_dummies(data, columns=one_hot_encode_cols)
    
    # Scale numerical features (without Month/Year)
    numerical_cols = ["Age", "Annual Income", "Number of Dependents", "Health Score", 
                     "Previous Claims", "Vehicle Age", "Credit Score", "Insurance Duration"]
    data[numerical_cols] = scaler.transform(data[numerical_cols])
    
    # Ensure all expected columns exist (add missing ones as 0)
    for col in expected_features:
        if col not in data.columns:
            data[col] = 0
    
    # Reorder columns to match training data
    data = data[expected_features]
    
    # Convert all columns to float
    data = data.astype(float)
    
    # Predict
    prediction = model.predict(data)
    prediction_proba = model.predict_proba(data)
    
    # Show results
    st.subheader("🔮 Prediction")
    st.write(f"Claim Probability: **{round(prediction_proba[0][1] * 100, 2)}%**")
    
    if prediction[0] == 1:
        st.success("✅ The model predicts that a **claim is likely** to be filed.")
    else:
        st.error("❌ The model predicts that a **claim is unlikely** to be filed.")
        
except Exception as e:
    st.error(f"An error occurred during prediction: {str(e)}")
    st.warning("Please check that:")
    st.warning("1. All categorical values match exactly what was used in training")
    st.warning("2. All required features are present")