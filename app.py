import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# Load model and preprocessing objects
with open('XGBRegressor.pkl', 'rb') as f:
    model = pickle.load(f)
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Expected feature order for the model
expected_features = [
    'Age', 'Gender', 'Annual Income', 'Number of Dependents', 'Education Level', 
    'Health Score', 'Policy Type', 'Previous Claims', 'Vehicle Age', 'Credit Score', 
    'Insurance Duration', 'Customer Feedback', 'Smoking Status', 'Exercise Frequency', 
    'Marital Status_Divorced', 'Marital Status_Married', 'Marital Status_Single', 
    'Occupation_Employed', 'Occupation_Self-Employed', 'Occupation_Unemployed', 
    'Location_Rural', 'Location_Suburban', 'Location_Urban', 'Property Type_Apartment', 
    'Property Type_Condo', 'Property Type_House', 'Year', 'Month'
]

st.title("Insurance Claim Prediction")

def get_user_input():
    st.header("Personal Information")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 19)  # Default: 19
        gender = st.selectbox("Gender", ["Male", "Female"], index=1)  # Default: Female
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], index=1)  # Default: Married
        dependents = st.number_input("Number of Dependents", 0, 10, 1)  # Default: 1
        
    with col2:
        income = st.number_input("Annual Income ($)", 1000, 500000, 10049)  # Default: 10049
        occupation = st.selectbox("Occupation", ["Unemployed", "Self-Employed", "Employed"], index=1)  # Default: Self-Employed
        education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"], index=1)  # Default: Bachelor's
        location = st.selectbox("Location", ["Urban", "Suburban", "Rural"], index=0)  # Default: Urban
    
    st.header("Health & Insurance Details")
    col1, col2 = st.columns(2)
    with col1:
        health = st.slider("Health Score", 0, 100, 23)  # Default: 22.59876067 (Rounded to 23)
        smoking = st.selectbox("Smoking Status", ["Yes", "No"], index=1)  # Default: No
        exercise = st.selectbox("Exercise Frequency", ["Rarely", "Monthly", "Weekly", "Daily"], index=2)  # Default: Weekly
        
    with col2:
        claims = st.number_input("Previous Claims", 0, 50, 2)  # Default: 2
        vehicle_age = st.number_input("Vehicle Age", 0, 30, 17)  # Default: 17
        credit = st.slider("Credit Score", 300, 850, 372)  # Default: 372
        insurance_years = st.number_input("Insurance Duration (years)", 0, 50, 5)  # Default: 5
    
    st.header("Policy Details")
    policy_type = st.selectbox("Policy Type", ["Basic", "Standard", "Comprehensive", "Premium"], index=3)  # Default: Premium
    property_type = st.selectbox("Property Type", ["Apartment", "House", "Condo"], index=1)  # Default: House
    feedback = st.selectbox("Customer Feedback", ["Poor", "Average", "Good"], index=0)  # Default: Poor
    
    # Get current year and month
    year = datetime.now().year
    month = datetime.now().month

    data = {
        "Age": age,
        "Gender": 1 if gender == "Male" else 0,  # Encode Gender
        "Annual Income": income,
        "Number of Dependents": dependents,
        "Education Level": education,
        "Health Score": health,
        "Policy Type": policy_type,
        "Previous Claims": claims,
        "Vehicle Age": vehicle_age,
        "Credit Score": credit,
        "Insurance Duration": insurance_years,
        "Customer Feedback": feedback,
        "Smoking Status": 1 if smoking == "Yes" else 0,  # Encode Smoking Status
        "Exercise Frequency": exercise,
        "Marital Status": marital_status,
        "Occupation": occupation,
        "Location": location,
        "Property Type": property_type,
        "Year": year,
        "Month": month
    }
    
    return pd.DataFrame([data])

# Get input
input_df = get_user_input()

# Add prediction button
if st.button("Predict"):
    try:
        # Encode categorical variables using the fitted encoder
        categorical_cols = ["Education Level", "Policy Type", "Customer Feedback", "Exercise Frequency"]
        input_df[categorical_cols] = encoder.transform(input_df[categorical_cols])

        # One-hot encode remaining categorical features
        ohe_cols = ["Marital Status", "Occupation", "Location", "Property Type"]
        input_df = pd.get_dummies(input_df, columns=ohe_cols)

        # Ensure all expected columns exist
        for feature in expected_features:
            if feature not in input_df.columns:
                input_df[feature] = 0  # Add missing columns with default value 0

        # Reorder columns to match model training order
        input_df = input_df[expected_features]

        # Scale numerical features
        num_cols = ["Age", "Annual Income", "Number of Dependents", "Health Score", 
                    "Previous Claims", "Vehicle Age", "Credit Score", "Insurance Duration"]
        input_df[num_cols] = scaler.transform(input_df[num_cols])

        # Make prediction (scaled output)
        predicted_value_scaled = model.predict(input_df).reshape(-1, 1)  # Ensure correct shape
        
        # Create a dummy array with the same shape as the scaler's training data
        dummy_array = np.zeros((1, 8))  # Assuming 8 numerical features were scaled
        dummy_array[0, -1] = predicted_value_scaled[0, 0]  # Insert predicted premium amount in last column
        
        # Apply inverse transform
        predicted_value_original = scaler.inverse_transform(dummy_array)[0, -1]  # Extract correct value
        
        # Display results
        st.success("Prediction completed successfully!")
        st.subheader("Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Premium", f"${predicted_value_original:.2f}")  # Display actual premium amount

        prediction = 1 if predicted_value_original > 2000 else 0  # Example threshold for claim risk
        with col2:
            st.metric("Claim Risk", "Likely" if prediction else "Unlikely")

        if prediction:
            st.warning("High claim risk detected based on input parameters")
        else:
            st.success("Low claim risk predicted")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
