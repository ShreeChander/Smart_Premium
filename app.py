import pandas as pd
import mlflow.pyfunc
import streamlit as st
# Load the trained model
model_uri = 'runs:/899d644802f5443c9bc96e02144efddf/xgboost_model'
model = mlflow.pyfunc.load_model(model_uri)

# Define expected features based on training data
expected_features = [
    "Age", "Gender", "Annual Income", "Marital Status", "Number of Dependents",
    "Education Level", "Occupation", "Health Score", "Location", "Policy Type",
    "Previous Claims", "Vehicle Age", "Credit Score", "Insurance Duration",
    "Smoking Status", "Exercise Frequency", "Property Type"
]

# Function to get user input
def get_user_input():
    data = {
        "Age": st.number_input("Age", min_value=18, max_value=100, value=30),
        "Gender": st.selectbox("Gender", ["Male", "Female"]),
        "Annual Income": st.number_input("Annual Income", min_value=5000, max_value=500000, value=30000),
        "Marital Status": st.selectbox("Marital Status", ["Single", "Married", "Divorced"]),
        "Number of Dependents": st.number_input("Number of Dependents", min_value=0, max_value=10, value=1),
        "Education Level": st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"]),
        "Occupation": st.selectbox("Occupation", ["Employed", "Self-Employed", "Unemployed"]),
        "Health Score": st.number_input("Health Score", min_value=0.0, max_value=100.0, value=50.0),
        "Location": st.selectbox("Location", ["Urban", "Suburban", "Rural"]),
        "Policy Type": st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"]),
        "Previous Claims": st.number_input("Previous Claims", min_value=0, max_value=10, value=1),
        "Vehicle Age": st.number_input("Vehicle Age", min_value=0, max_value=30, value=5),
        "Credit Score": st.number_input("Credit Score", min_value=300, max_value=850, value=600),
        "Insurance Duration": st.number_input("Insurance Duration", min_value=1, max_value=10, value=5),
        "Smoking Status": st.selectbox("Smoking Status", ["Yes", "No"]),
        "Exercise Frequency": st.selectbox("Exercise Frequency", ["Daily", "Weekly", "Monthly", "Rarely"]),
        "Property Type": st.selectbox("Property Type", ["House", "Apartment", "Condo"]),
    }

    # Convert to DataFrame and ensure column order
    user_df = pd.DataFrame([data])
    user_df = user_df[expected_features]  # Ensure correct column order

    return user_df

# Streamlit UI
st.title("🚀 Insurance Premium Prediction")

# Get user input
user_input = get_user_input()

# Make Prediction when button is clicked
if st.button("Predict Premium Amount"):
    prediction = model.predict(user_input)
    st.success(f"💰 Predicted Premium Amount: **${prediction[0]:,.2f}**")