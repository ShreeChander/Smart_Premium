import streamlit as st 
import pandas as pd
import joblib

# Expand page width
st.set_page_config(page_title="Smart Premium Predictor", layout="wide")

# Load the trained model
model = joblib.load("ml_model.pkl")

# Page title and description
st.title("Smart Premium Predictor")
st.write("Easily estimate your insurance premium based on key personal and policy details.")

# Default values
default_values = {
    "Age": 19,
    "Gender": "Female",
    "Annual Income": 10049,
    "Marital Status": "Married",
    "Number of Dependents": 1,
    "Education Level": "Bachelor's",
    "Occupation": "Self-Employed",
    "Health Score": 22.6,
    "Location": "Urban",
    "Policy Type": "Premium",
    "Previous Claims": 2,
    "Vehicle Age": 17,
    "Credit Score": 372,
    "Insurance Duration": 5,
    "Property Type": "House",
    "Customer Feedback": "Poor",
    "Smoking Status": "No",
    "Exercise Frequency": "Weekly"
}

# Checkbox for using default values
use_defaults = st.checkbox("Use Default Values")

# User input form
with st.form(key='premium_form'):
    col1, col2, col3 = st.columns([3, 3, 3])

    with col1:
        Age = st.number_input("Age", min_value=0, max_value=100, step=1, value=default_values["Age"] if use_defaults else 0)
        Gender = st.selectbox("Gender", ["Female", "Male"], index=0 if use_defaults else 0)
        Annual_Income = st.number_input("Annual Income", min_value=0, max_value=1000000, step=1000, value=default_values["Annual Income"] if use_defaults else 0)
        Marital_Status = st.selectbox("Marital Status", ["Married", "Divorced", "Single"], index=0 if use_defaults else 0)

    with col2:
        Number_of_Dependents = st.number_input("Number of Dependents", min_value=0, max_value=4, step=1, value=default_values["Number of Dependents"] if use_defaults else 0)
        Education_Level = st.selectbox("Education Level", ["Bachelor's", "Master's", "High School", "PhD"], index=0 if use_defaults else 0)
        Occupation = st.selectbox("Occupation", ["Self-Employed", "Employed", "Unemployed"], index=0 if use_defaults else 0)
        Health_Score = st.number_input("Health Score", min_value=0.0, max_value=60.0, step=0.5, value=default_values["Health Score"] if use_defaults else 0.0)

    with col3:
        Location = st.selectbox("Location", ["Urban", "Rural", "Suburban"], index=0 if use_defaults else 0)
        Policy_Type = st.selectbox("Policy Type", ["Premium", "Comprehensive", "Basic"], index=0 if use_defaults else 0)
        Previous_Claims = st.number_input("Previous Claims", min_value=0, max_value=10, step=1, value=default_values["Previous Claims"] if use_defaults else 0)
        Vehicle_Age = st.number_input("Vehicle Age", min_value=0, max_value=20, step=1, value=default_values["Vehicle Age"] if use_defaults else 0)
       

    col4, col5, col6 = st.columns([3, 3, 3])
    with col4:
        Property_Type = st.selectbox("Property Type", ["House", "Apartment", "Condo"], index=0 if use_defaults else 0)
        Insurance_Duration = st.number_input("Insurance Duration", min_value=1, max_value=10, step=1, value=default_values["Insurance Duration"] if use_defaults else 1)
    with col5:
        Customer_Feedback = st.selectbox("Customer Feedback", ["Poor", "Average", "Good"], index=0 if use_defaults else 0)
        Credit_Score = st.number_input("Credit Score", min_value=0, max_value=1000, step=50, value=default_values["Credit Score"] if use_defaults else 0)
        
    with col6:
        Smoking_Status = st.selectbox("Smoking Status", ["No", "Yes"], index=0 if use_defaults else 0)
        Exercise_Frequency = st.selectbox("Exercise Frequency", ["Weekly", "Monthly", "Daily", "Rarely"], index=0 if use_defaults else 0)

    submit_button = st.form_submit_button(label='Predict Premium Amount')

if submit_button:
    df = pd.DataFrame({
        'Age': [Age], 'Gender': [Gender], 'Annual Income': [Annual_Income],
        'Marital Status': [Marital_Status], 'Number of Dependents': [Number_of_Dependents],
        'Education Level': [Education_Level], 'Occupation': [Occupation],
        'Health Score': [Health_Score], 'Location': [Location], 'Policy Type': [Policy_Type],
        'Previous Claims': [Previous_Claims], 'Vehicle Age': [Vehicle_Age],
        'Credit Score': [Credit_Score], 'Insurance Duration': [Insurance_Duration],
        'Property Type': [Property_Type], 'Customer Feedback': [Customer_Feedback],
        'Smoking Status': [Smoking_Status], 'Exercise Frequency': [Exercise_Frequency]
    })
    premium_amt = model.predict(df)
    st.subheader(f"Predicted Premium Amount: **${premium_amt[0]:.2f}**")

# Explanation Section
st.markdown("---")
st.write(
    """
    ### How Premium is Calculated?
    Insurance premium is influenced by various factors such as age, income, lifestyle habits, and past claims. Here’s how each factor contributes:
    - **Age**: Younger individuals may have lower premiums.
    - **Annual Income**: Higher income can indicate more expensive policies.
    - **Health Score & Lifestyle**: Smokers or those with lower exercise frequency may face higher premiums.
    - **Previous Claims**: A history of claims can increase future premiums.
    - **Policy Type**: Premium and comprehensive policies tend to have higher costs.
    
    🔍 Ensure accuracy in details for a precise premium estimate!
    """
)
