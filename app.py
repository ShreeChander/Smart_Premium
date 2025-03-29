import streamlit as st
import pandas as pd
import numpy as np
import pickle
import mlflow.pyfunc

# Load the trained model
model_uri = "runs:/899d644802f5443c9bc96e02144efddf/xgboost_model"
model = mlflow.pyfunc.load_model(model_uri)

# Load OneHotEncoder
with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Define expected numerical & categorical features
num_cols = ['Age', 'Annual Income', 'Number of Dependents', 'Health Score',
            'Previous Claims', 'Vehicle Age', 'Credit Score', 'Insurance Duration']

cat_cols = ['Gender', 'Marital Status', 'Education Level', 'Occupation', 'Location',
            'Policy Type', 'Customer Feedback', 'Smoking Status', 'Exercise Frequency', 'Property Type']

# Load expected feature order
expected_features = num_cols + list(encoder.get_feature_names_out(cat_cols))

# Streamlit UI
st.title("Insurance Premium Prediction")

# **Updated Default Values**
default_values = {
    'Age': 19,
    'Annual Income': 10049,
    'Number of Dependents': 1,
    'Health Score': 22.59876,
    'Previous Claims': 2,
    'Vehicle Age': 17,
    'Credit Score': 372,
    'Insurance Duration': 5,
    'Gender': 'Female',
    'Marital Status': 'Married',
    'Education Level': "Bachelor's",
    'Occupation': "Self-Employed",
    'Location': "Urban",
    'Policy Type': "Premium",
    'Customer Feedback': "Poor",
    'Smoking Status': "No",
    'Exercise Frequency': "Weekly",
    'Property Type': "House"
}

# User input fields with default values
user_inputs = {}
for col in num_cols:
    user_inputs[col] = st.number_input(f"Enter {col}", min_value=0.0, value=float(default_values[col]), format="%.2f")

for col in cat_cols:
    user_inputs[col] = st.selectbox(f"Select {col}", 
                                    options=['Unknown', 'Male', 'Female'] if col == 'Gender' else ['Unknown', 'Yes', 'No', 'Married', 'Single', 'Divorced'],
                                    index=['Unknown', 'Male', 'Female'].index(default_values[col]) if col == 'Gender' else 
                                          ['Unknown', 'Yes', 'No', 'Married', 'Single', 'Divorced'].index(default_values[col]) if default_values[col] in ['Yes', 'No', 'Married', 'Single', 'Divorced', 'Unknown'] else 0)

# Convert input to DataFrame
input_data = pd.DataFrame([user_inputs])

# **Handle Categorical Encoding**
encoded_cat = encoder.transform(input_data[cat_cols])
encoded_cat_df = pd.DataFrame(encoded_cat.toarray(), columns=encoder.get_feature_names_out(cat_cols))

# Merge encoded categorical & numerical data
input_data = input_data.drop(columns=cat_cols)
input_data = pd.concat([input_data, encoded_cat_df], axis=1)

# **Ensure Feature Order Matches Model**
input_data = input_data.reindex(columns=expected_features, fill_value=0)

# **Check Feature Shape**
if input_data.shape[1] != len(expected_features):
    st.error(f"Feature shape mismatch: Expected {len(expected_features)}, got {input_data.shape[1]}")
else:
    if st.button("Predict Premium"):
        prediction = model.predict(input_data)
        st.success(f"Predicted Insurance Premium: ${prediction[0]:.2f}")
