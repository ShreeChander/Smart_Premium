import streamlit as st 
import pandas as pd
import joblib

model = joblib.load("ml_model.pkl")
st.title("Smart Premium ")
form = st.form(key='registration_form')
Age = form.number_input("Age", min_value=0, max_value=100, step=1)
Gender = form.selectbox("Choose Gender",('--None--','Female', 'Male'))
Annual_Income = form.number_input("Annual_Income", min_value=0, max_value=1000000, step=1000)
Marital_Status = form.selectbox("Choose Marital status",('--None--','Married', 'Divorced', 'Single'))
Number_of_Dependents = form.number_input("Number of Dependents", min_value=0, max_value=4, step=1)
Education_Level= form.selectbox("Choose Education Level",('--None--',"Bachelor's", "Master's", 'High School', 'PhD'))
Occupation = form.selectbox("Choose Occupation",('--None--','Self-Employed','Employed', 'Unemployed'))
Health_Score = form.number_input("Health Score", min_value=0.0, max_value=60.0, step=0.5)
Location  = form.selectbox("Choose Location",('--None--','Urban', 'Rural', 'Suburban'))
Policy_Type = form.selectbox("Choose Police type",('--None--','Premium', 'Comprehensive', 'Basic'))
Previous_Claims = form.number_input("Previous Claims", min_value=0, max_value=10, step=1)
Vehicle_Age = form.number_input("Vehicle Age", min_value=0, max_value=20, step=1)
Credit_Score = form.number_input("Credit Score", min_value=0, max_value=1000, step=50)
Insurance_Duration = form.number_input("Insurance Duration", min_value=1, max_value=10, step=1)
Property_Type = form.selectbox("Choose property type",('--None--','House', 'Apartment', 'Condo'))
Customer_Feedback = form.selectbox("Choose feedback",('--None--','Poor', 'Average', 'Good'))
Smoking_Status = form.selectbox("Choose Smoking Status",('--None--','No', 'Yes'))
Exercise_Frequency = form.selectbox("Choose Exercise Frequency",('--None--','Weekly', 'Monthly', 'Daily', 'Rarely'))

submit_button = form.form_submit_button(label='Predict Premium Amount')

if(submit_button)  :
    df=df=pd.DataFrame({'Age':[Age],'Gender':[Gender],'Annual Income':[Annual_Income],
                        'Marital Status':[Marital_Status],'Number of Dependents':[Number_of_Dependents],
                        'Education Level':[Education_Level],'Occupation':[Occupation],
                        'Health Score':[Health_Score],'Location':[Location],'Policy Type':[Policy_Type],
                        'Previous Claims':[Previous_Claims],'Vehicle Age':[Vehicle_Age],
                        'Credit Score':[Credit_Score],'Insurance Duration':[Insurance_Duration],
                        'Property Type':[Property_Type],'Customer Feedback':[Customer_Feedback],
                        'Smoking Status':[Smoking_Status],'Exercise Frequency':[Exercise_Frequency]})
    
    premium_amt=model.predict(df)
    st.write("Predicted Premium amount is ",premium_amt[0])
