# Smart Premium Prediction

## Overview
The Smart Premium Prediction project aims to predict insurance premium amounts based on customer demographics, financial details, and lifestyle habits. The project follows an end-to-end machine learning workflow, including Exploratory Data Analysis (EDA), model training, pipeline creation, MLflow tracking, and deployment using Streamlit.

## Project Components
1. **Exploratory Data Analysis (EDA)**
2. **Model Training & Pipeline Setup**
3. **MLflow for Experiment Tracking**
4. **Deployment using Streamlit**

---

## 1. Exploratory Data Analysis (EDA)
EDA was conducted to understand the dataset, identify missing values, detect outliers, and find correlations between features.

### Steps:
- **Loading Data**: Data is loaded from structured sources like CSV or SQL.
- **Feature Distribution**: Histograms and boxplots were used to analyze numerical features.
- **Correlation Analysis**: Heatmaps and pair plots were created to visualize relationships.
- **Missing Data Handling**: Imputation strategies were applied for missing values.
- **Feature Engineering**: Categorical features were encoded appropriately.

Visualization tools: **Matplotlib, Seaborn, Plotly**

---

## 2. Model Training & Pipeline Setup
A machine learning model was trained to predict insurance premiums based on multiple input features.

### Steps:
- **Data Preprocessing**:
  - Label Encoding for binary categorical features.
  - One-Hot Encoding for multi-category features.
  - Scaling numerical features.
- **Model Selection**: Regression models like RandomForest, XGBoost, and Linear Regression were tested.
- **Hyperparameter Tuning**: GridSearchCV and RandomizedSearchCV were used to optimize model performance.
- **Model Evaluation**:
  - Metrics: RMSE, MAE, R-squared score.
  - Cross-validation was performed to assess generalization.
- **Pipeline Creation**:
  - Scikit-learn `Pipeline` was used to integrate preprocessing and model steps.
  
---

## 3. MLflow for Experiment Tracking
MLflow was used to track experiments and store trained models.

### Steps:
- **MLflow Setup**: Installed and initialized MLflow tracking server.
- **Logging Experiments**:
  - Tracked parameters, metrics, and artifacts.
  - Stored trained models for easy retrieval.
- **Model Registry**: Used MLflow's model registry to manage versioning of trained models.

---

## 4. Deployment using Streamlit
The trained model is deployed as an interactive web app using **Streamlit**.

### Steps:
- **Frontend Development**:
  - User-friendly UI with input fields for customer details.
  - A "Predict" button to generate insurance premium predictions.
- **Backend Integration**:
  - Loaded the trained model (`joblib` format) in Streamlit.
  - Preprocessed user input to match training format.
  - Displayed the predicted premium amount.
- **UI Enhancements**:
  - Streamlit themes and markdown for better visualization.
  - Custom CSS to hide default UI elements.

---

## Deployment Steps
1. **Run the Streamlit App Locally**
```bash
streamlit run app.py
```
2. **Deploy on Cloud (AWS, GCP, or Heroku)**
   - Create a virtual environment and install dependencies.
   - Deploy using `streamlit sharing`, Heroku, or AWS EC2.

---

## Live Demo
You can access the deployed Streamlit application here:
[Smart Premium App](https://smart-premium-app.streamlit.app/)

## Requirements
All dependencies required to run this project are listed in `requirements.txt`.

---

## Future Improvements
- **Enhance Model Accuracy**: Fine-tune hyperparameters and explore deep learning models.
- **Add More Features**: Include additional lifestyle and financial attributes.
- **Improve UI/UX**: Enhance the user experience with better visuals and interactive graphs.

---

## Technologies Used
- **Python**: Pandas, NumPy, Scikit-Learn, XGBoost
- **Visualization**: Matplotlib, Seaborn, Plotly
- **MLflow**: Experiment tracking & model registry
- **Streamlit**: Web app deployment
- **Cloud Deployment**: AWS/GCP/Heroku (Optional)

This project provides a robust framework for predicting insurance premiums and can be further improved with additional data and model optimization.

