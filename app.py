import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('salary_model.pkl')
scaler = joblib.load('salary_scaler.pkl')

# Define feature names based on your dataset
feature_names = ['Age', 'Workclass', 'Education', 'Education-num', 'Marital-status', 'Occupation', 
                  'Relationship', 'Race', 'Sex', 'Capital-gain', 'Capital-loss', 'Hours-per-week', 'Native-country']

# Streamlit user inputs
st.title('HR Salary Prediction Dashboard')

age = st.number_input('Age', min_value=20, max_value=100, value=30)
workclass = st.selectbox('Workclass', ['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov', 'Unknown'])
education = st.selectbox('Education', ['Bachelors', 'HS-grad', 'Masters', 'Doctorate', 'Assoc-acdm', 'Assoc-voc', '11th', '9th', '7th-8th', 'Some-college', '5th', '10th', '12th', '1st-4th', 'Preschool'])
education_num = st.slider('Education-num', min_value=1, max_value=16, value=9)
marital_status = st.selectbox('Marital-status', ['Never-married', 'Married-civ-spouse', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent'])
occupation = st.selectbox('Occupation', ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty', 'Sales', 'Tech-support', 'Craft-repair', 'Other-service', 'Transport-moving', 'Farming-fishing', 'Machine-op-inspct', 'Protective-serv', 'Armed-Forces'])
relationship = st.selectbox('Relationship', ['Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'])
race = st.selectbox('Race', ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
sex = st.selectbox('Sex', ['Male', 'Female'])
capital_gain = st.number_input('Capital-gain', min_value=0, max_value=100000, value=0)
capital_loss = st.number_input('Capital-loss', min_value=0, max_value=5000, value=0)
hours_per_week = st.number_input('Hours-per-week', min_value=1, max_value=100, value=40)
native_country = st.selectbox('Native-country', ['United-States', 'Mexico', 'Canada', 'Germany', 'India', 'China', 'Other'])

# Prepare input data
input_data = pd.DataFrame([[age, workclass, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country]],
                          columns=feature_names)

# Convert categorical features to numeric
for feature in ['Workclass', 'Education', 'Marital-status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Native-country']:
    input_data[feature], _ = pd.factorize(input_data[feature])

# Scale input data
input_scaled = scaler.transform(input_data)

# Predict salary
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

st.subheader('Prediction')
if prediction[0] == '>50K':
    st.write('Predicted Salary: >50K')
else:
    st.write('Predicted Salary: <=50K')

st.subheader('Prediction Probability')
st.write(f'Probability of earning >50K: {prediction_proba[0][1]:.2f}')
st.write(f'Probability of earning <=50K: {prediction_proba[0][0]:.2f}')
