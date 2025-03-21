import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import pickle as pkl
import tensorflow
from keras.models import load_model

# Loading model and encoder, scaler
model = load_model('ann_model.h5')
with open('one_hot_encoded_geo.pkl', 'rb') as f:
    one_hot_encode = pkl.load(f)
with open('label_encoded_gender.pkl', 'rb') as f:
    label_encode = pkl.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pkl.load(f)

# Now creating frontend using Streamlit
st.title('Customer Churn Prediction')
st.header('Predicts whether a customer would churn or not')

# User input
geography = st.selectbox('Geography', one_hot_encode.categories_[0])
gender = st.selectbox('Gender', label_encode.classes_)
age = st.slider('Age', 18, 85)
tenure = st.slider('Tenure', 0, 10)
balance = st.number_input('Balance')
no_of_products = st.slider('No of products', 1, 10)
has_cr_card = st.selectbox('Has credit card', [0, 1])
is_active_member = st.selectbox('Active Member', [0, 1])
estimated_salary = st.number_input('Estimated Salary')
credit_score = st.number_input('Credit Score', min_value=300, max_value=850)  # Added CreditScore input

# Create input DataFrame with feature names matching the training data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],  # Added CreditScore
    'Gender': [label_encode.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [no_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode Geography
geo_encoded = one_hot_encode.transform([[geography]]).toarray()
geo_df = pd.DataFrame(geo_encoded, columns=one_hot_encode.get_feature_names_out(['Geography']))

# Concatenate input data with one-hot encoded Geography
input_data = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)

# Reorder columns to match the scaler's expected order
input_data = input_data[scaler.feature_names_in_]

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make prediction
pred = model.predict(input_data_scaled)  # Use scaled data for prediction
pred_probability = pred[0][0]

# Display results
st.write(f'Churn probability: {pred_probability:.2f}')
if pred_probability > 0.5:
    st.write('Customer likely to churn')
else:
    st.write('Not likely to churn')