import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

BASE_PATH = "/content/drive/MyDrive/annclassification"

model = load_model(f"{BASE_PATH}/model.h5")

with open(f"{BASE_PATH}/label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open(f"{BASE_PATH}/one_hot_encoder_geo.pkl", "rb") as file:
    one_hot_encoder_geo = pickle.load(file)

with open(f"{BASE_PATH}/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

st.title("Customer Churn Prediction")

credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)

geography = st.selectbox(
    "Geography",
    one_hot_encoder_geo.categories_[0]
)

gender = st.selectbox(
    "Gender",
    label_encoder_gender.classes_
)

age = st.slider("Age", 18, 92)

tenure = st.slider("Tenure", 0, 10)

balance = st.number_input("Balance", value=0.0)

num_of_products = st.slider("Number of Products", 1, 4)

has_cr_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

is_active_member = st.selectbox(
    "Is Active Member",
    [0, 1]
)

estimated_salary = st.number_input(
    "Estimated Salary"
)

gender_encoded = label_encoder_gender.transform([gender])[0]

geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=one_hot_encoder_geo.get_feature_names_out(["Geography"])
)

input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [gender_encoded],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

input_data = pd.concat(
    [input_data.reset_index(drop=True),
     geo_encoded_df],
    axis=1
)

input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)

prediction_probability = prediction[0][0]

st.write("Churn Probability:", round(prediction_probability, 4))

if prediction_probability > 0.5:
    st.error("Customer is likely to churn")
else:
    st.success("Customer is not likely to churn")

