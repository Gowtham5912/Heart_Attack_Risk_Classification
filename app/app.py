import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load('../model/heart_model.pkl')

# Load Scaler
scaler = joblib.load('../model/scaler.pkl')

# Title
st.title("Heart Attack Risk Prediction System")

# User Inputs

age = st.number_input("Age")

sex = st.selectbox(
    
    "Sex",
    
    ["Male", "Female"]
)

chestpain = st.selectbox(
    
    "Chest Pain Type",
    
    ["ATA", "NAP", "ASY", "TA"]
)

restingbp = st.number_input("Resting Blood Pressure")

cholesterol = st.number_input("Cholesterol")

fastingbs = st.selectbox(
    
    "Fasting Blood Sugar",
    
    [0,1]
)

restingecg = st.selectbox(
    
    "Resting ECG",
    
    ["Normal", "ST", "LVH"]
)

maxhr = st.number_input("Maximum Heart Rate")

exerciseangina = st.selectbox(
    
    "Exercise Angina",
    
    ["Y","N"]
)

oldpeak = st.number_input("Oldpeak")

stslope = st.selectbox(
    
    "ST Slope",
    
    ["Up","Flat","Down"]
)

# Encoding Inputs

sex = 1 if sex == "Male" else 0

exerciseangina = 1 if exerciseangina == "Y" else 0

cp_dict = {
    
    "ATA":0,
    "NAP":1,
    "ASY":2,
    "TA":3
}

restecg_dict = {
    
    "LVH":0,
    "Normal":1,
    "ST":2
}

slope_dict = {
    
    "Down":0,
    "Flat":1,
    "Up":2
}

chestpain = cp_dict[chestpain]

restingecg = restecg_dict[restingecg]

stslope = slope_dict[stslope]

# Create DataFrame

input_data = pd.DataFrame([[

    age,
    sex,
    chestpain,
    restingbp,
    cholesterol,
    fastingbs,
    restingecg,
    maxhr,
    exerciseangina,
    oldpeak,
    stslope

]])

# Scale Data

input_data = scaler.transform(input_data)

# Prediction

if st.button("Predict"):
    
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        
        st.error("High Risk of Heart Disease")
        
    else:
        
        st.success("Low Risk of Heart Disease")