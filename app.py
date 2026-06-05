import streamlit as st
import pandas as pd
import joblib

model = joblib.load('logistic_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

st.title('Heart Disease Prediction')
st.markdown('Provide the following information to predict the likelihood of heart disease:')

age = st.slider("Age",18,100,40)
sex = st.selectbox("Sex",["Male","Female"])

chest_pain = st.selectbox("Chest Pain Type",["ATA","NAP","TA","ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)",80,200,120)
cholesterol = st.number_input("Serum Cholesterol (mg/dl)",100,600,200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl",[0,1])
rest_ecg = st.selectbox("Resting ECG",["Normal","ST","LVH"])
max_hr = st.slider("Maximum Heart Rate Achieved",60,220,150)
exercise_angina = st.selectbox("Exercise Induced Angina",["Y","N"])
oldpeak = st.slider("Oldpeak (ST depression)",0.0,6.0,1.0)
st_slope = st.selectbox("ST Slope",["Up","Flat","Down"])

if st.button("Predict"):
  raw_input = {
    "age": age,
    "resting_bp": resting_bp,
    "cholesterol": cholesterol,
    "fasting_bs": fasting_bs,
    "max_hr": max_hr,
    "oldpeak": oldpeak,
    "sex_"+sex: 1,
    "chect_pain_type_"+chest_pain: 1,
    "restingECG_"+rest_ecg: 1,
    "exercise_angina_"+exercise_angina: 1,
    "st_slope_"+st_slope: 1
  }
  input_df = pd.DataFrame([raw_input])
  for col in expected_columns:
    if col not in input_df.columns:
      input_df[col] = 0

  input_df = input_df[expected_columns]
  scaled_input = scaler.transform(input_df)
  prediction = model.predict(scaled_input)[0]
  if prediction == 1:
    st.error("High likelihood of heart disease. Please consult a doctor.")
  else:
    st.success("Low likelihood of heart disease. Keep up the healthy lifestyle!")