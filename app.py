"""
Created on Sat Nov 23 07:22:30 2024

@author: Shambhavi
"""

import pickle 
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu

# Load models safely
try:
    diabetes_model = pickle.load(open('saved_models/diabetes_model.sav', 'rb'))
    heart_disease_model = pickle.load(open('saved_models/heart_disease_model.sav', 'rb'))
except Exception as e:
    st.error(f"Error loading models: {e}")

# Sidebar menu
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction', 'Heart Disease Prediction'],
                           icons=['activity', 'heart'],
                           default_index=0)

# Diabetes Prediction
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('Number of pregnancies', min_value=0, step=1)
    
    with col2:
        Glucose = st.number_input('Glucose Level', min_value=0.0)
    
    with col3:
        BloodPressure = st.number_input('Blood Pressure', min_value=0.0)
        
    with col1:
        SkinThickness = st.number_input('Skin Thickness', min_value=0.0)
    
    with col2:
        Insulin = st.number_input('Insulin Level', min_value=0.0)
       
    with col3:
        BMI = st.number_input('BMI Value', min_value=0.0)
    
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', min_value=0.0)
    Age = st.slider('Age of the person', 0, 130)

    diab_diagnosis = ''

    if st.button('Diabetes Test Result'):
        try:
            diab_prediction = diabetes_model.predict(np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]))

            diab_diagnosis = 'The person is Diabetic' if diab_prediction[0] == 1 else 'The person is NOT Diabetic'
        except Exception as e:
            diab_diagnosis = f"Error in prediction: {e}"

    st.success(diab_diagnosis)

# Heart Disease Prediction
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age of the person', min_value=0, step=1)
    
    with col2:
        sex = st.selectbox('Sex of the person', options=[0, 1])  # 0 = Female, 1 = Male
    
    with col3:
        cp = st.number_input('Chest Pain type', min_value=0, max_value=3, step=1)
        
    with col1:
        trestbps = st.number_input('Resting Blood Pressure', min_value=0.0)
    
    with col2:
        chol = st.number_input('Cholesterol Level', min_value=0.0)
       
    with col3:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])  # 0 = No, 1 = Yes
        
    with col1:
        restecg = st.number_input('Resting ECG', min_value=0, max_value=2, step=1)
    
    with col2:
        thalach = st.number_input('Max Heart Rate Achieved', min_value=0.0)
       
    with col3:
        exang = st.selectbox('Exercise induced angina', options=[0, 1])  # 0 = No, 1 = Yes
        
    oldpeak = st.number_input('ST Depression', min_value=0.0)

    heart_diagnosis = ''

    if st.button('Heart Disease Test Result'):
        try:
            heart_prediction = heart_disease_model.predict(np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]]))

            heart_diagnosis = 'The person has a Heart Disease' if heart_prediction[0] == 1 else 'The person does not have a Heart Disease'
        except Exception as e:
            heart_diagnosis = f"Error in prediction: {e}"

    st.success(heart_diagnosis)
