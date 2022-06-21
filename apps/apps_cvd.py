# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 14:45:43 2022

@author: Afiq Sabqi
"""

import streamlit as st
import base64
import sklearn
import os
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
#Load the saved model


MODEL_PATH=os.path.join(os.getcwd(),'best_model.pkl')
with open(MODEL_PATH,'rb') as file:
    model=pickle.load(file)

PIPELINE_PATH=os.path.join(os.getcwd(),'best_pipeline.pkl')
with open(PIPELINE_PATH,'rb') as file:
    pipeline=pickle.load(file)

SS_FILE_NAME=os.path.join(os.getcwd(),'standard_scaler.pkl')
with open(SS_FILE_NAME,'rb') as file:
    ss=pickle.load(file)


st.set_page_config(page_title="Healthy Heart App",page_icon="⚕️",
                   layout="centered",initial_sidebar_state="expanded")

st.title('Heart Disease Diagnosis Assistant')
st.markdown('This application is meant to **_assist_ _doctors_ _in_ diagnosing**,\
            if a patient has a **_Heart_ _Disease_ _or_ not**\
            using few details about their health')

st.markdown('Please **Enter _the_ _below_ details** to know the results -')


def preprocess(age,sex,cp,trtbps,restecg,chol,fbs,thalachh,
               exng,oldpeak,slp,caa,thall ):   
 
    
    if sex=="female":
        sex=0
    elif sex=="male":
        sex=1
    
    if cp=="Typical angina":
        cp=0
    elif cp=="Atypical angina":
        cp=1
    elif cp=="Non-anginal pain":
        cp=2
    elif cp=="Asymptomatic":
        cp=3
    
    if exng=="Yes":
        exng=1
    elif exng=="No":
        exng=0
 
    if fbs=="Yes":
        fbs=1
    elif fbs=="No":
        fbs=0
 
    if slp=="Upsloping: better heart rate with excercise(uncommon)":
        slp=0
    elif slp=="Flatsloping: minimal change(typical healthy heart)":
        slp=1
    elif slp=="Downsloping: signs of unhealthy heart":
        slp=2  
 
    if thall=="fixed defect: used to be defect but ok now":
        thall=1
    elif thall=="normal":
        thall=2
    elif thall=="reversable defect: no proper blood movement when excercising":
        thall=3

    if restecg=="Nothing to note":
        restecg=0
    elif restecg=="ST-T Wave abnormality":
        restecg=1
    elif restecg=="Possible or definite left ventricular hypertrophy":
        restecg=2


    user_input=[age,sex,cp,trtbps,restecg,chol,fbs,thalachh,
                exng,oldpeak,slp,caa,thall]
    user_input=np.array(user_input)
    user_input=user_input.reshape(1,-1)
    user_input=scaler.fit_transform(user_input)
    prediction = model.predict(user_input)

    return prediction

     
      

# age=st.selectbox("Age",range(1,121,1))
age=st.number_input('Age')
sex = st.radio("Select Gender: ", ('male', 'female'))
cp = st.selectbox('Chest Pain Type',("Typical angina","Atypical angina",
                                     "Non-anginal pain","Asymptomatic")) 
trtbps=st.selectbox('Resting Blood Sugar',range(1,500,1))
restecg=st.selectbox('Resting Electrocardiographic Results',
                     ("Nothing to note","ST-T Wave abnormality",
                      "Possible or definite left ventricular hypertrophy"))
chol=st.selectbox('Cholestoral in mg/dl',range(1,1000,1))
fbs=st.radio("Fasting Blood Sugar higher than 120 mg/dl", ['Yes','No'])
thalachh=st.selectbox('Maximum Heart Rate Achieved',range(1,300,1))
exng=st.selectbox('Exercise Induced Angina',["Yes","No"])
oldpeak=st.number_input('Oldpeak')
slp = st.selectbox('Heart Rate Slope',
                   ("Upsloping: better heart rate with excercise(uncommon)",
                    "Flatsloping: minimal change(typical healthy heart)",
                    "Downsloping: signs of unhealthy heart"))
caa=st.selectbox('Number of Major Vessels Colored by Flourosopy',range(0,6,1))
thall=st.selectbox('Thallesemia: 1-defect,2-normal,3-no proper blood movement',
                   range(1,4,1))


model=preprocess(age,sex,cp,trtbps,restecg,chol,fbs,thalachh,
                exng,oldpeak,slp,caa,thall)


if st.button("Predict"):    
  if model==0:
    st.error('Warning! You have high risk of getting a CVD!')
  else:
    st.success('You have lower risk of getting a CVD!')



