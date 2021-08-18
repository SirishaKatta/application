import numpy as np
import pickle
import pandas as pd
import streamlit as st
df=pd.read_csv('diabetes.csv')
x=df.iloc[:,0:8].values
y=df.iloc[:,8].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
from sklearn.linear_model import LogisticRegression
Log = LogisticRegression()
Log.fit(x_train,y_train)



@st.cache()


def prediction(std_data):
    prediction = Log.predict(std_data)

    if (prediction[0] == 0):
        pred =  'Non Diabetic'
    else:
        pred = 'Diabetic'
    return pred

def main():
	
  st.title("Diabetics Predictor")
  html_temp = """
  <body bgcolor ="#FFFFFF">
  <div style="background-color:#54c79b;padding:10px">
  <h2 style="color:#0b124d;text-align:center;"><b>Streamlit Diabetes Predictor App </b></h2>
  </div>
  """
  st.markdown(html_temp, unsafe_allow_html=True)
  st.write("a logo and text next to eachother")
  col1, mid, col2 = st.beta_columns([1,1,20])
  with col1:
    st.image('emoji image.html', width=60)
  with col2:
    st.write('A Name')

  Pregnancies= st.number_input("Pregnancies")
  Glucose=st.number_input("Glucose")
  BloodPressure=st.number_input("BloodPressure")
  SkinThickness=st.number_input("SkinThickness")
  Insulin=st.number_input("Insulin")
  BMI=st.number_input("BMI")
  DiabetesPedigreeFunction=st.number_input("DiabetesPedigreeFunction")
  Age=st.number_input("Age")
  result = ""
	
  if st.button("Predict"):
      input_data=(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
      input_data_as_numpy_array = np.asarray(input_data)
      input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
      std_data = sc.transform(input_data_reshaped)
      result=prediction(std_data)
      st.success('The person is {}'.format(result))

if __name__ == '__main__':
   main()
