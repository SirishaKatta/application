import numpy as np
import pickle
import pandas as pd
import streamlit as st

pickle_in = open('model_pickle.pkl', 'rb')
classifier=pickle.load(pickle_in)

@st.cache()

#def prediction(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
def prediction(std_data):
    prediction = classifier.predict(std_data)

    if (prediction[0] == 0):
        pred =  'Non Diabetic'
    else:
        pred = 'Diabetic'
    return pred

def main():
	
  st.title("Diabetics Predictor")
  html_temp = """
  <div style="background-color:#546beb;padding:10px">
  <h2 style="color:#93f50a;text-align:center;">Streamlit Diabetics Predictor </h2>
  </div>
  """
  st.markdown(html_temp, unsafe_allow_html=True)

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
      std_data = Log.transform(input_data_shaped)
      result=prediction(std_data)
      st.success('The person is {}'.format(result))

if __name__ == '__main__':
   main()
