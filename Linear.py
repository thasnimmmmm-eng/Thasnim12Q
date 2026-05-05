import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df=pd.read_csv("student_scores.csv")
X=df.iloc[:, :-1].values
Y=df.iloc[:,-1].values
X_train,X_test,Y_test=train_test_split(X, Y,test_size=0.2,random=42)
model= LinearRegression()
model.fit(X_train,Y_train)
st.title("exam score prediction model")
st.write("enter the no. of hours you studied for the exam")
hours = st.number_input("hours studied", min_value=0.0,step=0.1)
if st.button("Predict Score "):
  predicted_score=model.predict([[hours]])[0]
  st.success(f"Predicted score: {predicted_score:.2f}")
st.write(" sample training DATA")
st.dataframe(df)
