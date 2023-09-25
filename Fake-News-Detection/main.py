import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv("https://raw.githubusercontent.com/satheesh22g/Datasets/satheesh22g/news.csv")

st.header("Fake News Detection app")
st.text_input("Enter your Name: ", key="name")

tfid = TfidfVectorizer()
with open('Fake-News-Detection/tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

clf = PassiveAggressiveClassifier()
with open('Fake-News-Detection/pac_model.pkl', 'rb') as f:
    pac = pickle.load(f)

if st.checkbox('Show Training Dataframe'):
    data
text = st.text_input("Enter the text")

if st.checkbox('Show Training Dataframe'):
    data
text = st.text_input("Enter the text",key="text")

if st.button('Make Prediction'):
    input_text = tfidf.transform([text])
    prediction = pac.predict(input_text)
    s=str(np.squeeze(prediction, -1))
    st.write("It is look like fake:",s)

    st.write(f"Thank you {st.session_state.name}! I hope you liked it.")
