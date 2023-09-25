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
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

clf = PassiveAggressiveClassifier()
with open('pac_model.pkl', 'rb') as f:
    pac = pickle.load(f)

if st.checkbox('Show Training Dataframe'):
    data
text = st.text_input("Enter the text")

if st.button('Make Prediction'):
    input_text = tfidf.transform(text)
    prediction = pac.predict(input_text)
    print("final pred", np.squeeze(prediction, -1))
    st.write(f"Your fish weight is: {np.squeeze(prediction, -1):.2f}g")

    st.write(f"Thank you {st.session_state.name}! I hope you liked it.")