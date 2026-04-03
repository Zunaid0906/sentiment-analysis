import streamlit as st
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "models", "sentiment_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "models", "vectorizer.pkl"))

st.title("Sentiment Analyzer 😎")

text = st.text_input("Enter text:")

if st.button("Analyze"):
    if text:
        vec = vectorizer.transform([text])
        result = model.predict(vec)[0]

        if result == 1:
            st.success("Positive 😊")
        else:
            st.error("Negative 😠")