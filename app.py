import joblib
import os
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "models", "sentiment_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

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