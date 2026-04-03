import streamlit as st
import pickle
import os

# Get base directory safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build correct paths
model_path = os.path.join(BASE_DIR, "models", "sentiment_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

# Load files
with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# UI
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