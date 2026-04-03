import streamlit as st
import pickle
import os

# ---------- LOAD MODEL SAFELY ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "models", "sentiment_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# ---------- UI ----------
st.set_page_config(page_title="Sentiment Analyzer", page_icon="😎")

st.title("Sentiment Analyzer 😎")
st.write("Type something and check sentiment")

# Input box
user_input = st.text_input("Enter text:")

# Button
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        # Transform text
        transformed = vectorizer.transform([user_input])

        # Predict
        prediction = model.predict(transformed)[0]

        # Output
        if prediction == 1:
            st.success("Positive 😊")
        else:
            st.error("Negative 😡")