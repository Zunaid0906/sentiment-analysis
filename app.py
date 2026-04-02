import streamlit as st
import pickle

# Load model
with open('models/sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# UI
st.title("Sentiment Analyzer 😎")
st.write("Type something and check sentiment")

text = st.text_input("Enter text:")

if st.button("Analyze"):
    if text.strip() != "":
        text_vec = vectorizer.transform([text])
        prediction = model.predict(text_vec)[0]

        if prediction == 1:
            st.success("Positive 😊")
        else:
            st.error("Negative 😡")
    else:
        st.warning("Please enter text")