import streamlit as st

st.title("Sentiment Analyzer 😎")

text = st.text_input("Enter text:")

# Simple keyword-based logic
positive_words = ["good", "great", "love", "amazing", "happy", "excellent"]
negative_words = ["bad", "hate", "terrible", "sad", "worst", "angry"]

if st.button("Analyze"):
    if text:
        text_lower = text.lower()

        if any(word in text_lower for word in positive_words):
            st.success("Positive 😊")

        elif any(word in text_lower for word in negative_words):
            st.error("Negative 😠")

        else:
            st.info("Neutral 😐")