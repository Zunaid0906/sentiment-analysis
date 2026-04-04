import streamlit as st
import pickle
import os

# Load model and vectorizer
model_path = os.path.join("models", "sentiment_model.pkl")
vectorizer_path = os.path.join("models", "vectorizer.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# Page config
st.set_page_config(page_title="Sentiment Analyzer", page_icon="😎")

# Title
st.title("Sentiment Analyzer 😎")

# Input
text = st.text_input("Enter text:")

# Predict button
if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        lower_text = text.lower()

        # 🔥 RULE-BASED BOOST (fix real-life sentences)
        if any(word in lower_text for word in ["love", "like you", "crush", "miss you"]):
            prediction = "Positive"
        elif any(word in lower_text for word in ["hate", "worst", "angry"]):
            prediction = "Negative"
        else:
            # ML Prediction
            vec = vectorizer.transform([text])
            pred = model.predict(vec)[0]
            prediction = "Positive" if pred == 1 else "Negative"

        # Output styling
        if prediction == "Positive":
            st.success(f"{prediction} 😄")
        elif prediction == "Negative":
            st.error(f"{prediction} 😡")
        else:
            st.info(f"{prediction} 😐")