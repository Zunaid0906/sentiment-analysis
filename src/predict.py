import pickle

with open('../models/sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('../models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

text = input("Enter a sentence: ")

text_vec = vectorizer.transform([text])

prediction = model.predict(text_vec)[0]

if prediction == 1:
    print("Positive 😊")
else:
    print("Negative 😡")