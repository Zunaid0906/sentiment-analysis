import pandas as pd
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# Load dataset
df = pd.read_csv('../data/dataset.csv')

# IMPORTANT: dataset columns
df.columns = ['id', 'entity', 'sentiment', 'tweet']

# Remove neutral (optional but better)
df = df[df['sentiment'] != 'Neutral']

# Convert labels
df['label'] = df['sentiment'].apply(lambda x: 1 if x == 'Positive' else 0)

# Clean text
df['tweet'] = df['tweet'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['tweet'], df['label'], test_size=0.2, random_state=42
)

# Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)

# Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Save model
with open('../models/sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save vectorizer
with open('../models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("DONE ✅ Model trained")