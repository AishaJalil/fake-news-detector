# model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Load datasets
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

# Assign labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
df = pd.concat([fake, true])
df["content"] = df["title"] + " " + df["text"]
df = df[["content", "label"]].dropna()

print(df['label'].value_counts())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["content"], df["label"], test_size=0.2, random_state=42
)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
#model = LogisticRegression()
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vec, y_train)

# Create 'model' folder if it doesn't exist
if not os.path.exists("model"):
    os.makedirs("model")

# Save model and vectorizer
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("✅ Model and vectorizer saved in /model folder.")
