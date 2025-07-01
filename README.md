# 📰 Fake News Detector

This is a machine learning web app that detects whether a news article is real or fake using NLP.

### 🔧 Tech Stack
- Python, Pandas, Scikit-learn
- Gradio (for UI)
- Trained on Kaggle's Fake/Real News Dataset

### 🚀 Live App
👉 Try it on [Hugging Face Spaces] https://huggingface.co/spaces/Aisha-Jalil-990/fake-news-detector

### Sample Input

# Real Input:
The United Nations has passed a new resolution aimed at increasing humanitarian aid to Gaza, following weeks of conflict in the region. The resolution, which received broad support in the General Assembly, calls for an immediate ceasefire and unrestricted access for aid organizations. UN Secretary-General António Guterres emphasized the need for international cooperation to prevent further escalation.

# Fake Input
BREAKING: NASA Confirms Earth Will Experience 15 Days of Darkness in November!
According to NASA, the world is going to experience 15 days of complete darkness starting November 15. The rare astronomical event, known as “November Blackout,” will occur due to a cosmic alignment causing massive solar interference.



### 🧠 How It Works
- Preprocesses text using TF-IDF
- Classifies using Random Forest
- UI built with Gradio
