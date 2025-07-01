# app.py

import gradio as gr
import pickle

# Load saved model and vectorizer
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# Define prediction function
def predict_fake_news(text):
    vectorized = vectorizer.transform([text])
    prediction = model.predict(vectorized)[0]
    return "🟢 Real News" if prediction == 1 else "🔴 Fake News"

# Gradio interface
interface = gr.Interface(
    fn=predict_fake_news,
    inputs=gr.Textbox(lines=10, placeholder="Paste news article here..."),
    outputs="text",
    title="📰 Fake News Detector",
    description="Enter a news article and let the AI decide if it's real or fake."
)

# Launch app
interface.launch()
