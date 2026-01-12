from flask import Flask, render_template, request
import pickle
import re
import os

app = Flask(__name__)

# Load model and vectorizer once at start
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text cleaning function (same as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    summary_text = None
    confidence = None
    confidence_value = None

    if request.method == "POST":
        news_text = request.form["news"]
        cleaned_text = clean_text(news_text)
        text_vector = vectorizer.transform([cleaned_text])
        
        prediction = model.predict(text_vector)[0]
        prob = model.predict_proba(text_vector)[0]

        if prediction == 1:
            prediction_text = "REAL NEWS"
            confidence_value = prob[1] * 100  # float number for width
            confidence = f"{confidence_value:.2f}%"
        else:
            prediction_text = "FAKE NEWS"
            confidence_value = prob[0] * 100
            confidence = f"{confidence_value:.2f}%"

        # For now, no summary generation code here (or add your own)
        summary_text = "Summary feature coming soon."  # placeholder

        result = prediction_text

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        confidence_value=confidence_value,
        summary=summary_text
    )
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
