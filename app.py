import os
import joblib
from flask import Flask, request, jsonify

application = Flask(__name__)

print("Starting app...")

model_path = os.path.join(os.getcwd(), "sentiment_model.joblib")
print("Loading model from:", model_path)

model = joblib.load(model_path)

print("Model loaded successfully")

@application.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    prediction = model.predict([text])[0]

    return jsonify({
        "input_text": text,
        "sentiment_prediction": prediction
    })

if __name__ == "__main__":
    application.run(host="0.0.0.0", port=5000)