from flask import Flask, request, jsonify
import joblib
import os

application = Flask(__name__)

# Load trained model (FIXED)
model_path = os.path.join(os.getcwd(), "sentiment_model.joblib")
model = joblib.load(model_path)

@application.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({
            "error": "No text provided. Please send a JSON body with a 'text' key."
        }), 400

    prediction = model.predict([text])[0]

    return jsonify({
        "input_text": text,
        "sentiment_prediction": prediction
    })

if __name__ == "__main__":
    application.run(host="0.0.0.0", port=5000)