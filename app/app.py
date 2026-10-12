"""Web app: draw a digit, pick one of the three saved models, get a prediction.

Run from the project root:   python app/app.py      then open http://127.0.0.1:5000
"""
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from inference import MODEL_LABELS, DigitRecognizer, canvas_to_mnist, data_url_to_image

app = Flask(__name__)
recognizer = DigitRecognizer()
print("Loaded models:", ", ".join(recognizer.models))


@app.get("/")
def index():
    return render_template("index.html", models=MODEL_LABELS)


@app.post("/api/predict")
def predict():
    body = request.get_json(silent=True) or {}
    key = body.get("model")
    if key not in recognizer.models:
        return jsonify(error=f"Unknown model '{key}'."), 400
    try:
        image28 = canvas_to_mnist(data_url_to_image(body["image"]))
    except Exception as exc:
        return jsonify(error=f"Could not read the drawing: {exc}"), 400
    if image28 is None:
        return jsonify(error="The canvas is empty - draw a digit first."), 400
    digit, probs = recognizer.predict(key, image28)
    return jsonify(model=key, model_label=MODEL_LABELS[key], digit=digit,
                   probabilities=[round(float(p), 4) for p in probs])


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
