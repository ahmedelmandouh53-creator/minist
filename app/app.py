"""Web app: draw a digit, pick one of the three saved models, get a prediction.

Run from the project root:   python app/app.py      then open http://127.0.0.1:5000
"""
import csv
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from inference import MODEL_LABELS, DigitRecognizer, canvas_to_mnist, data_url_to_image

CHARTS_DIR = Path(__file__).resolve().parent / "charts"

app = Flask(__name__)
recognizer = DigitRecognizer()
print("Loaded models:", ", ".join(recognizer.models))


@app.get("/")
def index():
    return render_template("index.html", models=MODEL_LABELS)


CHART_TITLES = {
    "comparison.png": "Accuracy, training time and parameters",
    "cmp_performance.png": "Performance",
    "cmp_cost.png": "Cost: time, speed and size",
    "cmp_tradeoff.png": "Accuracy vs cost",
    "cmp_learning_curves.png": "Learning curves",
    "cmp_confusion.png": "Confusion matrices",
    "cmp_per_class_f1.png": "F1 score per digit",
    "cmp_search_overview.png": "Hyperparameter search",
}


def count_text(n):
    return f"{n / 1e6:.2f} M" if n >= 1e6 else f"{n / 1e3:.0f} k"


# column in model_comparison.csv -> (table heading, how to print it, which value is best)
TABLE_COLUMNS = [
    ("test acc", "Test accuracy", lambda v: f"{v:.2%}", max),
    ("gap (train - test)", "Train-test gap", lambda v: f"{v:.2%}", min),
    ("train time (s)", "Training time", lambda v: f"{v:.1f} s", min),
    ("parameters", "Parameters", count_text, min),
]


def comparison_table():
    """Summary table from app/charts/model_comparison.csv (written by notebook 04); None if it is missing."""
    path = CHARTS_DIR / "model_comparison.csv"
    if not path.exists():
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    name_col = list(rows[0])[0]  # the first column holds the model name
    best = {col: pick(float(r[col]) for r in rows) for col, _, _, pick in TABLE_COLUMNS}
    return {
        "headers": [heading for _, heading, _, _ in TABLE_COLUMNS],
        "rows": [(r[name_col], [(show(float(r[col])), float(r[col]) == best[col]) for col, _, show, _ in TABLE_COLUMNS])
                 for r in rows],
    }


def chart_names():
    """Every comparison chart in app/charts/."""
    return sorted(p.name for p in CHARTS_DIR.glob("*.png"))


@app.get("/comparison")
def comparison():
    charts = [(n, CHART_TITLES.get(n, n.removesuffix(".png").replace("_", " ").title())) for n in chart_names()]
    # cmp_*.png come from the earlier training run; everything else is made by notebook 04
    return render_template("comparison.html", table=comparison_table(),
                           latest=[c for c in charts if not c[0].startswith("cmp_")],
                           earlier=[c for c in charts if c[0].startswith("cmp_")])


@app.get("/charts/<name>")
def chart(name):
    if name not in chart_names():
        abort(404)
    return send_from_directory(CHARTS_DIR, name)


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
