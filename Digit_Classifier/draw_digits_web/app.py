#!/usr/bin/env python3
import io
import base64
from typing import Dict

import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, render_template

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# ---- Train two simple models on sklearn 8x8 digits ----
digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

logreg = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("clf", LogisticRegression(max_iter=2000))
])
logreg.fit(X_train, y_train)

rf = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
])
rf.fit(X_train, y_train)

MODELS: Dict[str, Pipeline] = {
    "LogisticRegression(8x8)": logreg,
    "RandomForest(8x8)": rf,
}

def pil_to_8x8_vector(img: Image.Image) -> np.ndarray:
    # Ensure grayscale
    g = img.convert("L")
    # Resize to 8x8
    g8 = g.resize((8, 8), Image.LANCZOS)
    arr = np.asarray(g8).astype(np.float32)
    # Our canvas is white-on-black; sklearn digits are 0..16 intensity with dark digit on light background.
    arr = 255.0 - arr  # invert
    arr = (arr / 255.0) * 16.0
    return arr.reshape(1, -1)

@app.route("/")
def index():
    return render_template("index.html", model_names=list(MODELS.keys()))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    b64 = data.get("image", "")
    model_name = data.get("model", "")
    if model_name not in MODELS:
        return jsonify({"error": f"Unknown model '{model_name}'"}), 400

    # Expect "data:image/png;base64,...."
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw))
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    X = pil_to_8x8_vector(img)
    clf = MODELS[model_name]
    # Predict proba if available
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)[0]
    else:
        # fallback to decision_function or one-hot-ish
        if hasattr(clf, "decision_function"):
            scores = clf.decision_function(X)[0]
        else:
            pred = int(clf.predict(X)[0])
            scores = np.full(10, -1e9, dtype=np.float32)
            scores[pred] = 0.0
        e = np.exp(scores - np.max(scores))
        probs = e / e.sum()
    pred_digit = int(np.argmax(probs))

    return jsonify({
        "prediction": pred_digit,
        "probs": [float(p) for p in probs],
        "model": model_name
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
