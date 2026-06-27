"""
app.py
------
Flask backend for Handwritten Character Recognition.
Loads a trained Keras model and exposes:
  GET  /          → serves the web UI
  POST /predict   → returns digit prediction + confidence
  GET  /health    → liveness probe for Render
"""

import os
import io
import base64
import logging

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf

# ─── Setup ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)   # Allow cross-origin requests (needed for canvas → API calls)

# Limit TensorFlow threads to stay within Render free-tier RAM
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# ─── Load model once at startup ───────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.h5")

logger.info("Loading model from %s …", MODEL_PATH)
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully!")
except Exception as exc:
    logger.error("Failed to load model: %s", exc)
    model = None

# Label map – extend this for EMNIST letters (A-Z)
DIGIT_LABELS = [str(i) for i in range(10)]


# ─── Image preprocessing ──────────────────────────────────────────────────────
def preprocess_image(image_data: str) -> np.ndarray:
    """
    Convert a base64-encoded PNG (from HTML5 canvas) into a
    (1, 28, 28, 1) float32 numpy array ready for model.predict().
    """
    # Strip the data-URL prefix if present
    if "base64," in image_data:
        image_data = image_data.split("base64,")[1]

    # Decode bytes
    img_bytes = base64.b64decode(image_data)

    # Open as RGBA, composite on black bg, convert to grayscale
    img_rgba = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    background = Image.new("RGBA", img_rgba.size, (0, 0, 0, 255))
    background.paste(img_rgba, mask=img_rgba.split()[3])   # alpha-composite
    img_gray = background.convert("L")

    # Resize to 28×28 (MNIST format)
    img_gray = img_gray.resize((28, 28), Image.LANCZOS)

    arr = np.array(img_gray, dtype="float32")

    # If the canvas background is dark (drawn digit is white), keep as-is.
    # If background is light (white paper look), invert so digit pixels are bright.
    if arr.mean() > 127:
        arr = 255.0 - arr

    # Normalise to [0, 1]
    arr /= 255.0

    # Reshape to (1, 28, 28, 1)
    return arr.reshape(1, 28, 28, 1)


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the main web UI."""
    return render_template("index.html")


@app.route("/health")
def health():
    """Liveness probe – Render pings this to confirm the service is up."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON body: { "image": "<base64 data-URL>" }
    Returns:
        {
          "prediction": "7",
          "confidence": 98.42,
          "top3": [
            { "label": "7", "confidence": 98.42 },
            { "label": "1", "confidence":  1.03 },
            { "label": "9", "confidence":  0.55 }
          ]
        }
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 503

    data = request.get_json(force=True)
    if not data or "image" not in data:
        return jsonify({"error": "Missing 'image' field in request body."}), 400

    try:
        img_array = preprocess_image(data["image"])
    except Exception as exc:
        logger.exception("Image preprocessing failed")
        return jsonify({"error": f"Image preprocessing failed: {exc}"}), 422

    try:
        preds = model.predict(img_array, verbose=0)[0]   # shape (10,)
    except Exception as exc:
        logger.exception("Model inference failed")
        return jsonify({"error": f"Model inference failed: {exc}"}), 500

    predicted_idx  = int(np.argmax(preds))
    confidence_pct = round(float(preds[predicted_idx]) * 100, 2)

    # Top-3 predictions
    top3_idx = np.argsort(preds)[::-1][:3]
    top3 = [
        {"label": DIGIT_LABELS[i], "confidence": round(float(preds[i]) * 100, 2)}
        for i in top3_idx
    ]

    return jsonify({
        "prediction": DIGIT_LABELS[predicted_idx],
        "confidence": confidence_pct,
        "top3": top3
    })


# ─── Entry point (local dev only) ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
