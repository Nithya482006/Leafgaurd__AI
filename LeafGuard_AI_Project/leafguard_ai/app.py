import os
import io
import json
import sqlite3
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
MODEL_PATH = BASE_DIR / "models" / "plant_disease_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"
DB_PATH = BASE_DIR / "plantguard.db"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
IMAGE_SIZE = (224, 224)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-development-secret")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "models").mkdir(exist_ok=True)

_model = None
_class_names = None
_model_error = None

# These are general educational suggestions, not a substitute for local agricultural advice.
TREATMENTS = {
    "healthy": {
        "title": "Healthy leaf",
        "summary": "The model did not detect one of the supported disease classes with high confidence.",
        "actions": [
            "Continue regular field monitoring and keep records of symptoms.",
            "Water at the base of plants where practical and avoid prolonged leaf wetness.",
            "Maintain balanced nutrition and follow local crop-management guidance."
        ],
        "urgency": "Routine monitoring"
    },
    "default": {
        "title": "General next steps",
        "summary": "Use the prediction as a screening aid. Confirm the diagnosis before applying treatment.",
        "actions": [
            "Photograph several affected and unaffected leaves in natural, even light.",
            "Separate visibly affected plants where practical and avoid handling healthy plants immediately afterward.",
            "Remove heavily affected plant material only when recommended for that crop and disease.",
            "Ask a local agricultural extension officer or qualified agronomist to confirm the diagnosis.",
            "Use only locally approved products and follow their label, crop, dosage, and safety instructions."
        ],
        "urgency": "Confirm before treatment"
    },
    "tomato___early_blight": {
        "title": "Tomato — Early blight",
        "summary": "Early blight often causes brown target-like spots, frequently on older leaves.",
        "actions": [
            "Avoid overhead irrigation and allow foliage to dry.",
            "Remove badly affected lower leaves when appropriate; do not compost diseased material unless your local guidance permits it.",
            "Rotate crops and clear crop debris after harvest.",
            "Confirm the diagnosis locally. If a fungicide is warranted, use a locally registered product strictly according to its label."
        ],
        "urgency": "Monitor and confirm"
    },
    "tomato___late_blight": {
        "title": "Tomato — Late blight",
        "summary": "Late blight can spread rapidly in cool, wet conditions and may damage leaves, stems, and fruit.",
        "actions": [
            "Seek prompt confirmation from an agricultural extension officer, especially if symptoms are spreading quickly.",
            "Avoid moving wet plant material through the field and reduce leaf wetness where possible.",
            "Follow local disease alerts and crop-specific guidance.",
            "Use only locally approved control products as directed by the label."
        ],
        "urgency": "Prompt confirmation"
    },
    "potato___late_blight": {
        "title": "Potato — Late blight",
        "summary": "Late blight can spread quickly and affect both foliage and tubers.",
        "actions": [
            "Contact a local agricultural extension service promptly if late blight is suspected.",
            "Avoid moving potentially infected plants or equipment between fields without cleaning.",
            "Follow local disease advisories and crop-specific recommendations.",
            "Use only locally registered products and follow label instructions."
        ],
        "urgency": "Prompt confirmation"
    },
    "corn___common_rust": {
        "title": "Corn — Common rust",
        "summary": "Common rust typically forms raised rust-colored pustules on leaf surfaces.",
        "actions": [
            "Check several plants across the field to estimate how widely symptoms are distributed.",
            "Record crop stage and whether symptoms are increasing.",
            "Consult local extension guidance about resistant varieties and whether intervention is justified."
        ],
        "urgency": "Monitor crop"
    }
}

def db_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with db_connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                original_name TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()

def _remove_null_glorot_axes(value):
    changed = False
    if isinstance(value, dict):
        if value.get("class_name") == "GlorotUniform" and isinstance(value.get("config"), dict):
            initializer_config = value["config"]
            for key in ("input_axes", "output_axes"):
                if key in initializer_config and initializer_config[key] is None:
                    del initializer_config[key]
                    changed = True
        for child in value.values():
            changed = _remove_null_glorot_axes(child) or changed
    elif isinstance(value, list):
        for child in value:
            changed = _remove_null_glorot_axes(child) or changed
    return changed

def _load_model(tf):
    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except TypeError as exc:
        message = str(exc)
        if not any(
            f"unexpected keyword argument '{key}'" in message
            for key in ("input_axes", "output_axes")
        ):
            raise

        with zipfile.ZipFile(MODEL_PATH) as source:
            config = json.loads(source.read("config.json"))
            if not _remove_null_glorot_axes(config):
                raise
            with tempfile.TemporaryDirectory() as temp_dir:
                compatible_path = Path(temp_dir) / MODEL_PATH.name
                with zipfile.ZipFile(compatible_path, "w") as compatible:
                    for member in source.infolist():
                        content = (
                            json.dumps(config).encode("utf-8")
                            if member.filename == "config.json"
                            else source.read(member.filename)
                        )
                        compatible.writestr(member, content)
                return tf.keras.models.load_model(compatible_path, compile=False)

def load_model():
    global _model, _class_names, _model_error
    if _model is not None:
        return _model, _class_names, None
    if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
        return None, None, "No trained model found yet. Follow README.md to download the dataset and run train.py."
    try:
        import tensorflow as tf
        _model = _load_model(tf)
        _class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
        if not isinstance(_class_names, list) or not _class_names:
            raise ValueError("class_names.json must contain a non-empty JSON list.")
        if _model.output_shape[-1] != len(_class_names):
            raise ValueError(
                "The model output count does not match the class_names.json entries."
            )
        return _model, _class_names, None
    except Exception as exc:
        _model_error = str(exc)
        return None, None, f"Could not load model: {exc}"

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(path):
    image = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def treatment_for(class_name):
    key = class_name.lower().replace(" ", "_")
    if "healthy" in key:
        return TREATMENTS["healthy"]
    if key in TREATMENTS:
        return TREATMENTS[key]
    return TREATMENTS["default"]

def predict_image(path):
    model, class_names, error = load_model()
    if error:
        return None, None, error
    batch = prepare_image(path)
    predictions = model.predict(batch, verbose=0)[0]
    index = int(np.argmax(predictions))
    return class_names[index], float(predictions[index]), None

@app.route("/")
def index():
    with db_connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        recent = conn.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 5").fetchall()
    model_ready = MODEL_PATH.exists() and CLASS_NAMES_PATH.exists()
    return render_template("index.html", total=total, recent=recent, model_ready=model_ready)

@app.route("/predict", methods=["POST"])
def predict():
    uploaded = request.files.get("leaf_image")
    if not uploaded or uploaded.filename == "":
        flash("Choose a leaf image first.", "error")
        return redirect(url_for("index"))
    if not allowed_file(uploaded.filename):
        flash("Please upload a PNG, JPG, JPEG, or WEBP image.", "error")
        return redirect(url_for("index"))

    from werkzeug.utils import secure_filename
    safe_original = secure_filename(uploaded.filename) or "leaf-image"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    saved_name = f"{stamp}_{safe_original}"
    saved_path = UPLOAD_DIR / saved_name
    try:
        raw = uploaded.read()
        if len(raw) > 10 * 1024 * 1024:
            flash("Image is too large. Please use an image under 10 MB.", "error")
            return redirect(url_for("index"))
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        image.save(saved_path, format="JPEG", quality=92)
    except (UnidentifiedImageError, OSError):
        flash("That file does not appear to be a valid image.", "error")
        return redirect(url_for("index"))

    class_name, confidence, error = predict_image(saved_path)
    if error:
        saved_path.unlink(missing_ok=True)
        flash(error, "error")
        return redirect(url_for("index"))

    recommendation = treatment_for(class_name)
    with db_connect() as conn:
        conn.execute(
            "INSERT INTO predictions (filename, original_name, predicted_class, confidence, created_at) VALUES (?, ?, ?, ?, ?)",
            (saved_name, safe_original, class_name, confidence, datetime.now().isoformat(timespec="seconds"))
        )
        conn.commit()
        prediction_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    return render_template(
        "result.html",
        filename=saved_name,
        original_name=safe_original,
        class_name=class_name.replace("___", " — ").replace("_", " "),
        raw_class=class_name,
        confidence=round(confidence * 100, 1),
        recommendation=recommendation,
        prediction_id=prediction_id
    )

@app.route("/history")
def history():
    with db_connect() as conn:
        rows = conn.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 100").fetchall()
    return render_template("history.html", rows=rows)

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route("/health")
def health():
    _, _, error = load_model()
    return {"status": "ok", "model_ready": error is None, "model_message": error or "Model loaded"}

init_db()

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
