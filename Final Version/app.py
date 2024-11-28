import matplotlib

matplotlib.use("Agg")

from flask import Flask, render_template, request, jsonify
import pandas as pd
from utils.viz_handler import VisualizationHandler
from models.ml_handler import MLModelHandler
import os
import json
from datetime import datetime

UPLOAD_FOLDER = "data/uploads"
LOGS_FOLDER = "data/logs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOGS_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

viz_handler = VisualizationHandler()
ml_handler = MLModelHandler()

def read_csv_safely(filepath):
    try:

        chunks = pd.read_csv(filepath, chunksize=1000, encoding="utf-8")
        data = pd.concat(chunks, ignore_index=True)
        return data, None
    except Exception as e:
        try:

            data = pd.read_csv(filepath, low_memory=False, encoding="utf-8")
            return data, None
        except Exception as e:
            return None, str(e)

def save_debug_log(original_filename, data_info, visuals, error=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"debug_log_{timestamp}.txt"
    log_path = os.path.join(LOGS_FOLDER, log_filename)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Debug Log - {timestamp}\n")
        f.write(f"Original File: {original_filename}\n\n")

        f.write("Data Information:\n")
        if data_info is not None:
            f.write(f"Columns: {', '.join(data_info.columns)}\n")
            f.write(f"Shape: {data_info.shape}\n")
            f.write(f"Data Types:\n{data_info.dtypes}\n\n")

        f.write("Visualization Results:\n")
        if visuals:
            for key, value in visuals.items():
                f.write(f"{key}:\n")
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        f.write(f"  {subkey}: {subvalue}\n")
                else:
                    f.write(f"  {value}\n")

        if error:
            f.write("\nErrors Encountered:\n")
            f.write(str(error))

    return log_filename

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No selected file"}), 400

    if file and file.filename.endswith(".csv"):
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        data, error = read_csv_safely(filepath)
        if error:
            save_debug_log(file.filename, None, {}, error=error)
            return jsonify({"success": False, "error": f"Error reading file: {error}"})

        try:
            visuals = viz_handler.generate_visualizations(data)

            model_type = request.form.get("model_type", "random_forest")
            retrain = request.form.get("retrain", "false").lower() == "true"

            if retrain:
                parameters = request.form.get("parameters", "{}")
                try:
                    parameters = json.loads(parameters)
                except json.JSONDecodeError:
                    error_msg = "Invalid JSON for parameters"
                    save_debug_log(file.filename, data, visuals, error=error_msg)
                    return jsonify({"success": False, "error": error_msg}), 400

                ml_results = ml_handler.train_model(
                    data, model_type=model_type, parameters=parameters
                )
                visuals["ml_results"] = ml_results

            metadata = {
                "filename": file.filename,
                "model_type": model_type,
                "retrained": retrain,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            log_filename = save_debug_log(file.filename, data, visuals)

            return jsonify(
                {
                    "success": True,
                    "html": render_template(
                        "results.html", visuals=visuals, metadata=metadata
                    ),
                    "debug_log": log_filename,
                }
            )

        except Exception as e:
            save_debug_log(file.filename, data, {}, error=str(e))
            return jsonify(
                {"success": False, "error": f"Processing error: {str(e)}"}
            ), 500

    return jsonify({"success": False, "error": "Invalid file format"}), 400

@app.route("/predict", methods=["POST"])
def predict():
    content = request.json
    if not content:
        return jsonify({"error": "No data provided"}), 400

    model_type = content.get("model_type")
    if not model_type:
        return jsonify({"error": "Model type not specified"}), 400

    ml_handler.load_latest_model(model_type)

    data = content.get("data")
    if not data:
        return jsonify({"error": "No prediction data provided"}), 400

    try:
        predictions = ml_handler.predict(data)
        return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

@app.route("/clear-model-cache", methods=["POST"])
def clear_model_cache():
    success = ml_handler.clear_model_cache()
    return jsonify({"success": success})

@app.route("/api/models", methods=["GET"])
def get_available_models():
    """Return a list of available models and their configurations"""
    available_models = {
        "random_forest": {
            "name": "Random Forest Classifier",
            "parameters": {
                "n_estimators": "int (default: 100)",
                "max_depth": "int or None (default: None)",
                "min_samples_split": "int (default: 2)",
            },
        },
        "logistic_regression": {
            "name": "Logistic Regression",
            "parameters": {
                "C": "float (default: 1.0)",
                "max_iter": "int (default: 100)",
                "penalty": "str: 'l1' or 'l2' (default: 'l2')",
            },
        },
    }
    return jsonify(available_models)

@app.route("/api/health", methods=["GET"])
def health_check():
    """API health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
        }
    )

if __name__ == "__main__":
    app.run(debug=True)