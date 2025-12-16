from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.sql.types import DoubleType
from datetime import datetime
import sqlite3
import os

# ---------------------------
# CONFIGURATIONS
# ---------------------------
MODEL_DIR = "models/lr_v1"
MODEL_VERSION = "v1.0"
DB_PATH = "predictions.db"

# ---------------------------
# INIT FLASK APP
# ---------------------------
app = Flask(__name__)

# ---------------------------
# INIT SPARK SESSION
# ---------------------------
spark = SparkSession.builder.appName("DiabetesAPI").getOrCreate()

# ---------------------------
# LOAD MODEL
# ---------------------------
try:
    lr_model = LogisticRegressionModel.load(MODEL_DIR)
    print("Model loaded successfully!")
except Exception as e:
    print("MODEL LOAD ERROR:", str(e))
    lr_model = None

# ---------------------------
# FEATURES
# ---------------------------
feature_cols = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# ---------------------------
# DB SETUP (ONLY RUNS ONCE)
# ---------------------------
def create_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            model_version TEXT,
            input_data TEXT,
            prediction INTEGER,
            probability REAL
        )
        """
    )
    conn.commit()
    conn.close()

create_db()

# ---------------------------
# HOME ROUTE
# ---------------------------
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Diabetes prediction API is running!",
        "model_version": MODEL_VERSION
    })

# ---------------------------
# PREDICTION ROUTE
# ---------------------------
@app.route('/predict', methods=['POST'])
def predict():

    # Check model loaded
    if lr_model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.json

        # Validate required fields
        for col in feature_cols:
            if col not in data:
                return jsonify({"error": f"Missing feature: {col}"}), 400

        # Convert to Spark DataFrame
        input_df = spark.createDataFrame([data])

        # Cast columns to double
        for col in feature_cols:
            input_df = input_df.withColumn(col, input_df[col].cast(DoubleType()))

        # Feature Vector
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        input_df = assembler.transform(input_df)

        # Predict
        pred_df = lr_model.transform(input_df)
        pred_row = pred_df.select("prediction", "probability").first()

        prediction = int(pred_row["prediction"])
        probability = float(pred_row["probability"][1])  # prob of class 1 (diabetic)

        # Save to SQLite DB
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO predictions (timestamp, model_version, input_data, prediction, probability) VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                MODEL_VERSION,
                str(data),
                prediction,
                probability
            )
        )
        conn.commit()
        conn.close()

        # Response
        return jsonify({
            "model_version": MODEL_VERSION,
            "prediction": prediction,
            "probability": probability
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------
# START FLASK
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)