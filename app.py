from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
import pickle
import sqlite3

# Initialize Flask app
app = Flask(__name__)

# Get absolute paths for better reliability
BASE_DIR = os.getcwd()
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
CHARTS_FOLDER = os.path.join(BASE_DIR, 'charts')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'chest_xray_model.h5')
CLASS_INDICES_PATH = os.path.join(BASE_DIR, 'model', 'class_indices.pkl')
DB_PATH = os.path.join(BASE_DIR, 'patient_data.db')

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHARTS_FOLDER, exist_ok=True)

# Load model and class indices
try:
    model = load_model(MODEL_PATH)
    with open(CLASS_INDICES_PATH, 'rb') as f:
        class_indices = pickle.load(f)
    index_to_class = {v: k for k, v in class_indices.items()}
except Exception as e:
    print(f"Error loading model or class indices: {e}")
    index_to_class = {}

# Initialize Database
def init_db():
    """Create patients table if it does not exist."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_name TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                age INTEGER NOT NULL,
                doctor TEXT NOT NULL,
                filename TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database initialization error: {e}")

init_db()

# Function to preprocess image
def preprocess_image(img_path):
    """Preprocesses an image for model prediction."""
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Route for home
@app.route('/')
def home():
    return render_template('project.html')

# Route to handle file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'xray_image' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['xray_image']
        patient_name = request.form.get('patient_name', '').strip()
        patient_id = request.form.get('patient_id', '').strip()
        age = request.form.get('age', '').strip()
        doctor = request.form.get('doctor', '').strip()
        gender = request.form.get('gender', '').strip()

        # Validate inputs
        if not all([patient_name, patient_id, age, doctor, gender]):
            return jsonify({"error": "Missing required fields"}), 400
        if not age.isdigit():
            return jsonify({"error": "Age must be a number"}), 400

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        print(f"📁 File saved to: {file_path}")

        # Process and predict
        img_array = preprocess_image(file_path)
        predictions = model.predict(img_array)[0]
        predicted_class_index = np.argmax(predictions)
        predicted_label = index_to_class.get(predicted_class_index, "Unknown")
        confidence = float(predictions[predicted_class_index]) * 100  

        # Generate Bar Chart
        chart_path = os.path.join(CHARTS_FOLDER, f"{file.filename}_chart.png")
        plt.figure(figsize=(6, 4))
        plt.bar(index_to_class.values(), predictions * 100, color=['red', 'blue', 'green'])
        plt.xlabel("Classes")
        plt.ylabel("Confidence Level (%)")
        plt.title("Prediction Confidence Levels")
        plt.ylim(0, 100)
        plt.savefig(chart_path)
        plt.close()

        print(f"📊 Chart saved to: {chart_path}")

        # Save patient data to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO patients (patient_name, patient_id, age, doctor, filename, predicted_class, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)""", 
            (patient_name, patient_id, int(age), doctor, file.filename, predicted_label, confidence))
        conn.commit()
        conn.close()
        
        return jsonify({
            "filename": file.filename,
            "patient_name": patient_name,
            "patient_id": patient_id,
            "age": age,
            "doctor": doctor,
            "gender": gender,
            "predicted_class": predicted_label,
            "confidence": f"{confidence:.2f}",
            "image_url": f"/uploads/{file.filename}",
            "chart_url": f"/charts/{file.filename}_chart.png"
        })
    
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Routes to serve static files
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/charts/<path:filename>')
def chart_file(filename):
    return send_from_directory(CHARTS_FOLDER, filename)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
