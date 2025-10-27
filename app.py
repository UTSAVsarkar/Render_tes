from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import traceback

app = Flask(__name__)
CORS(app)

# --- Load model safely ---
try:
    loaded = pickle.load(open("model.pkl", "rb"))
    model = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

# --- Example mappings (must match training preprocessing) ---
gender_map = {"Male": 0, "Female": 1}
education_map = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
job_map = {"Data Scientist": 0, "Software Engineer": 1, "Manager": 2, "Analyst": 3}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded on server.'}), 500

        data = request.get_json()
        print("📥 Received:", data)

        # Extract & encode
        age = float(data['Age'])
        gender = gender_map.get(data['Gender'], 0)
        education = education_map.get(data['Education Level'], 0)
        job = job_map.get(data['Job Title'], 0)
        exp = float(data['Years of Experience'])

        X = np.array([[age, gender, education, job, exp]])
        salary = model.predict(X)[0]

        return jsonify({'Predicted Salary': float(round(salary, 2))})

    except Exception as e:
        print("🔥 Error in /predict:", e)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return "✅ Salary prediction API is running fine!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
