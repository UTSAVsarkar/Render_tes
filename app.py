from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import traceback

app = Flask(__name__)
CORS(app)

# --- Load model ---
model = pickle.load(open("model.pkl", "rb"))

# --- Example mappings (must match training preprocessing) ---
gender_map = {"Male": 0, "Female": 1}
education_map = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
job_map = {"Data Scientist": 0, "Software Engineer": 1, "Manager": 2, "Analyst": 3}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract fields
        age = float(data['Age'])
        gender = gender_map.get(data['Gender'], 0)
        education = education_map.get(data['Education Level'], 0)
        job = job_map.get(data['Job Title'], 0)
        exp = float(data['Years of Experience'])

        # Prepare numeric input for model
        X = np.array([[age, gender, education, job, exp]])

        # Predict
        salary = model.predict(X)[0]

        return jsonify({'prediction': float(salary)})

    except Exception as e:
        print("🔥 Error:", e)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return "Salary prediction API is running ✅"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
