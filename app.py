# app.py
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model and encoders
with open('model.pkl', 'rb') as f:
    data = pickle.load(f)
model = data['model']
le_gender = data['le_gender']
le_edu = data['le_edu']
le_job = data['le_job']

@app.route('/')
def home():
    return "Salary Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    content = request.get_json()
    age = content['Age']
    gender = le_gender.transform([content['Gender']])[0]
    edu = le_edu.transform([content['Education Level']])[0]
    job = le_job.transform([content['Job Title']])[0]
    exp = content['Years of Experience']

    X = np.array([[age, gender, edu, job, exp]])
    salary = model.predict(X)[0]

    return jsonify({'Predicted Salary': round(float(salary), 2)})

if __name__ == '__main__':
    app.run(debug=True)
