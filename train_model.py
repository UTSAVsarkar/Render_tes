# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv('Salary Data.csv')

df.dropna(inplace=True)

# Encode categorical columns
le_gender = LabelEncoder()
le_edu = LabelEncoder()
le_job = LabelEncoder()

df['Gender'] = le_gender.fit_transform(df['Gender'])
df['Education Level'] = le_edu.fit_transform(df['Education Level'])
df['Job Title'] = le_job.fit_transform(df['Job Title'])

# Split features and target
X = df[['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']]
y = df['Salary']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model and encoders
with open('model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'le_gender': le_gender,
        'le_edu': le_edu,
        'le_job': le_job
    }, f)

print("✅ Model saved as model.pkl")