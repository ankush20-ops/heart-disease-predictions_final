from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os
import requests

app = Flask(__name__)

# Function to download model if not present
def download_model_from_gdrive():
    url = "https://drive.google.com/uc?export=download&id=1LCBxOvygsdJSRZ9IXbTlX3DvAmd1F4Pd"
    response = requests.get(url)
    os.makedirs("models", exist_ok=True)
    with open("models/heart_disease_rf_optimized.pkl", "wb") as f:
        f.write(response.content)

# Check and download model if needed
model_path = os.path.join('models', 'heart_disease_rf_optimized.pkl')
if not os.path.exists(model_path):
    os.makedirs('models', exist_ok=True)
    download_model_from_gdrive()

# Load the trained model
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('ankush.html')

@app.route('/ankush_model', methods=['POST'])
def ankush_model():
    try:
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        height = int(request.form['height'])
        weight = int(request.form['weight'])
        ap_hi = int(request.form['ap_hi'])
        ap_lo = int(request.form['ap_lo'])
        cholesterol = int(request.form['cholesterol'])
        gluc = int(request.form['gluc'])
        smoke = int(request.form['smoke'])
        alco = int(request.form['alco'])
        active = int(request.form['active'])

        # Feature engineering (must match training pipeline)
        bmi = weight / ((height / 100) ** 2)
        hypertension = int(ap_hi >= 140 or ap_lo >= 90)
        pulse_pressure = ap_hi - ap_lo

        age_group_MidAge = int(30 < age <= 45)
        age_group_Old = int(45 < age <= 60)
        age_group_VeryOld = int(age > 60)

        cholesterol_2 = int(cholesterol == 2)
        cholesterol_3 = int(cholesterol == 3)
        gluc_2 = int(gluc == 2)
        gluc_3 = int(gluc == 3)

        input_features = np.array([[age, gender, height, weight, ap_hi, ap_lo, smoke, alco, active,
                                    bmi, hypertension, pulse_pressure,
                                    cholesterol_2, cholesterol_3, gluc_2, gluc_3,
                                    age_group_MidAge, age_group_Old, age_group_VeryOld]])

        prediction = model.predict(input_features)[0]
        probability = model.predict_proba(input_features)[0][1]

        if prediction == 1:
            message = "🔴 High likelihood of heart disease."
        else:
            message = "🟢 Low likelihood of heart disease."

        return render_template('ankush.html', prediction_result=message, probability=round(probability, 2))

    except Exception as e:
        return f"An error occurred during prediction: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)