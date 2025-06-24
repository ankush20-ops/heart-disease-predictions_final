from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os
import urllib.request

app = Flask(__name__)

# Load your trained pipeline
MODEL_PATH = 'models/heart_disease_rf_optimized.pkl'
MODEL_URL = 'https://drive.google.com/uc?export=download&id=1LCBxOvygsdJSRZ9IXbTlX3DvAmd1F4Pd'

if not os.path.exists(MODEL_PATH):
os.makedirs('models', exist_ok=True)
print("Downloading model from Google Drive...")
urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
print("Download completed.")

model = joblib.load(MODEL_PATH)

@app.route('/')
def home():
    return render_template('ankush.html')

@app.route('/ankush_model', methods=['POST'])
def ankush_model():
    if request.method == 'POST':
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

            # Feature engineering
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