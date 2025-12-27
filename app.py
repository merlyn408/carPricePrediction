from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model + encoders
model = pickle.load(open("car_price_model.pkl", "rb"))
company_encoder = pickle.load(open("company_encoder.pkl", "rb"))
fuel_encoder = pickle.load(open("fuel_encoder.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form
    company = request.form['company']
    year = int(request.form['year'])
    kms_driven = int(request.form['kms_driven'])
    fuel_type = request.form['fuel_type']

    # Encode categorical inputs using SAME encoders from training
    company_encoded = company_encoder.transform([company])[0]
    fuel_encoded = fuel_encoder.transform([fuel_type])[0]

    # Order MUST match training: ['company','year','kms_driven','fuel_type']
    features = np.array([[company_encoded, year, kms_driven, fuel_encoded]])

    # Predict
    prediction = model.predict(features)[0]

    # Return result to webpage
    return render_template('index.html', prediction_text=f"Estimated Car Price: ₹{prediction:,.2f}")

if __name__ == "__main__":
    app.run(debug=True)
