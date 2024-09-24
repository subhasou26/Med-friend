# flask_app.py
from flask import Flask, request, jsonify,render_template
import joblib
import numpy as np
import pandas as pd
app = Flask(__name__)

# Load the saved model
model = joblib.load('all_disease_prediction.pkl')
@app.route("/")
def home():
    input_data=pd.read_csv("hyper.csv");
    # Make prediction
    prognosis = model.predict(input_data)
    print(prognosis)
    return render_template('index.html')


@app.route('/predict')
def predict():
    data = request.json  # Get data from POST request (in JSON format)
    symptoms = np.array([data['symptoms']])  # Convert to 2D array (model expects it)
    input_data=pd.read_csv("hyper.csv");
    # Make prediction
    prognosis = model.predict(input_data)
    print(prognosis)
    return jsonify({'prognosis': prognosis[0]})

if __name__ == '__main__':
    app.run(debug=True)
