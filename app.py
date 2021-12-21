import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from zipfile import ZipFile
import json

app = Flask(__name__)
model = pickle.load(open('./model/RandomForestClassifier.pkl', 'rb'))


@app.route("/")
def hello():
    """
    Ping the API.
    """
    return jsonify({"text":"Hello, the API is up and running..." })

@app.route('/predict', methods=['POST'])
def predict():
    # parse input features from request
    request_json = request.get_json()
    df = pd.json_normalize(request_json)
 
    # load model
    prediction = model.predict_proba(df)[:, 1][0]
    # Format prediction in percentage with 2 decimal points
    prediction = "The client has a " + str(round(prediction*100,2)) + "% risk of defaulting on their loan."
    print("prediction: ", prediction)

    # Return output
    return jsonify(json.dumps(str(prediction)))

if __name__ == '__main__':
    app.run(debug=True)