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

@app.route('/credit/<id_client>', methods=['GET'])
def credit(id_client):


    score, predict = load_prediction(sample,id_client, clf)

    
    # round the predict proba value and set to new variable
    percent_score = score*100
    id_risk = np.round(percent_score, 3)
    # create JSON object
    output = {'prediction': int(predict), 'client risk in %': float(id_risk)}


    print('Nouvelle Prédiction : \n', output)

    return jsonify(output)
if __name__ == '__main__':
    app.run(debug=True)