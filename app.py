import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from zipfile import ZipFile
import json
from flask_caching import Cache

config = {
    "DEBUG": True,          # some Flask specific configs
    "CACHE_TYPE": "SimpleCache",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300
}
app = Flask(__name__)
# tell Flask to use the above defined config
app.config.from_mapping(config)
cache = Cache(app)
#----------------------------------- Functions ---------------------------------------#
@cache.cached()
def load_data():
        z = ZipFile("data/data_default_risk.zip")
        data = pd.read_csv(z.open('application_train.csv'), index_col='SK_ID_CURR', encoding ='utf-8')

        z = ZipFile("data/X_sample_30.zip")
        sample = pd.read_csv(z.open('X_sample.csv'), index_col='SK_ID_CURR', encoding ='utf-8')

        return data, sample
@cache.cached()
def load_model():
        '''loading the trained model'''
        pickle_in = open('model/RandomForestClassifier.pkl', 'rb') 
        clf = pickle.load(pickle_in)
        return clf
@cache.cached()
def identite_client(data, id):
        data_client = data[data.index == int(id)]
        return data_client

@cache.cached()
def load_prediction(sample, id, clf):
        X=sample.iloc[:, :126]
        score = clf.predict_proba(X[X.index == int(id)])[:,1]
        predict = clf.predict(X[X.index == int(id)])
        return score, predict
        
#Chargement des donner :
data, sample = load_data()
id_client = sample.index.values
clf = load_model()
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