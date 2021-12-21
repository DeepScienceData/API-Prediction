import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from zipfile import ZipFile

app = Flask(__name__)
model = pickle.load(open('./model/RandomForestClassifier.pkl', 'rb'))

#----------------------------------- Functions ---------------------------------------#
def load_data():
        z = ZipFile("data/data_default_risk.zip")
        data = pd.read_csv(z.open('application_train.csv'), index_col='SK_ID_CURR', encoding ='utf-8')

        z = ZipFile("data/X_sample_30.zip")
        sample = pd.read_csv(z.open('X_sample.csv'), index_col='SK_ID_CURR', encoding ='utf-8')

        return data, sample

def load_model():
        '''loading the trained model'''
        pickle_in = open('model/RandomForestClassifier.pkl', 'rb') 
        clf = pickle.load(pickle_in)
        return clf

def identite_client(data, id):
        data_client = data[data.index == int(id)]
        return data_client

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