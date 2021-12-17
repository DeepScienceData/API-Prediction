
#---------------------------------- Libarie ---------------------------------------#
import streamlit as st
import pandas as pd
import numpy as np
from zipfile import ZipFile
import pickle

def main() :
    #----------------------------------- Functions ---------------------------------------#
    @st.cache
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
    @st.cache
    def load_prediction(sample, id, clf):
            X=sample.iloc[:, :126]
            score = clf.predict_proba(X[X.index == int(id)])[:,1]
            predict = clf.predict(X[X.index == int(id)])
            return score, predict

    #Chargement des donner :
    data, sample = load_data()
    id_client = sample.index.values
    clf = load_model()

    #######################################
    ####### HOME PAGE - MAIN CONTENT ######
    #######################################

    #Customer information display : Customer Gender, Age, Family status, Children, …
    st.header("**API Predction**")
    #Loading selectbox
    chk_id = st.sidebar.selectbox("Client ID", id_client)
    prediction = load_prediction(sample, chk_id, clf)
    st.write("**Default probability : **{:.0f} %".format(round(float(prediction)*100, 2)))
    
    


    #Compute decision according to the best threshold 50% (it's just a guess)
    if prediction <= 50.0 :
        decision = "<font color='green'>**LOAN GRANTED**</font>" 
    else:
        decision = "<font color='red'>**LOAN REJECTED**</font>"

    st.write("**Decision** *(with threshold 50%)* **: **", decision, unsafe_allow_html=True)
    
if __name__ == '__main__':
    main()