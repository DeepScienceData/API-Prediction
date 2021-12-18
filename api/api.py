
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
        prediction = clf.predict_proba(X[X.index == int(id)])[:,1]
        predict = clf.predict(X[X.index == int(id)])
        return prediction, predict

    #Chargement des donner :
    data, sample = load_data()
    id_client = sample.index.values
    clf = load_model()

    #######################################
    ####### HOME PAGE - MAIN CONTENT ######
    #######################################
    html_temp = """
    <div style="background-color: tomato; padding:10px; border-radius:10px">
    <h1 style="color: white; text-align:center">API Predction</h1>
    </div>
    <p style="font-size: 20px; font-weight: bold; text-align:center">Credit decision prediction</p>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    #Loading selectbox
    chk_id = st.selectbox("Client ID", id_client)
    predict,score = load_prediction(sample, chk_id, clf)
    prediction = np.round(predict, 3)*100
    st.write("**Default score : **{} ".format(float(np.round(score, 2))))
    st.write("**Default probability : **{} %".format(float(prediction)))

    # Seuil d'acceptabliter 
    
    number = st.slider("Pick threshold Decision.", 0, 100,5)


    #Compute decision according to the best threshold  (it's just a guess)
    st.write("**Decision with threshold {}** :".format(number))
    if prediction <= number :
        decision = "<font color='green'>**LOAN GRANTED**</font>"
    else:
        decision = "<font color='red'>**LOAN REJECTED**</font>"
    
    st.write("**Decision**  **: **", decision, unsafe_allow_html=True)
    

if __name__ == '__main__':
    main()