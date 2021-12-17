#API FLASK run (commande : python api/api.py)
# Local Adresse :  http://127.0.0.1:5000/credit/IDclient
# serve(app, host="0.0.0.0", port=8080)
#web: gunicorn  --bind 0.0.0.0:$PORT api:app
from waitress import serve


#---------------------------------- Libarie ---------------------------------------#
from zipfile import ZipFile
import pickle
from flask import Flask, render_template, jsonify, request, flash, redirect, url_for
from flask_wtf import Form, validators  
from wtforms.fields import StringField
from wtforms import TextField, BooleanField, PasswordField, TextAreaField, validators
from wtforms.widgets import TextArea





# Création d'une instance FLASK
app = Flask(__name__)
#api = Api(app)

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



#--------------------- Creation of methode for API -----------------------------------------------------------#
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

#api.add_resource(credit)

#formulaire d'appel à l'API (facultatif)
class SimpleForm(Form):
    form_id = TextField('id:', validators=[validators.required()])
    
    @app.route("/", methods=['GET', 'POST'])
    def form():
        form = SimpleForm(request.form)
        print(form.errors)

        if request.method == 'POST':
            form_id=request.form['id']
            print(form_id)
            return(redirect('credit/'+form_id)) 
    
        if form.validate():
            # Save the comment here.
            flash('Vous avez demandé l\'ID : ' + form_id)
            redirect('')
        else:
            flash('Veuillez compléter le champ. ')
    
        return render_template('formulaire_id.html', form=form)




#lancement de l'application
if __name__ == "__main__":
        #app.run(debug=True)
        app.run(port = 5000, debug=True, use_reloader=False)
        #serve(app, host="127.0.0.1",port=8000)

