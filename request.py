import requests
import pandas as pd
import json

from zipfile import ZipFile
def load_data():

        z = ZipFile("data/X_sample_30.zip")
        sample = pd.read_csv(z.open('X_sample.csv'), encoding ='utf-8')

        return sample

# URL
url = 'http://127.0.0.1:8000/predict'
#url = 'https://home-credit-risk.herokuapp.com/predict'

# User ID to be provided in input form. Currently hard-coded to test.
user_id = 100004

# Get the data from the POST request.
df = load_data()
df.iloc[:,:126]

# Converting dataset to a JSON file containing all the features
df_small = df[df['SK_ID_CURR'] == int(user_id)]
df_small.drop(columns = ['SK_ID_CURR'], inplace = True)
df_small = df_small.to_dict()
input_data = json.dumps(df_small)
input_data = json.loads(input_data)

# Post JSON file
r = requests.post(url, json = input_data)

# Visualize response
print(r.json())