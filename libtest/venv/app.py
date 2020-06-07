import gzip
import os
import pickle
import pickle
import re
import joblib
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy
from flask import Flask, jsonify, request
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

TOKEN_RE = re.compile(r'[\w\d]+')  #regular expression to start with

def tokenize_text_simple_regex(txt, min_token_size=2):
    txt = txt.lower()
    all_tokens = TOKEN_RE.findall(txt)
    return [token for token in all_tokens if len(token) >= min_token_size]

# load models
model_detection = None
model_country = None
model_gorod = None
model_region = None
model_rayon = None
model_mesto = None
model_type_street = None
model_type_region = None

clf_detection = 'model_detection_level_01.pkl'
clf_country = 'model_country01.pkl'
clf_gorod = 'model_gorod01.pkl'
clf_region = 'model_region01.pkl'
clf_rayon = 'model_rayon01.pkl'
clf_mesto = 'model_mesto01.pkl'
clf_type_street = 'model_type_street01.pkl'
clf_type_region = 'model_region_type01.pkl'

print("Loading the model detection...")
with open('./models/' + clf_detection, 'rb') as f:
    model_detection = joblib.load(f)

print("Loading the model country...")
with open('./models/' + clf_country, 'rb') as f:
    model_country = joblib.load(f)

print("Loading the model region...")
with open('./models/' + clf_region, 'rb') as f:
    model_region = joblib.load(f)

print("Loading the model region type...")
with open('./models/' + clf_type_region, 'rb') as f:
    model_type_region = joblib.load(f)

print("Loading the model rayon...")
with open('./models/' + clf_rayon, 'rb') as f:
    model_rayon = joblib.load(f)

print("Loading the model gorod...")
with open('./models/' + clf_gorod, 'rb') as f:
    model_gorod = joblib.load(f)

print("Loading the model mesto...")
with open('./models/' + clf_mesto, 'rb') as f:
    model_mesto = joblib.load(f)

print("Loading the model street...")
with open('./models/' + clf_type_street, 'rb') as f:
    model_type_street = joblib.load(f)

print("The models have been loaded...doing predictions now...")

def process_address(data, model):
    # gorod
    predictions = model.predict(data)
    # debug
    # print(predictions)
    prediction_series = list(pd.Series(predictions))
    if len(prediction_series) == 0:
        return (no_result())
    return prediction_series[0]

# app
app = Flask(__name__)

# routes
@app.route('/address-normalizer', methods=['POST'])
def normalize_one():
    try:
        content = request.get_json()
        test = content['address']
        #return jsonify({"address": test})
    except Exception as e:
        raise e

    if test == "":
        return (bad_request())

    else:
        # debug
        # print(test)
        list_test = [test]
        good_address = ""
        good_address = good_address + process_address(list_test, model_country)
        good_address = good_address + ";"
        good_address = good_address + process_address(list_test, model_type_region)
        good_address = good_address + " "
        good_address = good_address + process_address(list_test, model_region)
        good_address = good_address + ";"
        good_address = good_address + process_address(list_test, model_gorod)
        good_address = good_address + ";"
        good_address = good_address + process_address(list_test, model_mesto)
        good_address = good_address + ";"
        good_address = good_address + process_address(list_test, model_type_street)


        response = jsonify({"address": good_address})
        response.status_code = 200
        return response

@app.errorhandler(400)
def bad_request(error=None):
	message = {
			'status': 400,
			'message': 'Bad Request: ' + request.url + '--> Please check your data payload...',
	}
	resp = jsonify(message)
	resp.status_code = 400

	return resp

@app.errorhandler(404)
def no_result(error=None):
	message = {
			'status': 404,
			'message': 'No result for given address: ' + request.url + '--> Please check your data payload...',
	}
	resp = jsonify(message)
	resp.status_code = 400

	return resp

if __name__ == '__main__':
    app.run(port = 9090, debug=True)