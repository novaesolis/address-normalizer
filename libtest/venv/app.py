import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import re
import scipy
import gzip
import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request
import pickle

TOKEN_RE = re.compile(r'[\w\d]+')  #regular expression to start with

def tokenize_text_simple_regex(txt, min_token_size=2):
    """ This func tokenize text with TOKEN_RE applied ealier """
    txt = txt.lower()
    all_tokens = TOKEN_RE.findall(txt)
    return [token for token in all_tokens if len(token) >= min_token_size]

# def tokenize_text_simple_regex(txt, min_token_size=2):
#     """ This func tokenize text with TOKEN_RE applied ealier """
#     txt = txt.lower()
#     all_tokens = TOKEN_RE.findall(txt)
#     return [token for token in all_tokens if len(token) >= min_token_size]
#
# def tokenize_corpus(texts, tokenizer=tokenize_text_simple_regex, **tokenizer_kwargs):
#     """
#     This func tokenize corpus of docs
#     """
#     return [tokenizer(text, **tokenizer_kwargs) for text in texts]

# load model
# model = pickle.load(open('model_dj02.sav','rb'))

loaded_model = None
clf = 'model_detection_level_01.pkl'
print("Loading the model...")
with open('./models/' + clf, 'rb') as f:
    loaded_model = joblib.load(f)

print("The model has been loaded...doing predictions now...")

# app
app = Flask(__name__)

# routes
@app.route('/address-normalizer', methods=['POST'])
def normalize_one():



    try:
        content = request.data
        test = content
        #return jsonify({"address": test})
    except Exception as e:
        raise e

    if test == "":
        return (bad_request())

    else:
    #Load the saved model

        predictions = loaded_model.predict(test)

        """Add the predictions as Series to a new pandas dataframe
                                OR
           Depending on the use-case, the entire test data appended with the new files
        """
        prediction_series = list(pd.Series(predictions))

        final_predictions = pd.DataFrame(list(zip(loan_ids, prediction_series)))

        """We can be as creative in sending the responses.
           But we need to send the response codes as well.
        """
        responses = jsonify(predictions=final_predictions.to_json(orient="records"))
        responses.status_code = 200



@app.route('/file-normalizer', methods=['POST'])
def normalize_file():

    print("The model has been loaded...doing predictions now...")
    predictions = loaded_model.predict(test)

    """Add the predictions as Series to a new pandas dataframe
                            OR
       Depending on the use-case, the entire test data appended with the new files
    """
    prediction_series = list(pd.Series(predictions))

    final_predictions = pd.DataFrame(list(zip(loan_ids, prediction_series)))

    """We can be as creative in sending the responses.
       But we need to send the response codes as well.
    """
    responses = jsonify(predictions=final_predictions.to_json(orient="records"))
    responses.status_code = 200

@app.errorhandler(400)
def bad_request(error=None):
	message = {
			'status': 400,
			'message': 'Bad Request: ' + request.url + '--> Please check your data payload...',
	}
	resp = jsonify(message)
	resp.status_code = 400

	return resp

if __name__ == '__main__':
    app.run(port = 9090, debug=True)