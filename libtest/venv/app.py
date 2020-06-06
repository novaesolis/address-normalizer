import pandas as pd
import torch
import sklearn
import numpy
import re
import os
from flask import Flask, jsonify, request
import pickle

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
model = pickle.load(open('model_dj01.sav','rb'))

# app
app = Flask(__name__)

# routes
@app.route('/address-normalizer', methods=['POST'])

def predict():
    # get data
    data = request.get_json(force=True)

    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)

    # predictions
    result = model.predict(data_df)

    # send back to browser
    output = {'results': int(result[0])}

    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 9090, debug=True)