import numpy as np
from flask import Flask, request, make_response, render_template
from sklearn.metrics import pairwise_distances
from nltk.corpus import stopwords
from flask_cors import cross_origin
import re
import json
import pickle

app = Flask(__name__)
model = pickle.load(open('tfidf_model.pkl', 'rb'))
tfidf_features = pickle.load(open('model_features.pkl', 'rb'))

@app.route('/webhook', methods=['POST'])
@cross_origin()
def webhook():
    req = request.get_json(silent=True, force=True)
    res = manage_query(req)
    res = json.dumps(res, indent=4)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r

# processing the query using regular expression before transformation
def process_query(query):
    preprocessed_review = []
    sentence = re.sub("\S*\d\S*", "", query).strip()
    sentence = re.sub("[^A-Za-z]+", ' ', sentence)
    sentence = ' '.join(e.lower() for e in sentence.split() if e.lower() not in stopwords.words('english'))
    preprocessed_review.append(sentence.strip())
    return preprocessed_review

def manage_query(req):
    result = req.get("queryResult")
    original_query = str(result.get("queryText"))

    query = process_query(original_query)
    # query_transformed = model.transform(query)
    # pairwise_dist = pairwise_distances(tfidf_features, query_transformed)
    # index = np.argsort(pairwise_dist.flatten())[0]


    # return {
    #           "fulfillmentMessages": [
    #             {
    #               "text": {
    #                 "text": [
    #                   str(index)
    #                 ]
    #               }
    #             }
    #           ]
    #         }
    return {
              "fulfillmentMessages": [
                {
                  "text": {
                    "text": [
                      query
                    ]
                  }
                }
              ]
            }

if __name__ == '__main__':
    app.run(debug = True)
