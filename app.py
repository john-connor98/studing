from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, make_response, render_template
from sklearn.metrics import pairwise_distances
from flask_cors import cross_origin
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pandas as pd
import re
import json
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


app = Flask(__name__)
app.debug = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgres://ohtsqhdzdchoeo:142c974e0814071715177214be565c4dbb01a7c942e8c6f3e5da115d7d1284b3@ec2-3-91-139-25.compute-1.amazonaws.com:5432/d6n5re5cos3ja0'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class studdata(db.Model):
    __tablename__ = 'studydata'
    Id = db.Column(db.Integer, primary_key=True)
    Question = db.Column(db.Text(), nullable = False)
    Answer = db.Column(db.Text(), nullable = False)

    def __init__(self, Id, Question, Answer):
        self.Id = Id
        self.Question = Question
        self.Answer = Answer

# model = pickle.load(open('tfidf_model.pkl', 'rb'))
model = TfidfVectorizer()
quest_tuple_list = db.session.query(studdata.Question).all()
quest_list = [str(value) for value, in quest_tuple_list]
subans = ' '.join(quest_list)
# quest_dataframe = pd.DataFrame(quest_list)
# tfidf_features = model.fit_transform(quest_dataframe)


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
#     query_transformed = model.transform(query)
#     pairwise_dist = pairwise_distances(tfidf_features, query_transformed)
#     index = np.argsort(pairwise_dist.flatten())[0]
#     if index==None:
#         ans = "sorry check the database "
#     else:
#         ans = str(db.session.query(studdata).get(index))

    return {
              "fulfillmentMessages": [
                {
                  "text": {
                    "text": [
                      subans
                    ]
                  }
                }
              ]
            }
    # return {
    #           "fulfillmentMessages": [
    #             {
    #               "text": {
    #                 "text": [
    #                   query
    #                 ]
    #               }
    #             }
    #           ]
    #         }

if __name__ == '__main__':
    app.run()
