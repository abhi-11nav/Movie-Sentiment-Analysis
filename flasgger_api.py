#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 17:49:20 2022

@author: abhinav
"""

from flask import Flask, request, render_template 
import numpy as np 
import re
import pickle 

import nltk 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize

import flasgger 
from flasgger import Swagger 

nltk.download("punkt")

lemmatizer = WordNetLemmatizer()
word2vec = pickle.load(open("word2bec.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))


app = Flask(__name__, template_folder="template")
Swagger(app)

@app.route("/", methods=["GET"])
def home():
    return "WELCOME EVERYONE"

@app.route("/predict", methods=["POST"])
def predict():
    #if request.method == "POST":
        
        """ MOVIE SENTIMENT ANALYSIS
        
        ---
        parameters:
            - name : text 
              in : query
              type : string 
              required : true 
        responses:
            200:
                description : The OUTPUT VALUES
        """
        
        text = request.args.get("text")

        
        def text_preprocess(text):
            re.sub('[^a-zA-Z]'," ", text)
            
            text = " ".join(x for x in text.split() if x not in stopwords.words("english")).lower()
                        
            words = word_tokenize(text)
            
            words = [lemmatizer.lemmatize(word) for word in words]
            
            clean_text = []
                                           
            clean_text.append(np.mean([word2vec.wv[word] for word in words if word in word2vec.wv.index2word], axis=0))
            
            return clean_text
        
        text = text_preprocess(text)
            
        prediction = model.predict(text)
        
        if prediction == 1:
            #return render_template("/index.html", prediction_text = " POSITIVE REVIEW")
            return "POSITIVE REVIEW"
        else:
            return "NEGATIVE REVIEW"
           # return render_template("/index.html", prediction_text = "NEGATIVE REVIEW")
    
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8000)
