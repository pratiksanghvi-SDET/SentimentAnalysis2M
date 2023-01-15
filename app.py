# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 10:34:10 2023
@author: pratiksanghvi
"""
#================ Import statements==============================================================

from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
from collections.abc import Mapping, MutableMapping
import bisect

# utilities
import re
import numpy as np
import pickle
import pandas as pd

# plotting
import seaborn as sns
import matplotlib.pyplot as plt

# nltk
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import re
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')


# sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
#==================================================================================================
app = Flask(__name__)

# -----------------------------------------------------------------------------------




# Loading Pickle files-----------------------------------------------------
pickle_in = open("TFID.pkl", "rb")
tfid_vectorizer = pickle.load(pickle_in)

pickle_in_svc = open("Sentiment-SVC.pkl", "rb")
svc = pickle.load(pickle_in_svc)

pickle_in_LR = open("Sentiment-LR.pkl", "rb")
lr = pickle.load(pickle_in_LR)

pickle_in_bnb = open("Sentiment-BNB.pkl", "rb")
bnb = pickle.load(pickle_in_bnb)


# -----------------------------------------------------------------------------
# --- Pre-processing the text data--------------------------------------
#================================================================================================
#================================================================================================
# Defining dictionary containing all emojis with their meanings.
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed',
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

## Defining set containing all stopwords in english.
#================================================================================================

# stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
#              'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
#              'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
#              'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
#              'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
#              'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
#              'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
#              'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
#              'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
#              's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
#              't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
#              'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
#              'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
#              'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
#              'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
#              "youve", 'your', 'yours', 'yourself', 'yourselves']
#================================================================================================
def preprocess(textdata):
    processedText = []

    # Create Lemmatizer and Stemmer.
    wordLemm = WordNetLemmatizer()

    # Defining regex patterns.
    urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern = '@[^\s]+'
    alphaPattern = "[^a-zA-Z0-9]"
    sequencePattern = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"

    for tweet in textdata:
        tweet = tweet.lower()

        # Replace all URls with 'URL'
        tweet = re.sub(urlPattern, ' URL', tweet)
        # Replace all emojis.
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])
            # Replace @USERNAME to 'USER'.
        tweet = re.sub(userPattern, ' USER', tweet)
        # Replace all non alphabets.
        tweet = re.sub(alphaPattern, " ", tweet)
        # Replace 3 or more consecutive letters by 2 letter.
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

        tweetwords = ''
        for word in tweet.split():
            # Checking if the word is a stopword.
            #             if word not in stopwordlist:
            if len(word) > 1:
                # Lemmatizing the word.
                word = wordLemm.lemmatize(word)
                tweetwords += (word + ' ')

        processedText.append(tweetwords)

    return processedText



@app.route('/', methods=["GET", "POST"])
def sentimentAnalysis():
    if request.method == "POST":
        sentimentAnalysisRenderTemplate()

    return render_template("sentiment_analysis.html")

@app.route('/predictSentiment', methods=["GET", "POST"])
def sentimentAnalysisRenderTemplate():
    features = []
    textData = []
    for x in request.form.values():
        features.append(x)
    tweet = str(features[0])
    textData = [tweet]

    #----------------------------------------------------------------------------
    capture_algo = ''
    if request.form['action'] == 'BNB':
        processed_text_data = tfid_vectorizer.transform(preprocess(textData))
        #processed_text_data = tfid_vectorizer.transform(textData)
        sentiment = bnb.predict(processed_text_data)
        capture_algo = 'Bernoulli Naive Bayes (BernoulliNB)'
    if request.form['action'] == 'SVC':
        processed_text_data = tfid_vectorizer.transform(preprocess(textData))
        #processed_text_data = tfid_vectorizer.transform(textData)
        sentiment = svc.predict(processed_text_data)
        capture_algo = 'Linear Support Vector Classification (LinearSVC)'

    if request.form['action'] == 'LR':
        processed_text_data = tfid_vectorizer.transform(preprocess(textData))
        #processed_text_data = tfid_vectorizer.transform(textData)
        sentiment = lr.predict(processed_text_data)
        capture_algo = 'Logistic Regression (LR)'
    # Make a list of text with sentiment.
    data = []
    for text, pred in zip(textData, sentiment):
        data.append((text, pred))
    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns=['text', 'sentiment'])
    df = df.replace([0, 1], ["Negative", "Positive"])
    print(df)
    return render_template('sentiment_analysis.html',
                           prediction_text='The sentiment of the tweet is predicted to be {} '
                                           'with accuracy of approx 80 % using {}'.format(df['sentiment'][0],
                                                                                  capture_algo))



if __name__ == '__main__':
    # app.run(host='192.168.1.100', port=8000, debug=True)# - run on local machine
    app.run(debug=True, use_reloader=False)