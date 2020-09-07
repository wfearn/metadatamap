import multiprocessing as mp
import datetime
from collections import namedtuple
import re
import numpy as np
import random
import json
import time
from collections import defaultdict, Counter
import argparse
import os
import psutil
import sys
import contextlib
import random
import string
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_validate, ShuffleSplit
from sklearn.metrics import accuracy_score, make_scorer
from vowpalwabbit.sklearn_vw import VWClassifier
from vowpalwabbit import pyvw
from sklearn.preprocessing import MinMaxScaler, minmax_scale
from pympler import muppy, summary

GOLD_ATTR_NAME = 'party'
TWEET_ID = 'tweet_id'
DATE_CREATED = 'created_date'
user_file = 'users_final.json'
tweet_file = 'tweets.json'
#tweet_file = 'full_tweets.json'
DATASET_NAME = 'twitter'
url_find = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
DEMOCRATIC_CUTOFF = 0.5

Corpus = namedtuple('Corpus', 'documents vocabulary metadata')
Document = namedtuple('Document', 'text tokens metadata')
Token = namedtuple('Token', 'token loc')

PRELABELED_SIZE = 500
PICKLE_FOLDER_BASE = 'pickled_files'
subfolder = f'prelabeled_{PRELABELED_SIZE}'
PICKLE_FOLDER = os.path.join(PICKLE_FOLDER_BASE, DATASET_NAME, subfolder)
os.makedirs(PICKLE_FOLDER, exist_ok=True)

filename = (f'corpus.pickle')
corpus_filename = os.path.join(PICKLE_FOLDER, filename)

user_dict = dict()
with open(user_file, 'r') as f:
    user_text = f.readlines()

for line in user_text:
    u = json.loads(line)
    user_dict[u['id']] = u['party']

with open(tweet_file, 'r') as f:
    tweet_text = f.readlines()

documents = list()
vocabulary = dict()
metadata = dict()

for line in tweet_text:

    tweet = json.loads(line)
    uid = tweet['user_id']

    party = user_dict[uid]
    if party == 'Independent': continue

    text = tweet['text']

    if 'RT' in text:
        #print('Retweet:', text)
        continue

    text = url_find.sub('', text)

    tokens = list()

    for word in text.split():
        if word not in vocabulary:
            vocabulary[word] = len(vocabulary)

        tokens.append(Token(vocabulary[word], (0, 0)))

    doc_metadata = dict()
    doc_metadata[GOLD_ATTR_NAME] = 'D' if party == 'Democratic' else 'R'
    doc_metadata[TWEET_ID] = tweet['id']
    doc_metadata[DATE_CREATED] = str(datetime.datetime.fromtimestamp(tweet['created_at'])).split(' ')[0]

    d = Document(f'{text}', tokens, doc_metadata)
    documents.append(d)

corpus = Corpus(documents, vocabulary, metadata)


for state in range(5):
    ss = ShuffleSplit(n_splits=1, test_size=.20, random_state=state)

    republican_documents = [d for d in corpus.documents if d.metadata['party'] == 'R']
    republican_targets = [1] * len(republican_documents)

    democratic_documents = [d for d in corpus.documents if d.metadata['party'] == 'D']
    democratic_targets = [-1] * len(democratic_documents)

    democratic_documents.extend(republican_documents[:len(democratic_documents)])
    democratic_targets.extend(republican_targets[:len(democratic_targets)])

    print('Democratic Documents Length:', len(democratic_documents))

    train_ids = list()
    test_ids = list()
    for train_indices, test_indices in ss.split(democratic_documents):
        train_ids = train_indices
        test_ids = test_indices

    train_text = [democratic_documents[t] for t in train_ids]
    train_y = [democratic_targets[t] for t in train_ids]

    test_text = [democratic_documents[t] for t in test_ids]
    test_y = [democratic_targets[t] for t in test_ids]

    vw = pyvw.vw(quiet=True, loss_function='logistic', link='logistic')

    for i, train_doc in enumerate(train_text):
        cleaned_train = train_doc.text.replace(':', ' ').replace('|', '').replace('\n', ' ')

        ex = vw.example(f'{train_y[i]} | {cleaned_train}')
        ex.learn()

        del ex

    results = 0

    for i, test_doc in enumerate(test_text):
        cleaned_test = test_doc.text.replace(':', ' ').replace('|', '').replace('\n', ' ')
        test_target = test_y[i]
        ex = vw.example(f'{test_target} | {cleaned_test}')
        prediction = vw.predict(ex)

        del ex

        prediction = -1 if prediction < DEMOCRATIC_CUTOFF else 1
        results += 1 if prediction == test_target else 0

    print('Percent Correct:', results / len(test_y))
