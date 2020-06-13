from flask import Flask, render_template, jsonify, request, send_from_directory, redirect
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

# Init flask app
app = Flask(__name__)

DEBUG = False

# I think this is unnecessary for what we are doing as there are no cookies
#TODO: Add in selectable features such as loss function, unigrams, bigrams, etc.

# Attribute names:

# Number of unlabled docs to show per iteration on the web
UNLABELED_COUNT = 5

# Number of labels in our data
LABELS_COUNT = 2

# Name of the user_label (for metadata on each document)
USER_LABEL_ATTR = 'user_label'

# Percentage of documents to highlight
PERCENT_HIGHLIGHT = .3

# Parameters that affect the naming of the pickle (changing these will rename
#  the pickle, generating a new pickle if one of that name doesn't already
#  exist)
PRELABELED_SIZE = 500
USER_ID_LENGTH = 5

vw_model_name = 'model_{userid}.vw'
vw_dictionary_name = '{userid}.dictionary'
vw_model_dir = 'vw_models'

default_importance = 1
ignore_adherence = 1
override_adherence = 3
STARTING_ADHERENCE = 2
STARTING_UNCERTAINTY = 1

desired_adherence_values = np.geomspace(ignore_adherence, override_adherence, num=7)
possibly_label = .5
probably_label = 1

REPUBLICAN_LABEL = 1
DEMOCRATIC_LABEL = -1
DEMOCRATIC_CUTOFF = 0.5

SCORING_DICT = {
                    'accuracy': make_scorer(accuracy_score)
               }

Corpus = namedtuple('Corpus', 'documents vocabulary metadata')
Document = namedtuple('Document', 'text tokens metadata')
Token = namedtuple('Token', 'token loc')

NOPUNCT = str.maketrans('', '', string.punctuation)
GOLD_ATTR_NAME = 'party'
TWEET_ID = 'tweet_id'
DATE_CREATED = 'created_date'
INPUT_UNCERTAINTY = 'input_uncertainty'
url_find = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

LABELS = ['D', 'R']
DATASET_NAME = 'twitter'
PICKLE_FOLDER_BASE = 'pickled_files'
subfolder = f'prelabeled_{PRELABELED_SIZE}'
PICKLE_FOLDER = os.path.join(PICKLE_FOLDER_BASE, DATASET_NAME, subfolder)
os.makedirs(PICKLE_FOLDER, exist_ok=True)

user_file = 'users_final.json'
tweet_file = 'tweets.json'
filename = (f'corpus.pickle')
corpus_filename = os.path.join(PICKLE_FOLDER, filename)

def pickle_cache(pickle_filename):
    def decorator(fn):
        def wrapper():
            if os.path.isfile(pickle_filename):
                with open(pickle_filename, 'rb') as f:
                    results = pickle.load(f)
                return results

            results = fn()
            with open(pickle_filename, 'wb') as f:
                pickle.dump(results, f)
            return results

        return wrapper

    return decorator

@pickle_cache(corpus_filename)
def get_twitter_corpus():
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

    c = Corpus(documents, vocabulary, metadata)

    return c

def parse_args():
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass
    parser=argparse.ArgumentParser(
        description='Used for hosting a human-document classification interface with a given dataset',
        formatter_class=CustomFormatter)
    parser.add_argument('port', nargs='?', default=5000, type=int, help='Port to be used in hosting the webpage')
    parser.add_argument('-c', '--clean', action='store_true')
    parser.add_argument('-l', '--loss', default='logistic', type=str, choices=['logistic', 'hinge'], required=False)
    parser.add_argument('-n', '--ngrams', default=1, type=int, choices=set(range(5)), required=False)
    parser.add_argument('-s', '--seed', default=14, type=int, required=False)
    return parser.parse_args()


args = parse_args()
PORT = args.port
clean = args.clean
lossfn = args.loss
ngrams = args.ngrams


# Set the attr_name for the true label
# 'binary_rating' contains 'negative' and 'positive' for yelp, amz, and TA
corpus = get_twitter_corpus()

#rng = random.Random(args.seed)

# Place to save pickle files


filename = (f'corpus.pickle')
corpus_filename = os.path.join(PICKLE_FOLDER, filename)

# Checks to see if on second stage initializaiton for Flask
if clean and os.environ.get('WERKZEUG_RUN_MAIN') == 'true': # If clean, remove file and remake
    with contextlib.suppress(FileNotFoundError):
        os.remove(corpus_filename)

if clean: # If clean, remove file and remake
    with contextlib.suppress(FileNotFoundError):
        os.remove(corpus_filename)

def load_initial_data():
    print('***Loading initial data...')

    print('***Splitting labeled/unlabeled and test...')
    # Split to labeled and unlabeled (80-20)
    ss = ShuffleSplit(n_splits=1, test_size=2000, random_state=args.seed)

    republican_documents = [d for d in corpus.documents if d.metadata['party'] == 'R']
    democratic_documents = [d for d in corpus.documents if d.metadata['party'] == 'D']

    democratic_documents.extend(republican_documents[:len(democratic_documents)])

    train_ids = list()
    test_ids = list()
    for train_indices, test_indices in ss.split(democratic_documents):
        train_ids = train_indices
        test_ids = test_indices

    train_corpus = Corpus([democratic_documents[t] for t in train_ids], corpus.vocabulary, corpus.metadata)
    test_corpus = Corpus([democratic_documents[t] for t in test_ids], corpus.vocabulary, corpus.metadata)

    # Must have at least one labeled for each label

    starting_labeled_labels = set()
    all_label_set = set(LABELS)
    V = len(train_corpus.vocabulary)
    labels = {l: V + i for i, l in enumerate(LABELS)}
    labels = sorted(labels, key=labels.get)

    # TODO: What is the significance of this?
    while starting_labeled_labels != all_label_set:
        # random
        #starting_labeled_ids = set(random.sample(range(len(train_corpus.documents)), PRELABELED_SIZE))
        starting_labeled_ids = set(range(PRELABELED_SIZE))
        starting_labeled_labels = set(train_corpus.documents[i].metadata[GOLD_ATTR_NAME] for i in starting_labeled_ids)
        # TODO
        break

    c = Counter()

    for doc in train_corpus.documents:
        doc.metadata[INPUT_UNCERTAINTY] = STARTING_UNCERTAINTY
        c.update([doc.metadata[GOLD_ATTR_NAME]])

    return (labels, train_ids, train_corpus, test_ids, test_corpus, starting_labeled_ids)

class UserList:
    """List of user data in memory on the server"""
    def __init__(self, user_base_dir='user_logs', timeout=20, zfill=3):
        # Directory to save user info
        self.user_base_dir = user_base_dir
        os.makedirs(self.user_base_dir, exist_ok=True)
        self.zfill = zfill

        # Timeout in seconds
        self.timeout = timeout*60

        self.user_id_length = USER_ID_LENGTH

        # Dictionary mapping user_id to user data
        self.users = {}

    def is_duplicate(self, user_id):
        if user_id in os.listdir(self.user_base_dir):
            return True
        return False

    def generate_user_id(self):
        user_id = ''.join([random.choice(string.ascii_letters)
                           for i in range(self.user_id_length)]).upper()
        while self.is_duplicate(user_id):
            user_id = ''.join([random.choice(string.ascii_letters)
                               for i in range(self.user_id_length)]).upper()
        return user_id

    def add_user(self, corpus_file):

        user_id = self.generate_user_id()

        # Make the directory
        user_dir = self.get_user_dir(user_id)
        os.makedirs(user_dir, exist_ok=True)


        # Set up labeling
        web_unlabeled_ids = set()
        unlabeled_ids = {*range(len(train_corpus.documents))}
        unlabeled_ids.difference_update(STARTING_LABELED_IDS)
        labeled_docs = {i: train_corpus.documents[i].metadata[GOLD_ATTR_NAME]
                        for i in STARTING_LABELED_IDS}

        user = {'user_id': user_id,
                'labeled_docs': labeled_docs,
                'web_unlabeled_ids': set(),
                'unlabeled_ids': unlabeled_ids,
                'update_time': time.time(),
                'update_num': 0,
                'corpus_file': corpus_file,
                'user_dir': user_dir,
                'fc_acc': None,
                'lr_acc': None,
               }


        # Add the user to the dictionary
        self.users[user_id] = user
        return user_id

    def get_user_dir(self, user_id):
        return os.path.join(self.user_base_dir, user_id)

    def get_filename(self, user_id, update_num, ext='.pickle'):
        return os.path.join(self.get_user_dir(user_id),
                            str(update_num).zfill(self.zfill)+user_id+ext)

    def has_user(self, user_id):
        if not user_id:
            return False
        if user_id in self.users:
            return True
        if user_id in os.listdir(self.user_base_dir):
            return True
        return False

    def load_last_update(self, user_id):
        print('load last update for', self.get_user_dir(user_id))
        updates = [filename for filename in os.listdir(self.get_user_dir(user_id))
                   if user_id in filename]
        # some users do not have any updates
        if(len(updates)==0):
            print('no updates for user:', user_id)
            self.users[user_id] = None
            return
        last_update = sorted(updates)[-1]

        # QUESTION Why is this in here?
        time.sleep(2)

        try:
            self.load_update(user_id, int(last_update[:self.zfill]))
        except:
            print(updates)
            print(self.get_user_dir(user_id))

            sys.exit(1)

    def load_update(self, user_id, update_num):
        print(f'Loading {user_id} update number {update_num}')
        full_filename = self.get_filename(user_id, update_num)
        with open(full_filename, 'rb') as infile:
            user = pickle.load(infile)
        self.users[user_id] = user

    def undo_user(self, user_id):
        self.load_update(user_id, self.users[user_id] - 1)

    def save_user(self, user_id, remove=False, check_timeout=True):
        user = self.users[user_id]

        user['update_num'] += 1
        user['update_time'] = time.time()

        # Check to see if any users are inactive
        if check_timeout:
            self.check_user_timeout()

        full_filename = self.get_filename(user_id, user['update_num'])
        with open(full_filename, 'wb') as outfile:
            pickle.dump(user, outfile)

        if remove:
            print('*')
            print('removing', user_id)
            print('*')
            self.users.pop(user_id)

    def get_user_data(self, user_id):
        if user_id not in self.users:
            self.load_last_update(user_id)
        return self.users[user_id]

    def rem_user(self, user_id):
        if user_id in self.users:
            self.users.pop(user_id)
        else:
            print(f'unable to remove {user_id} from memory')

    def check_user_timeout(self):
        delete_list = []
        for user_id, user in self.users.items():
            if time.time()-user['update_time'] > self.timeout:
                delete_list.append(user_id)
        for user_id in delete_list:
            print(f'Saving {user_id} to file and removing from memory')
            self.save_user(user_id, remove=True,
                           check_timeout=False)

    def print_user_ids(self):
        print('***')
        print('user_ids')
        for user_id in self.users:
            print(user_id)
        print('***')

def start_new_vowpal_model(userid):
    print('Starting new model for user', userid)
    vw = initialize_vowpal_model(userid, True)

    corpus_text = np.asarray([train_corpus.documents[doc_id].text.strip('\n') for doc_id in set(STARTING_LABELED_IDS)])
    y = [1 if train_corpus.documents[doc_id].metadata['party'] == 'R' else -1 for doc_id in set(STARTING_LABELED_IDS)]

    np.random.shuffle(corpus_text)
    train_vw(vw, corpus_text, y, STARTING_ADHERENCE, np.ones(len(corpus_text)))

    #np.random.shuffle(corpus_text)
    #train_vw(vw, corpus_text, y, STARTING_ADHERENCE, np.ones(len(corpus_text)))

    return vw

def save_user_dictionary(userid, dictionary):
    dictionary_file_path = vw_dictionary_name.format(userid=userid)

    with open(dictionary_file_path, 'wb') as f:
        pickle.dump(dictionary, f)

def load_user_dictionary(userid):
    dictionary_file_path = vw_dictionary_name.format(userid=userid)
    if os.path.isfile(dictionary_file_path):
        with open(dictionary_file_path, 'rb') as f:
            user_dictionary = pickle.load(f)

    else:
        user_dictionary = dict()

    return user_dictionary

def initialize_vowpal_model(userid, start_new_model=True):
    user_model_name = vw_model_name.format(userid=userid)
    user_model_filename = os.path.join(vw_model_dir, user_model_name)

    if(start_new_model):
        vw = pyvw.vw(quiet=True, f=user_model_filename, loss_function='logistic', link='logistic')
    else:
        vw = pyvw.vw(quiet=True, f=user_model_filename, loss_function='logistic', link='logistic', i=user_model_filename)

    return vw

def clean_vowpal_text(text):
    return text.replace(':', ' ').replace('|', '').replace('\n', ' ')

def train_vw(vw_model, data, y, adherence, input_uncertainties):
    for i, train_doc in enumerate(data):
        cleaned_train = clean_vowpal_text(train_doc)

        input_uncertainty = input_uncertainties[i]
        document_importance = (default_importance + (adherence - 2)) * input_uncertainty

        ex = vw_model.example(f'{y[i]} {document_importance} | {cleaned_train}')
        ex.learn()

        del ex

(labels, train_ids, train_corpus, test_ids, test_corpus, STARTING_LABELED_IDS) = load_initial_data()
del corpus

for doc_id in STARTING_LABELED_IDS:
    doc = train_corpus.documents[doc_id]
    # Assign "user_label" to be the correct label
    doc.metadata[USER_LABEL_ATTR] = doc.metadata[GOLD_ATTR_NAME]
    doc.metadata['Prelabeled'] = True

@app.route('/index')
def index():
    return send_from_directory('.','index.html')

@app.route('/')
@app.route('/teaming')
def teaming():
    return send_from_directory('.', 'teaming.html')

@app.route('/index3')
def index3():
    return send_from_directory('.', 'index3.html')

@app.route('/answers')
def answers():
    return send_from_directory('.', 'answers.html')

@app.route('/api/allaccuracy')
def api_allaccuracy(fresh=False):
    users.print_user_ids()
    user_folder = os.path.join(PICKLE_FOLDER, 'UserData')
    user_ids = [f for f in os.listdir(user_folder) if len(f) == USER_ID_LENGTH]
    accuracy_data = {}
    for user_id in user_ids:
        # GET ACCURACY FOR BOTH FC AND LR
        accuracy_data[user_id + '_vw'] = get_vw_acc(user_id)
        accuracy_data[user_id + '_lr'] = get_lr_acc(user_id)

    ds = sum(1 for doc in test_corpus.documents if doc.metadata[GOLD_ATTR_NAME] == 'D')
    n = len(test_corpus.documents)

    print('BASELINE', f'{ds}/{n}', ds/n)

    return jsonify(accuracies=accuracy_data)

# GET - Send the vocabulary to the client
@app.route('/api/vocab')
def api_vocab():
    return jsonify(vocab=train_corpus.vocabulary)

users = UserList()

@app.route('/api/adduser', methods=['POST'])
def api_adduser():
    user_id = users.add_user(corpus_filename)
    users.print_user_ids()
    return jsonify(userId=user_id)

@app.route('/api/checkuserid/<user_id>', methods=['GET'])
def api_checkuserid(user_id):
    return jsonify(hasId=users.has_user(user_id))

@app.route('/api/getuserdata/<user_id>')
def api_getuserdata(user_id):
    user = users.get_user_data(user_id)
    web_unlabeled_ids = user['web_unlabeled_ids']
    labeled_docs = user['labeled_docs']
    unlabeled_ids = user['unlabeled_ids']


    for doc_id, label in labeled_docs.items():
        train_corpus.documents[doc_id].metadata[USER_LABEL_ATTR] = label

    ret_docs = []
    for i in labeled_docs:
        doc = train_corpus.documents[i]
        d = {}
        d['text'] = doc.text
        d['metadata'] = {k:v
                         for k, v in doc.metadata.items()
                         if type(v) != np.ndarray}
        d['metadata']['correct'] = (d['metadata'][USER_LABEL_ATTR] == d['metadata'][GOLD_ATTR_NAME])
        d['metadata']['docNum'] = i
        ret_docs.append(d)

    return jsonify(documents=ret_docs)

def write_to_logfile(text, user, uid):
    if text is not None:
        log_filename = users.get_filename(uid, user['update_num'], '.txt')
        with open(log_filename, 'w') as outfile:
            outfile.write(text)


def get_expected_prediction(doc, desired_adherence, label, input_uncertainty, userid):

    user_model_filename = os.path.join(vw_model_dir, vw_model_name.format(userid=userid))

    new_vw = pyvw.vw(quiet=True, i=user_model_filename, loss_function='logistic', link='logistic')

    # geomspace doesn't allow for non-positive values
    document_importance = (default_importance + (desired_adherence - 2)) * input_uncertainty

    doc_ex = new_vw.example(f'{label} {document_importance} | {doc}')
    new_vw.learn(doc_ex)
    prediction_confidence = new_vw.predict(doc_ex)

    if desired_adherence == override_adherence:
            if label == REPUBLICAN_LABEL:
                prediction_confidence = 1 if input_uncertainty == probably_label else .75
            else:
                prediction_confidence = 0 if input_uncertainty == probably_label else .25


    del doc_ex
    del new_vw

    return prediction_confidence

def get_expected_future_predictions(doc, userid):
    future_predictions = dict()

    future_predictions['democrat'] = dict()
    future_predictions['republican'] = dict()
    future_predictions['democrat']['possibly'] = list()
    future_predictions['democrat']['probably'] = list()
    future_predictions['republican']['possibly'] = list()
    future_predictions['republican']['probably'] = list()

    for value in desired_adherence_values:
        #print('Desired Adherence:', value)
        future_predictions['democrat']['possibly'].append(get_expected_prediction(doc, value, DEMOCRATIC_LABEL, possibly_label, userid))
        future_predictions['democrat']['probably'].append(get_expected_prediction(doc, value, DEMOCRATIC_LABEL, probably_label, userid))

        future_predictions['republican']['possibly'].append(get_expected_prediction(doc, value, REPUBLICAN_LABEL, possibly_label, userid))
        future_predictions['republican']['probably'].append(get_expected_prediction(doc, value, REPUBLICAN_LABEL, probably_label, userid))

    return future_predictions

@app.route('/api/update', methods=['POST'])
def call_update():

    p = mp.Pool(processes=1)
    data = p.map(update, range(1))
    p.close()
    process = psutil.Process(os.getpid())
    print('Process Memory:', (((process.memory_info().rss / 1000) / 1000) / 1000), 'GB')

    return data[0]

def update(i):
    print('Calling Update')
    data = request.get_json()

    global ngrams
    global lossfn

    # Data is expected to come back in this form:
    # data = {anchor_tokens: [[token_str,..],...]
    #         labeled_docs: [{doc_id: number
    #                         user_label: label},...]
    #         user_id: user_id_str
    #        }

    user_id = data.get('user_id')
    adherence = data.get('desired_adherence')
    user = users.get_user_data(user_id)
    users.save_user(user_id)
    web_unlabeled_ids = user['web_unlabeled_ids']
    labeled_docs = user['labeled_docs']
    unlabeled_ids = user['unlabeled_ids']

    model_filename = os.path.join(vw_model_dir, vw_model_name.format(userid=user_id))

    user_dictionary = load_user_dictionary(user_id)

    vw = None

    if not os.path.exists(model_filename):
        vw = start_new_vowpal_model(user_id)
    else:
        vw = initialize_vowpal_model(user_id, False)

    # Write the log file
    write_to_logfile(data.get('log_text'), user, user_id)
    users.print_user_ids()

    newly_labeled_docs = data.get('labeled_docs')

    # Label docs onto user
    for doc in newly_labeled_docs:
        user_label = doc[USER_LABEL_ATTR]
        input_uncertainty = user_label[1]
        user_label = user_label[0]
        doc_id = doc['doc_id']

        labeled_docs[doc_id] = user_label
        train_corpus.documents[doc_id].metadata[INPUT_UNCERTAINTY] = int(input_uncertainty) if input_uncertainty == '1' else .5
        unlabeled_ids.discard(doc_id)

    # Label docs into corpus
    for doc_id, label in labeled_docs.items():
        train_corpus.documents[doc_id].metadata[USER_LABEL_ATTR] = label[0]

    # Remove elements without creating new object
    web_unlabeled_ids.clear()
    web_unlabeled_ids.update(random.sample(unlabeled_ids, UNLABELED_COUNT))

    newly_labeled_doc_ids = {doc['doc_id'] for doc in newly_labeled_docs}
    labeled_ids = set(labeled_docs).union(newly_labeled_doc_ids)

    corpus_text = np.asarray([train_corpus.documents[doc_id].text for doc_id in newly_labeled_doc_ids])
    y = [REPUBLICAN_LABEL if train_corpus.documents[doc_id].metadata[USER_LABEL_ATTR] == 'R' else DEMOCRATIC_LABEL for doc_id in newly_labeled_doc_ids]

    input_uncertainties = [train_corpus.documents[doc_id].metadata[INPUT_UNCERTAINTY] for doc_id in newly_labeled_doc_ids]

    # I should add function wrappers to do this timing so its cleaner
    start = time.time()
    train_vw(vw, corpus_text, y, adherence, input_uncertainties)
    print('***Time - Train:', time.time() - start)

    vw.finish()
    vw = initialize_vowpal_model(user_id, start_new_model=False)

    test_docs = [doc.text for doc in test_corpus.documents]
    print('Test Doc Length:', len(test_docs))

    test_targets = [doc.metadata[GOLD_ATTR_NAME] for doc in test_corpus.documents]
    test_targets = [REPUBLICAN_LABEL if t == 'R' else DEMOCRATIC_LABEL for t in test_targets]

    results = int(0)

    for i, test_doc in enumerate(test_corpus.documents):
        test_doc_text = test_doc.text
        cleaned_test = clean_vowpal_text(test_doc_text)
        test_target = test_targets[i]
        ex = vw.example(f'{test_target} 1 | {cleaned_test}')
        prediction = vw.predict(ex)

        del ex

        prediction = DEMOCRATIC_LABEL if prediction < DEMOCRATIC_CUTOFF else REPUBLICAN_LABEL
        results += 1 if prediction == test_target else 0

    model_accuracy = results / len(test_targets)
    print('Model Accuracy', model_accuracy)

    # PREPARE TO SEND OBJECTS BACK

    unlabeled_docs = [train_corpus.documents[doc_id].text for doc_id in web_unlabeled_ids]
    web_tokens = list(set(' '.join(unlabeled_docs).split()))
    token_data = list()

    for i, token in enumerate(web_tokens):
        cleaned_token = clean_vowpal_text(token)

        # use a label of 1 because algorithm doesn't read it when its just predicting
        ex = vw.example(f'1 {default_importance} | {cleaned_token}')
        prediction = vw.predict(ex)
        word_label = DEMOCRATIC_LABEL if prediction < DEMOCRATIC_CUTOFF else REPUBLICAN_LABEL
        prob = prediction if word_label == REPUBLICAN_LABEL else (1 - prediction)

        del ex

        token_data.append({'token' : web_tokens[i], 'probs' : np.float32(prob), 'decision' : word_label})

    token_data.sort(key=lambda d: d['probs'], reverse=True)

    # 1 for R (Republican) and 0 for Democrat (D)
    highlight_dict = {d['token']: 0 if d['decision'] < 0 else 1
                      for d in token_data[:int( len(token_data) * PERCENT_HIGHLIGHT )]}

    def get_highlights(doc):
        highlights = []
        words = doc.split()
        for word in words:
            if word in highlight_dict:
                highlights.append((f'{word}',
                                   labels[highlight_dict[word]]))
        return highlights

    unlabeled_docs = []

    for i, doc_id in enumerate(web_unlabeled_ids):
        new_text = train_corpus.documents[doc_id].text

        cleaned_test = clean_vowpal_text(new_text)
        ex = vw.example(f'{test_target} {default_importance} | {cleaned_test}')
        prediction = vw.predict(ex)

        con = prediction
        rdif = con

        del ex

        i_label = 0 if prediction < .5 else 1
        predict_label = labels[i_label]

        hls = get_highlights(new_text)

        expected_future_predictions = get_expected_future_predictions(cleaned_test, user_id)

        unlabeled_docs.append(
           {
               'docId': doc_id,
               'text': new_text,
               'date': train_corpus.documents[doc_id].metadata[DATE_CREATED],
               'tokens': new_text.split(),
               'trueLabel': train_corpus.documents[doc_id].metadata[GOLD_ATTR_NAME], # FIXME Needs to be taken out before user study
               'prediction': {
                              'label': predict_label,
                              'confidence': con,
                              'relativeDif': rdif
                              }, # THIS IS WRONG
               'highlights': get_highlights(train_corpus.documents[doc_id].text),
               'expected_predictions' : expected_future_predictions
           }
         )

    labels_dict = {label: i for i, label in enumerate(labels)}
    # A bit of a complex sort, but gets the job done
    #TODO I don't know what this is doing and need to fix it to make sure it is behaving appropriately
    #doc_sort = lambda doc: (labels_dict[doc['prediction']['label']],
    #                        (-1)**(labels_dict[doc['prediction']['label']] + 1)
    #                            * doc['prediction']['relativeDif'])
    #unlabeled_docs.sort(key=doc_sort)

    # Calculate average for each label
    # TODO Find a better way of doing this?

    label_count = Counter()
    for doc_id in labeled_ids:
        doc = train_corpus.documents[doc_id]
        label = doc.metadata[USER_LABEL_ATTR]
        label_count[label] += 1

    return_labels = [{'labelId': i, 'label': label, 'count': label_count[label]} for i, label in enumerate(labels)]

    vw.finish()

    process = psutil.Process(os.getpid())
    print('Process Memory:', (((process.memory_info().rss / 1000) / 1000) / 1000), 'GB')

    return jsonify(labels=return_labels, unlabeledDocs=unlabeled_docs, modelAccuracy=model_accuracy)

@app.route('/api/accuracy', methods=['POST'])
def api_accuracy():
    data = request.get_json(force=True)

    user_id = data.get('user_id')

    #print('Running Accuracy')
    #print('User Id:', user_id)
    if user_id is None: return jsonify(accuracy=0.0)

    user = users.get_user_data(user_id)
    users.save_user(user_id)
    labeled_ids = user['labeled_docs']

    #print('Retrieving corpus')
    train_docs = [train_corpus.documents[docid].text for docid in labeled_ids]
    train_targets = [train_corpus.documents[docid].metadata[USER_LABEL_ATTR] for docid in labeled_ids]

    test_docs = [doc.text for doc in test_corpus.documents]
    test_targets = [doc.metadata[GOLD_ATTR_NAME] for doc in test_corpus.documents]

    train_targets = [-1 if t == 'R' else 1 for t in train_targets]
    test_targets = [-1 if t == 'R' else 1 for t in test_targets]


    #print('Vectorizing')
    tfv = CountVectorizer()
    train_vectors = tfv.fit_transform(train_docs)
    test_vectors = tfv.transform(test_docs)

    vw = VWClassifier()

    #print('Training')
    start = time.time()
    model = vw.fit(train_vectors, train_targets)
    #print('***Time - Get Classifier:', time.time() - start)

    start = time.time()
    predictions = vw.predict(test_vectors)
    #print('***Time - Classify:', time.time() - start)

    acc = accuracy_score(test_targets, predictions)

    #print('***Accuracy:', acc)
    return jsonify(accuracy=acc)


if __name__ =="__main__":
    app.run(debug=DEBUG, host='0.0.0.0', port=PORT)
