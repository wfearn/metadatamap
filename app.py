import ankura
from flask import Flask, render_template, jsonify, request, send_from_directory, redirect
import numpy as np
import random
import json
import time
from collections import defaultdict, Counter
import argparse
import os
import sys
import contextlib
import random
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate, ShuffleSplit
from sklearn.metrics import accuracy_score, make_scorer
from vowpalwabbit.sklearn_vw import VWClassifier
from sklearn.preprocessing import MinMaxScaler, minmax_scale

# Init flask app
app = Flask(__name__)

DEBUG = False

# I think this is unnecessary for what we are doing as there are no cookies
#TODO: Add in selectable features such as loss function, unigrams, bigrams, etc.

# Attribute names:
# Token topics
Z_ATTR = 'z' # UNUSED

# Document topics
THETA_ATTR = 'theta'

# Prior attr
PRIOR_ATTR = 'lambda' # UNUSED

# Number of unlabled docs to show per iteration on the web
#UNLABELED_COUNT = 10
UNLABELED_COUNT = 25

# Seed used in the shuffle
SHUFFLE_SEED = None #8448

# Number of labels in our data
LABELS_COUNT = 2

# Param for harmonic mean (In tandem_anchors)
ta_epsilon = 1e-15

# Epsilon for recover topics
rt_epsilon = 1e-5

# percentage of words to highlight per round (top 10%)
PERCENT_HIGHLIGHT = .1

# Name of the user_label (for metadata on each document)
USER_LABEL_ATTR = 'user_label'

# Parameters that affect the naming of the pickle (changing these will rename
#  the pickle, generating a new pickle if one of that name doesn't already
#  exist)
NUM_TOPICS = 10
PRELABELED_SIZE = 25
LABEL_WEIGHT = 1
USER_ID_LENGTH = 5

# Does NOT change pickle name.
# Changing these params requires making a clean version
# (run program and include the -c or --clean argument)
smoothing = 1e-4

SCORING_DICT = {
                    'accuracy': make_scorer(accuracy_score)
               }
#smoothing = 0


# if PRELABELED_SIZE < LABELS_COUNT:
#     raise ValueError("prelabled_size cannot be less than LABELS_COUNT")



#SELECTED_ANCHOR_TOKENS = [
#    ['energy', 'oil', 'production'],
#    ['security', 'national'],
#    ['health', 'care', 'support'],
#    ['poor', 'care', 'people'],
#    ['republican', 'republicans'],
#    ['health', 'care', 'bad'],
#    ['pension', 'benefit', 'support'],
#    ['medicare', 'spending'],
#    ['people', 'vote'],
#    ['energy', 'companies'],
#    ['iraq', 'war'],
#    ['president', 'administration'],
#    ['support', 'families', 'american', 'tax'],
#    ['children', 'child'],
#    ['border', 'immigration'],
#    ['prices', 'gas', 'oil', 'energy', 'bill'],
#    ['small', 'business'],
#    ['osha', 'workers'],
#    ['social', 'security', 'massive'],
#    ['retirement', 'plan', 'benefit'],
#    ['tax', 'relief'],
#    ['economy'],
#    ['rules', 'committee'],
#    ['water'],
#    ['united', 'states'],
#    ['housing',],
#    ['defense', 'spending'],
#    ['sexual', 'children'],
#    ['congress'],
#    ['tax', 'cut'],
#    ['retirement', 'pension', 'workers'],
#    ['911', 'terrorists'],
#    ['iraq', 'people'],
#    ['democrats', 'minority'],
#    ['family']
#]

def parse_args():
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass
    parser=argparse.ArgumentParser(
        description='Used for hosting tbuie with a given dataset',
        epilog=('See https://github.com/byu-aml-lab/tbuie\n' +
                '  and https://github.com/byu-aml-lab/ankura/tree/ankura2/ankura\n' +
                '  for source and dependencies\n \n'),
        formatter_class=CustomFormatter)
    parser.add_argument('dataset', metavar='dataset',
                        choices=['yelp', 'tripadvisor', 'amazon', 'congress'],
                        help='The name of a dataset to use in this instance of tbuie')
    parser.add_argument('port', nargs='?', default=5000, type=int,
                        help='Port to be used in hosting the webpage')
    parser.add_argument('-c', '--clean', action='store_true')
    parser.add_argument('-l', '--loss', default='logistic', type=str, choices=['logistic', 'hinge'], required=False)
    parser.add_argument('-n', '--ngrams', default=1, type=int, choices=set(range(5)), required=False)
    parser.add_argument('-s', '--seed', default=86, type=int, required=False)
    return parser.parse_args()


args = parse_args()
DATASET_NAME = args.dataset
PORT = args.port
clean = args.clean
lossfn = args.loss
ngrams = args.ngrams


# Set the attr_name for the true label
# 'binary_rating' contains 'negative' and 'positive' for yelp, amz, and TA
GOLD_ATTR_NAME = 'binary_rating'

if DATASET_NAME == 'yelp':
    corpus = ankura.corpus.yelp()
    LABELS = ['negative', 'positive']
elif DATASET_NAME == 'tripadvisor':
    corpus = ankura.corpus.tripadvisor()
    LABELS = ['negative', 'positive']
elif DATASET_NAME == 'amazon':
    corpus = ankura.corpus.amazon()
    LABELS = ['negative', 'positive']
elif DATASET_NAME == 'congress':
    corpus = ankura.corpus.congress()
    LABELS = ['D', 'R']
    GOLD_ATTR_NAME = 'party'

# Set seed and shuffle corpus documents if SHUFFLE_SEED
# Was implemented in case we were doing fully semi-supervised; if there is a
#   train/test split, that will shuffle the corpus.
if SHUFFLE_SEED:
    random.seed(SHUFFLE_SEED)
    random.shuffle(corpus.documents)

# Place to save pickle files
PICKLE_FOLDER_BASE = 'PickledFiles'

subfolder = f'K{NUM_TOPICS}_prelabeled{PRELABELED_SIZE}_lw{LABEL_WEIGHT}'
PICKLE_FOLDER = os.path.join(PICKLE_FOLDER_BASE, DATASET_NAME, subfolder)
os.makedirs(PICKLE_FOLDER, exist_ok=True)

filename = (f'corpus_SemiSup.pickle')
corpus_filename = os.path.join(PICKLE_FOLDER, filename)

# Checks to see if on second stage initializaiton for Flask
if clean and os.environ.get('WERKZEUG_RUN_MAIN') == 'true': # If clean, remove file and remake
    with contextlib.suppress(FileNotFoundError):
        os.remove(corpus_filename)

if clean: # If clean, remove file and remake
    with contextlib.suppress(FileNotFoundError):
        os.remove(corpus_filename)

@ankura.util.pickle_cache(corpus_filename)
def load_initial_data():
    print('***Loading initial data...')

    print('***Splitting labeled/unlabeled and test...')
    # Split to labeled and unlabeled (80-20)
    (train_ids, train_corpus), (test_ids, test_corpus) = ankura.pipeline.train_test_split(corpus, return_ids=True)

    # Must have at least one labeled for each label
    # TODO What does this do?
    # while (len({doc.metadata[GOLD_ATTR_NAME] for doc in train_corpus.documents}) < LABELS_COUNT):
    #     split = ankura.pipeline.train_test_split(corpus, return_ids=True)
    #     (train_ids, train_corpus), (test_ids, test_corpus) = split

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

    return (labels, train_ids, train_corpus, test_ids, test_corpus, starting_labeled_ids)

class UserList:
    """List of user data in memory on the server"""
    def __init__(self, user_base_dir='UserData', timeout=20, zfill=3):
        # Directory to save user info
        self.user_base_dir= os.path.join(PICKLE_FOLDER, user_base_dir)
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
                           for i in range(self.user_id_length)])
        while self.is_duplicate(user_id):
            user_id = ''.join([random.choice(string.ascii_letters)
                               for i in range(self.user_id_length)])
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
                #'Q': Q,
                #'D': D,
                #'anchor_tokens': gs_anchor_tokens.copy(),
                #'anchor_tokens': SELECTED_ANCHOR_TOKENS,
                'update_time': time.time(),
                'update_num': 0,
                'corpus_file': corpus_file,
                #'original_QD_file': QD_file,
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




(labels, train_ids, train_corpus, test_ids, test_corpus, STARTING_LABELED_IDS) = load_initial_data()
del corpus

for doc_id in STARTING_LABELED_IDS:
    doc = train_corpus.documents[doc_id]
    # Assign "user_label" to be the correct label
    doc.metadata[USER_LABEL_ATTR] = doc.metadata[GOLD_ATTR_NAME]
    doc.metadata['Prelabeled'] = True

@app.route('/')
@app.route('/index')
def index():
    return send_from_directory('.','index.html')

@app.route('/index2')
def index2():
    return send_from_directory('.', 'index2.html')

@app.route('/index3')
def index3():
    return send_from_directory('.', 'index3.html')

@app.route('/answers')
def answers():
    return send_from_directory('.', 'answers.html')


def get_lr_acc(user_id):
    from sklearn.linear_model import LogisticRegression
    user = users.get_user_data(user_id)
    if not user:
        return 0
    # if user['lr_acc'] is not None:
    #     return user['lr_acc']
    labeled_docs = user['labeled_docs']
    for doc_id, label in labeled_docs.items():
        train_corpus.documents[doc_id].metadata[USER_LABEL_ATTR] = label

    labeled_ids = set(labeled_docs)

    train_docs = [train_corpus.documents[docid].text for docid in labeled_ids]
    train_targets = [train_corpus.documents[docid].metadata[USER_LABEL_ATTR] for docid in labeled_ids]

    test_docs = [doc.text for doc in test_corpus.documents]
    test_targets = [doc.metadata[USER_LABEL_ATTR] for doc in test_corpus.documents]

    train_docs.extend(test_docs)
    train_targets.extend(test_targets)

    tfv = TfidfVectorizer()
    tfv.fit_transform(train_docs)

    ss = ShuffleSplit(n_splits=5, test_size=len(test_targets), random_state=0)
    lr = LogisticRegression()

    results = cross_validate(lr, train_docs, train_targets, cv=ss, scoring=SCORING_DICT)

    acc = np.mean(results['test_accuracy'])

    user['lr_acc'] = acc

    print('Logistic Regression Accuracy for user', user_id, ':', acc)

    return acc


def get_vw_acc(user_id):
    user = users.get_user_data(user_id)
    if not user:
        return 0

    labeled_docs = user['labeled_docs']
    for doc_id, label in labeled_docs.items():
        train_corpus.documents[doc_id].metadata[USER_LABEL_ATTR] = label

    labeled_ids = set(labeled_docs)

    train_docs = [train_corpus.documents[docid].text for docid in labeled_ids]
    train_targets = [train_corpus.documents[docid].metadata[USER_LABEL_ATTR] for docid in labeled_ids]

    test_docs = [doc.text for doc in test_corpus.documents]
    test_targets = [doc.metadata[USER_LABEL_ATTR] for doc in test_corpus.documents]

    train_docs.extend(test_docs)
    train_targets.extend(test_targets)

    train_targets = [-1 if t == 0 else 1 for t in train_targets]

    tfv = TfidfVectorizer()
    tfv.fit_transform(train_docs)

    ss = ShuffleSplit(n_splits=5, test_size=len(test_targets), random_state=0)
    vw = VWClassifier()

    results = cross_validate(vw, train_docs, train_targets, cv=ss, scoring=SCORING_DICT)

    acc = np.mean(results['test_accuracy'])

    user['vw_acc'] = acc

    return acc


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
#    print('labeled_docs', labeled_docs)
    unlabeled_ids = user['unlabeled_ids']
    Q = user['Q']
    D = user['D']
    anchor_tokens = user['anchor_tokens']
    anchor_vectors = ankura.anchor.tandem_anchors(anchor_tokens, Q,
                                                  train_corpus, epsilon=ta_epsilon)


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
        try:
            d.pop(THETA_ATTR)
        except KeyError:
            pass
        d['metadata']['docNum'] = i
        ret_docs.append(d)
        # print(d['metadata'])

    return jsonify(documents=ret_docs)

def write_to_logfile(text, user, uid):
    if text is not None:
        log_filename = users.get_filename(uid, user['update_num'], '.txt')
        with open(log_filename, 'w') as outfile:
            outfile.write(text)

@app.route('/api/update', methods=['POST'])
def api_update():
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
    user = users.get_user_data(user_id)
    users.save_user(user_id)
    web_unlabeled_ids = user['web_unlabeled_ids']
    labeled_docs = user['labeled_docs']
    unlabeled_ids = user['unlabeled_ids']

    # Write the log file
    write_to_logfile(data.get('log_text'), user, user_id)
    users.print_user_ids()

    # TODO: Get rid of anchors entirely
    #anchor_tokens = SELECTED_ANCHOR_TOKENS

    newly_labeled_docs = data.get('labeled_docs')

    # Label docs onto user
    for doc in newly_labeled_docs:
        labeled_docs[doc['doc_id']] = doc[USER_LABEL_ATTR]
        unlabeled_ids.discard(doc['doc_id'])

    # Label docs into corpus
    for doc_id, label in labeled_docs.items():
        train_corpus.documents[doc_id].metadata[USER_LABEL_ATTR] = label

    # Remove elements without creating new object
    web_unlabeled_ids.clear()
    web_unlabeled_ids.update(random.sample(unlabeled_ids, UNLABELED_COUNT))

    newly_labeled_doc_ids = {doc['doc_id'] for doc in newly_labeled_docs}
    labeled_ids = set(labeled_docs).union(newly_labeled_doc_ids)


    corpus_text = np.asarray([train_corpus.documents[doc_id].text for doc_id in labeled_ids])

    tfidfv = TfidfVectorizer(ngram_range=(ngrams, ngrams))

    start = time.time()
    X = tfidfv.fit_transform(corpus_text)
    print('Features:', tfidfv.get_feature_names()[:10])
    print('***Time - Vectorize:', time.time() - start)

    y = [1 if train_corpus.documents[doc_id].metadata[USER_LABEL_ATTR] == 'R' else -1 for doc_id in labeled_ids]

    vw = VWClassifier(loss_function=lossfn, invert_hash='output.txt')

    start = time.time()
    vw.fit(X, y)
    print('***Time - Train:', time.time() - start)

    # PREPARE TO SEND OBJECTS BACK

    unlabeled_docs = [train_corpus.documents[doc_id].text for doc_id in web_unlabeled_ids]
    u_X = tfidfv.transform(unlabeled_docs)
    inverse = tfidfv.inverse_transform(u_X)

    web_tokens = list(set({ng for arr in inverse for ng in arr}))

    token_vectors = tfidfv.transform(web_tokens)

    predictions = np.abs(vw.decision_function(token_vectors))
    prediction_scaler = MinMaxScaler(feature_range=(0, 1))
    prediction_scaler.fit(np.asarray([predictions]).reshape(-1, 1))

    token_data = list()
    for i in range(len(web_tokens)):
        decision = vw.decision_function(token_vectors[i])[0]

        prob = prediction_scaler.transform([[abs(decision)]])

        token_data.append({'token' : web_tokens[i], 'probs' : np.float32(np.squeeze(prob)), 'decision' : decision})

    token_data.sort(key=lambda d: d['probs'], reverse=True)

    for i, t in enumerate(token_data):
        if i > 20: break
        print('Token:', t['token'], 'Decision:', t['decision'])

    # 1 for R (Republican) and 0 for Democrat (D)
    highlight_dict = {d['token']: 0 if d['decision'] < 0 else 1
                      for d in token_data[:int( len(token_data) * PERCENT_HIGHLIGHT )]}

    def get_highlights(doc):
        highlights = []
        doc_ngrams = np.squeeze(tfidfv.inverse_transform(tfidfv.transform([doc.text])))
        for doc_ngram in doc_ngrams:
            if doc_ngram in highlight_dict:
                highlights.append((f'{doc_ngram}',
                                   labels[highlight_dict[doc_ngram]]))
        return highlights

    unlabeled_docs = []

    new_text = [train_corpus.documents[doc_id].text for doc_id in web_unlabeled_ids]
    predictions = vw.decision_function(tfidfv.transform(new_text))

    for i, doc_id in enumerate(web_unlabeled_ids):
        predict_logprobs = predictions[i]
        i_label = 0 if predict_logprobs < 0 else 1
        predict_label = labels[i_label]

        rdif = prediction_scaler.transform([[abs(predict_logprobs)]])
        con = prediction_scaler.transform([[abs(predict_logprobs)]])

        unlabeled_docs.append(
          {'docId': doc_id,
           'text': new_text[i],
           'tokens': new_text[i].split(),
           'trueLabel': train_corpus.documents[doc_id].metadata[GOLD_ATTR_NAME], # FIXME Needs to be taken out before user study
           'prediction': {
                          'label': predict_label,
                          'relativeDif': rdif[0][0],
                          'confidence': con[0][0]
                          }, # THIS IS WRONG
           'highlights': get_highlights(train_corpus.documents[doc_id])
           })

    labels_dict = {label: i for i, label in enumerate(labels)}
    # A bit of a complex sort, but gets the job done
    doc_sort = lambda doc: (labels_dict[doc['prediction']['label']],
                            (-1)**(labels_dict[doc['prediction']['label']] + 1)
                                * doc['prediction']['relativeDif'])
    unlabeled_docs.sort(key=doc_sort)

    # Calculate average for each label
    # TODO Find a better way of doing this?

    label_count = Counter()
    for doc_id in labeled_ids:
        doc = train_corpus.documents[doc_id]
        label = doc.metadata[USER_LABEL_ATTR]
        label_count[label] += 1

    return_labels = [{'labelId': i, 'label': label, 'count': label_count[label]} for i, label in enumerate(labels)]

#    for d in unlabeled_docs:
#        print(d['prediction'])

    return jsonify(labels=return_labels, unlabeledDocs=unlabeled_docs)

@app.route('/api/accuracy', methods=['POST'])
def api_accuracy():
    data = request.get_json(force=True)

    user_id = data.get('user_id')

    print('Running Accuracy')
    print('User Id:', user_id)
    if user_id is None: return jsonify(accuracy=0.0)

    user = users.get_user_data(user_id)
    users.save_user(user_id)
    labeled_ids = user['labeled_docs']

    print('Retrieving corpus')
    train_docs = [train_corpus.documents[docid].text for docid in labeled_ids]
    train_targets = [train_corpus.documents[docid].metadata[USER_LABEL_ATTR] for docid in labeled_ids]

    test_docs = [doc.text for doc in test_corpus.documents]
    test_targets = [doc.metadata[GOLD_ATTR_NAME] for doc in test_corpus.documents]

    train_targets = [-1 if t == 'R' else 1 for t in train_targets]
    test_targets = [-1 if t == 'R' else 1 for t in test_targets]


    print('Vectorizing')
    tfv = TfidfVectorizer()
    train_vectors = tfv.fit_transform(train_docs)
    test_vectors = tfv.transform(test_docs)

    vw = VWClassifier()

    print('Training')
    start = time.time()
    model = vw.fit(train_vectors, train_targets)
    print('***Time - Get Classifier:', time.time() - start)

    start = time.time()
    predictions = vw.predict(test_vectors)
    print('***Time - Classify:', time.time() - start)

    acc = accuracy_score(test_targets, predictions)

    print('***Accuracy:', acc)
    return jsonify(accuracy=acc)

@app.route('/api/addcorrect', methods=['POST'])
def api_add_correct():
    global Q, D

    data = request.form
    n = int(data.get('n'))
    newly_labeled_doc_ids = set()

    for i in range(n):
        doc_id = unlabeled_ids.pop()
        newly_labeled_doc_ids.add(doc_id)
        labeled_ids.add(doc_id)
        train_corpus.documents[doc_id].metadata[USER_LABEL_ATTR] = (
            train_corpus.documents[doc_id].metadata[GOLD_ATTR_NAME])

    start = time.time()
    Q = ankura.anchor.quick_Q(Q, train_corpus, GOLD_ATTR_NAME, labeled_ids,
                newly_labeled_doc_ids, labels,
                D, label_weight=LABEL_WEIGHT, smoothing=smoothing)
    print('***Time - quick_Q:', time.time()-start)

    print(len(labeled_ids))

    return jsonify(labeled_count=len(labeled_ids))

if __name__ =="__main__":
    app.run(debug=DEBUG, host='0.0.0.0', port=PORT)
