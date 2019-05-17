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

# Init flask app
app = Flask(__name__)

DEBUG = False

# I think this is unnecessary for what we are doing as there are no cookies
# app.secret_key = '-\xc2\xbe6\xeeL\xd0\xa2\x02\x8a\xee\t\xb7.\xa8b\xf0\xf9\xb8f'

# Attribute names:
# Token topics
Z_ATTR = 'z' # UNUSED

# Document topics
THETA_ATTR = 'theta'

# Prior attr
PRIOR_ATTR = 'lambda' # UNUSED

# Number of unlabled docs on the web
UNLABELED_COUNT = 10

# Seed used in the shuffle
SHUFFLE_SEED = None #8448

# Number of labels in our data
LABELS_COUNT = 2

# Param for harmonic mean (In tandem_anchors)
ta_epsilon = 1e-15

# Epsilon for recover topics
rt_epsilon = 1e-5

# Name of the user_label (for metadata on each document)
USER_LABEL_ATTR = 'user_label'

# Parameters that affect the naming of the pickle (changing these will rename
#  the pickle, generating a new pickle if one of that name doesn't already
#  exist)
NUM_TOPICS = 20
PRELABELED_SIZE = 200
LABEL_WEIGHT = 1
USER_ID_LENGTH = 5

# Does NOT change pickle name.
# Changing these params requires making a clean version
# (run program and include the -c or --clean argument)
smoothing = 1e-4


# if PRELABELED_SIZE < LABELS_COUNT:
#     raise ValueError("prelabled_size cannot be less than LABELS_COUNT")

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
    return parser.parse_args()


args = parse_args()
DATASET_NAME = args.dataset
PORT = args.port
clean = args.clean


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

# Naming of this pickle file
# filename = (f'SemiSup_{DATASET_NAME}_K{num_topics}_prelabeled{prelabeled_size}'
#             + f'_lw{LABEL_WEIGHT}_ss{SHUFFLE_SEED}.pickle')

subfolder = f'K{NUM_TOPICS}_prelabeled{PRELABELED_SIZE}_lw{LABEL_WEIGHT}'
PICKLE_FOLDER = os.path.join(PICKLE_FOLDER_BASE, DATASET_NAME, subfolder)
os.makedirs(PICKLE_FOLDER, exist_ok=True)


filename = (f'QD_SemiSup.pickle')
QD_filename = os.path.join(PICKLE_FOLDER, filename)

filename = (f'corpus_SemiSup.pickle')
corpus_filename = os.path.join(PICKLE_FOLDER, filename)

# Checks to see if on second stage initializaiton for Flask
if clean and os.environ.get('WERKZEUG_RUN_MAIN') == 'true': # If clean, remove file and remake
    with contextlib.suppress(FileNotFoundError):
        os.remove(corpus_filename)
        os.remove(QD_filename)
if clean: # If clean, remove file and remake
    with contextlib.suppress(FileNotFoundError):
        os.remove(corpus_filename)
        os.remove(QD_filename)

@ankura.util.multi_pickle_cache(QD_filename, corpus_filename)
def load_initial_data():
    print('***Loading initial data...')

    print('***Splitting labeled/unlabeled and test...')
    # Split to labeled and unlabeled (80-20)
    split = ankura.pipeline.train_test_split(corpus, return_ids=True)
    (train_ids, train_corpus), (test_ids, test_corpus) = split

    # Must have at least one labeled for each label
    # TODO
    # while (len({doc.metadata[GOLD_ATTR_NAME] for doc in train_corpus.documents}) < LABELS_COUNT):
    #     split = ankura.pipeline.train_test_split(corpus, return_ids=True)
    #     (train_ids, train_corpus), (test_ids, test_corpus) = split

    starting_labeled_labels = set()
    all_label_set = set(LABELS)
    while starting_labeled_labels != all_label_set:
        starting_labeled_ids = set(random.sample(range(len(train_corpus.documents)), PRELABELED_SIZE))
        starting_labeled_labels = set(train_corpus.documents[i].metadata[GOLD_ATTR_NAME] for i in starting_labeled_ids)
        # TODO
        break

    print('***Constructing Q...')
    Q, labels, D = ankura.anchor.build_labeled_cooccurrence(train_corpus,
                                                        GOLD_ATTR_NAME,
                                                        starting_labeled_ids,
                                                        label_weight=LABEL_WEIGHT,
                                                        smoothing=smoothing,
                                                        get_d=True,
                                                        labels=LABELS)

    gs_anchor_indices = ankura.anchor.gram_schmidt_anchors(train_corpus, Q,
                                                           k=NUM_TOPICS, return_indices=True)
    gs_anchor_vectors = Q[gs_anchor_indices]
    gs_anchor_tokens = [[train_corpus.vocabulary[index]] for index in gs_anchor_indices]

    return (Q, D), (labels, train_ids, train_corpus, test_ids, test_corpus,
                    gs_anchor_vectors, gs_anchor_indices, gs_anchor_tokens,
                    starting_labeled_ids)


class UserList:
    """List of user data in memory on the server"""
    def __init__(self, user_base_dir='UserData', timeout=20, zfill=3):
        # Directory to save user info
        self.user_base_dir= os.path.join(PICKLE_FOLDER, user_base_dir)
        os.makedirs(self.user_base_dir, exist_ok=True)
        self.zfill = zfill

        # Timeout in seconds
        self.timeout = 20*60

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

    def add_user(self, corpus_file, QD_file):

        user_id = self.generate_user_id()

        # Make the directory
        user_dir = self.get_user_dir(user_id)
        os.makedirs(user_dir, exist_ok=True)

        # Load Q and D
        with open(QD_file, 'rb') as infile:
            Q, D = pickle.load(infile)


        # Set up labeling
        web_unlabeled_ids = set()
        unlabeled_ids = set(range(len(train_corpus.documents)))
        unlabeled_ids.difference_update(STARTING_LABELED_IDS)
        labeled_docs = {i: train_corpus.documents[i].metadata[GOLD_ATTR_NAME]
                        for i in STARTING_LABELED_IDS}

        user = {'user_id': user_id,
                'labeled_docs': labeled_docs,
                'web_unlabeled_ids': set(),
                'unlabeled_ids': unlabeled_ids,
                'Q': Q,
                'D': D,
                'anchor_tokens': gs_anchor_tokens.copy(),
                'update_time': time.time(),
                'update_num': 0,
                'corpus_file': corpus_file,
                'original_QD_file': QD_file,
                'user_dir': user_dir
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
        print(self.get_user_dir(user_id))
        updates = [filename for filename in os.listdir(self.get_user_dir(user_id))
                   if user_id in filename]
        last_update = sorted(updates)[-1]
        time.sleep(5)
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
            self.users.pop(user_id)

    def get_user_data(self, user_id):
        if user_id not in self.users:
            self.load_last_update(user_id)
        return self.users[user_id]

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




(Q, D), (labels, train_ids, train_corpus, test_ids, test_corpus,
         gs_anchor_vectors, gs_anchor_indices, gs_anchor_tokens,
         STARTING_LABELED_IDS) = load_initial_data()

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


# GET - Send the vocabulary to the client
@app.route('/api/vocab')
def api_vocab():
    return jsonify(vocab=train_corpus.vocabulary)

users = UserList()

@app.route('/api/adduser', methods=['POST'])
def api_adduser():
    user_id = users.add_user(corpus_filename, QD_filename)
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

@app.route('/api/update', methods=['POST'])
def api_update():
    data = request.get_json()
#    print(data)

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
#    print('labeled_docs', labeled_docs)
    unlabeled_ids = user['unlabeled_ids']
    Q = user['Q']
    D = user['D']

    # Write the log file
    log_text = data.get('log_text')
    if log_text is not None:
        log_filename = users.get_filename(user_id, user['update_num'], '.txt')
        with open(log_filename, 'w') as outfile:
            outfile.write(log_text)

    users.print_user_ids()

    anchor_tokens = [t for t in data.get('anchor_tokens') if t]

    if not anchor_tokens:
        anchor_tokens = user['anchor_tokens']
    if not anchor_tokens:
        anchor_tokens = gs_anchor_tokens

    print(anchor_tokens)
    anchor_vectors = ankura.anchor.tandem_anchors(anchor_tokens, Q,
                                                  train_corpus, epsilon=ta_epsilon)
    user['anchor_tokens'] = anchor_tokens

    newly_labeled_docs = data.get('labeled_docs')

    # Label docs onto user
    for doc in newly_labeled_docs:
        print(doc['doc_id'])
        labeled_docs[doc['doc_id']] = doc[USER_LABEL_ATTR]
        unlabeled_ids.discard(doc['doc_id'])

    # Label docs into corpus
    # FIXME Really probably shouldn't relabel the corpus like this.
    # It doesn't increase big-O, but it does seem a bit odd
    # Would need to make Ankura changes to change this
    for doc_id, label in labeled_docs.items():
        train_corpus.documents[doc_id].metadata[USER_LABEL_ATTR] = label

    # Replace the unlabeled docs

    # Remove elements without creating new object
    web_unlabeled_ids.clear()

    web_unlabeled_ids.update(random.sample(unlabeled_ids, UNLABELED_COUNT))

    newly_labeled_doc_ids = {doc['doc_id'] for doc in newly_labeled_docs}

    labeled_ids = set(labeled_docs)

    # QUICK Q update with newly_labeled_docs
    if newly_labeled_doc_ids:
        start = time.time()
        Q = ankura.anchor.quick_Q(Q, train_corpus, USER_LABEL_ATTR, labeled_ids,
                                  newly_labeled_doc_ids, labels,
                                  D, label_weight=LABEL_WEIGHT, smoothing=smoothing)
        print('***Time - quick_Q:', time.time()-start)
    user['Q'] = Q

    # Get anchor vectors
    start = time.time()
    anchor_vectors = ankura.anchor.tandem_anchors(anchor_tokens, Q,
                                                  train_corpus, epsilon=ta_epsilon)
    print('***Time - tandem_anchors:', time.time()-start)

    # TODO OPTIMIZE Look into using the parallelism keyword or some other way
    #   to make this faster, as it is currently the longest thing by a long shot.
    start = time.time()
    C, topics = ankura.anchor.recover_topics(Q, anchor_vectors, epsilon=rt_epsilon, get_c=True)
    print('***Time - recover_topics:', time.time()-start)

    start=time.time()
    ankura.topic.gensim_assign(train_corpus, topics, theta_attr=THETA_ATTR,
                               needs_assign=(web_unlabeled_ids | labeled_ids))
    print('***Time - gensim_assign:', time.time()-start, '-Could be optimized')

    start = time.time()
    topic_summary = ankura.topic.topic_summary(topics[:len(train_corpus.vocabulary)], train_corpus)
    print('***Time - topic_summary:', time.time()-start)

    start = time.time()
    # OPTIMIZE This will be slower than necessary because we will be recounting all
    # the documents that have a specific label every time. Look into changing
    # how we use the prior_attr_name, or maybe manipulate the corpus outside of
    # this function. More thought needs to be put into this first
    clf = ankura.topic.free_classifier_dream(train_corpus, attr_name=USER_LABEL_ATTR,
                                             labeled_docs=labeled_ids, topics=topics,
                                             C=C, labels=labels)
                                             #prior_attr_name=PRIOR_ATTR)
    print('***Time - Get Classifier:', time.time()-start, '-Could be optimized')

    # PREPARE TO SEND OBJECTS BACK

    # TODO Consider redoing this somehow.... I don't like it.
    # It's kind of a mess....
    start=time.time()
    # Classify, getting probabilities
    left_right = lambda arr: -1 if arr[0]>arr[1] else 1
    relative_dif = lambda arr: abs((arr[0]-arr[1])/((arr[0]+arr[1])/2))
    labeled_relative_dif = lambda arr: left_right(arr) * relative_dif

    # FIXME Sometimes arr is [0,0]... This is because the log probabilities are
    # just too small to convert back to regular prob space in the free
    # classifier. I haven't nailed down exactly what leads to this thing, but
    # we may imagine anything could do this as you are just multiplying a bunch
    # of numbers in [0,1], with most of them tending toward 0 anyway.
    def relative_dif(arr, i):
        if arr[0]==0 and arr[1]==0:
            return 0
        #return abs((arr[0]-arr[1])/((arr[0]+arr[1])/2))
        return abs((arr[i])/(arr.sum()))

    # probs = 0
    # fine = 0
    # for doc in train_corpus.documents:
    #     arr = clf(doc, get_probabilities=True)
    #     if arr[0]==0 and arr[1]==0:
    #         probs += 1
    #     else:
    #         fine += 1
    # print(probs)
    # print(fine)
    # sys.exit()

    def get_highlights(doc):
        base = clf(doc, get_log_probabilities=True)
        label = base.argmax()
        ranking = []
        highlights = []
        for i in range(len(doc.tokens)):
            loc = doc.tokens[i].loc
            new_doc = ankura.pipeline.Document('',
                        doc.tokens[:i]+doc.tokens[i+1:], {})
            probs = clf(new_doc, get_log_probabilities=True)
            new_label = probs.argmax()
            if new_label != label:
                highlights.append((loc, labels[new_label]))
            else:
                ratio = probs/base
                l = ratio.argmin()
                lab = labels[l]
                x = ratio[l]
                ranking.append((x, loc, lab))
        highlights += [(loc, lab) for x, loc, lab in
                    sorted(ranking)[:int(len(doc.tokens)*.5)]]
        return highlights

    web_tokens = {tok.token for doc in
                  (train_corpus.documents[doc_id] for doc_id in web_unlabeled_ids)
                  for tok in doc.tokens}

    tok_docs  = (ankura.pipeline.Document('', [ankura.pipeline.TokenLoc(t, ())], {})
                 for t in web_tokens)

    tok_data = [{'token': d.tokens[0].token,
                 'probs': clf(d, get_probabilities=True)} for d in tok_docs]

    for data in tok_data:
        p = data['probs']
        data['weight'] = p.max()/p.sum()

    tok_data.sort(key=lambda d: d['weight'], reverse=True)

    highlight_dict = {d['token']: d['probs'].argmax()
                      for d in tok_data[:int(len(tok_data)*.2)]}


    def get_highlights2(doc):
        highlights = []
        for tok_loc in doc.tokens:
            if tok_loc.token in highlight_dict:
                highlights.append((tok_loc.loc,
                                   labels[highlight_dict[tok_loc.token]]))
        return highlights

    unlabeled_docs = []

    for doc_id in web_unlabeled_ids:
        doc = train_corpus.documents[doc_id]
        predict_logprobs = clf(doc, get_log_probabilities=True)
        i_label = np.argmax(predict_logprobs)
        predict_label = labels[i_label]

        predict_probs = normalize_logprobs(predict_logprobs)
        # predict_probs = np.exp(predict_logprobs)
        # print(predict_probs/predict_probs.sum())
        unlabeled_docs.append(
          {'docId': doc_id,
           'text': doc.text,
           'tokens': [train_corpus.vocabulary[tok.token] for tok in doc.tokens],
           'trueLabel': doc.metadata[GOLD_ATTR_NAME], # FIXME Needs to be taken out
                                                      # before user study
           'prediction': {'label': predict_label,
                          'relativeDif': relative_dif(predict_probs, i_label),
                          'confidence': predict_probs[i_label]
                          }, # THIS IS WRONG
           'anchorIdToValue': {i: float(val)
                               for i, val in enumerate(doc.metadata[THETA_ATTR])},
           #'highlight': [list(tok.loc) for tok in doc.tokens],
           'highlights': get_highlights2(doc)
           })
    print('***Time - Classify:', time.time()-start)

    labels_dict = {label: i for i, label in enumerate(labels)}
    # A bit of a complex sort, but gets the job done
    doc_sort = lambda doc: (labels_dict[doc['prediction']['label']],
                            (-1)**(labels_dict[doc['prediction']['label']] + 1)
                                * doc['prediction']['relativeDif'])
    unlabeled_docs.sort(key=doc_sort)
    # with open('vals.txt', 'a') as outfile:
    #     outfile.writelines(f'{p[0]} {p[1]}\n' for p in ps)

    # for p in ps:
    #     print(p)
    #     print(np.exp(p) / np.exp(p).sum())
    #     print(normalize_logprobs(p))
    #     print()


    # Calculate average for each label
    # TODO Find a better way of doing this?
    labeled_topic_total = defaultdict(lambda: np.zeros(len(anchor_tokens)))
    label_count = Counter()
    for doc_id in labeled_ids:
        doc = train_corpus.documents[doc_id]
        label = doc.metadata[USER_LABEL_ATTR]
        labeled_topic_total[label] += doc.metadata[THETA_ATTR]
        label_count[label] += 1

    labeled_averages = {label: list(labeled_topic_total[label]/label_count[label])
        if label_count[label] else [.5]*len(anchor_tokens) for label in labels}

    return_labels = [
        {'labelId': i,
         'label': label,
         'anchorIdToValue': {i: val for i, val in enumerate(labeled_averages[label])},
         'count': label_count[label]}
        for i, label in enumerate(labels)]

    return_anchors = [
        {'anchorId': i,
         'anchorWords': anchors,
         'topicWords': topic_summary[i]}
        for i, anchors in enumerate(anchor_tokens)]

    for d in unlabeled_docs:
        print(d['prediction'])

    return jsonify(anchors=return_anchors,
                   labels=return_labels,
                   unlabeledDocs=unlabeled_docs)

# Maybe
# POST - Something to do with getting more documents?
# @app.route('', methods=['POST'])

# Maybe
# POST - Something about reshuffling unlabelable documents?
# @app.route('', methods=['POST'])

def normalize_logprobs(probs):
    a = max(probs)
    probs = np.exp(probs - a)
    return probs/probs.sum()

@app.route('/api/accuracy', methods=['POST'])
def api_accuracy():
    data = request.get_json(force=True)
    anchor_tokens = data.get('anchor_tokens')
    if not anchor_tokens:
        anchor_tokens, anchor_vectors = gs_anchor_tokens, gs_anchor_vectors
    else:
        anchor_vectors = ankura.anchor.tandem_anchors(anchor_tokens, Q,
                                                     train_corpus, epsilon=ta_epsilon)

    start = time.time()
    C, topics = ankura.anchor.recover_topics(Q, anchor_vectors, epsilon=rt_epsilon, get_c=True)
    print('***Time - recover_topics:', time.time()-start)

    # Gensim assignment
    start=time.time()
    ankura.topic.gensim_assign(test_corpus, topics, theta_attr=THETA_ATTR)
    print('***Time - gensim_assign:', time.time()-start, '-Could be optimized')

    start = time.time()
    topic_summary = ankura.topic.topic_summary(topics[:len(train_corpus.vocabulary)], train_corpus)
    print('***Time - topic_summary:', time.time()-start)

    start = time.time()
    # OPTIMIZE This will be slower than necessary because we will be recounting all
    # the documents that have a specific label every time. Look into changing
    # how we use the prior_attr_name, or maybe manipulate the corpus outside of
    # this function. More thought needs to be put into this first
    clf = ankura.topic.free_classifier_dream(train_corpus,
                                             attr_name=USER_LABEL_ATTR,
                                             labeled_docs=labeled_ids, topics=topics,
                                             C=C, labels=labels)
                                             #prior_attr_name=PRIOR_ATTR)
    print('***Time - Get Classifier:', time.time()-start, '-Could be optimized')


    contingency = ankura.validate.Contingency()

    start=time.time()
    for doc in test_corpus.documents:
        gold = doc.metadata[GOLD_ATTR_NAME]
        pred = clf(doc)
        contingency[gold, pred] += 1
    print('***Time - Classify:', time.time()-start)
    print('***Accuracy:', contingency.accuracy(),
          f'on {len(test_corpus.documents)} test documents')
    return jsonify(accuracy=contingency.accuracy())

# with open('BESTAMZ.txt', 'r') as infile:
#     AMZ_BEST = [line.strip().split(' ') for line in infile.readlines()]

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

# @app.errorhandler(404)
# def page_not_found(e):
#     # Rickrolled
#     rollers = [
#                'https://en.wikipedia.org/wiki/Rick_Astley',
#                'https://youtu.be/dQw4w9WgXcQ?t=43',
#                'https://youtu.be/lXMskKTw3Bc?t=24'
#               ]
#     roll = random.choice(rollers)
#     return redirect(roll)

if __name__ =="__main__":
    app.run(debug=DEBUG, host='0.0.0.0', port=PORT)
