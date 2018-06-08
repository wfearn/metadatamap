import ankura
from flask import Flask, render_template, jsonify, request, send_from_directory
import numpy as np
import json
import time
import argparse
import os
import contextlib
import random

# Init flask app
app = Flask(__name__)

# I think this is unnecessary for what we are doing as there are no cookies
# app.secret_key = '-\xc2\xbe6\xeeL\xd0\xa2\x02\x8a\xee\t\xb7.\xa8b\xf0\xf9\xb8f'

# Attribute names:
# Token topics
Z_ATTR = 'z'

# Document topics
THETA_ATTR = 'theta'

# Number of unlabled docs on the web
UNLABELED_COUNT = 30

# Seed used in the shuffle
SHUFFLE_SEED = None #8448

# Parameters that affect the naming of the pickle (changing these will rename
#  the pickle, generating a new pickle if one of that name doesn't already
#  exist)
num_topics = 20
prelabeled_size = 100
label_weight = 1

# Does NOT change pickle name. Changing these params requires making a clean version (run program
#  and include the -c or --clean argument)
smoothing = 1e-4
epsilon = 1e-5
user_label = 'user_label'

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass
parser=argparse.ArgumentParser(
    description='Used for hosting tbuie with a given dataset',
    epilog=('See https://github.com/byu-aml-lab/tbuie\n' +
            '  and https://github.com/byu-aml-lab/ankura/tree/ankura2/ankura\n' +
            '  for source and dependencies\n \n'),
    formatter_class=CustomFormatter)
parser.add_argument('dataset', metavar='dataset',
                    choices=['yelp', 'tripadvisor', 'amazon'],
                    help='The name of a dataset to use in this instance of tbuie')
parser.add_argument('port', nargs='?', default=5000, type=int,
                    help='Port to be used in hosting the webpage')
parser.add_argument('-c', '--clean', action='store_true')
args=parser.parse_args()

dataset_name = args.dataset
port = args.port
clean = args.clean

if dataset_name == 'newsgroups':
    attr_name = 'coarse_newsgroup'
    corpus = ankura.corpus.newsgroups()
elif dataset_name == 'yelp':
    attr_name = 'binary_rating'
    corpus = ankura.corpus.yelp()
elif dataset_name == 'tripadvisor':
    attr_name = 'label'
    corpus = ankura.corpus.tripadvisor()
elif dataset_name == 'amazon':
    attr_name = 'binary_rating'
    corpus = ankura.corpus.amazon()

# Set seed and shuffle corpus documents if SHUFFLE_SEED
# Was implemented in case we were doing fully semi-supervised; if there is a
#   train/test split, that will shuffle the corpus.
if SHUFFLE_SEED:
    random.seed(SHUFFLE_SEED)
    random.shuffle(corpus.documents)

# Place to save pickle files
folder = 'PickledFiles'
with contextlib.suppress(FileExistsError):
    os.mkdir(folder)

# Naming of this pickle file
filename = (f'SemiSup{dataset_name}_K{num_topics}_prelabeled{prelabeled_size}'
            + f'_lw{label_weight}_ss{SHUFFLE_SEED}.pickle')
full_filename = os.path.join(folder, filename)

# Checks to see if on second stage initializaiton for Flask
if clean and os.environ.get('WERKZEUG_RUN_MAIN') == 'true': # If clean, remove file and remake
    with contextlib.suppress(FileNotFoundError):
        os.remove(full_filename)

@ankura.util.pickle_cache(full_filename)
def load_initial_data():
    print('Loading initial data...')

    print('Splitting labeled/unlabeled and test...')
    # Split to labeled and unlabeled
    split = ankura.pipeline.train_test_split(corpus, return_ids=True)
    (train_ids, train_corpus), (test_ids, test_corpus) = split

    print('Constructing Q...')
    Q, labels = ankura.anchor.build_labeled_cooccurrence(train_corpus, attr_name, labeled_ids,
                                                        label_weight=label_weight, smoothing=smoothing)

    gs_anchor_indices = ankura.anchor.gram_schmidt_anchors(train_corpus, Q,
                                                           k=num_topics, return_indices=True)
    gs_anchor_vectors = Q[gs_anchor_indices]
    gs_anchor_tokens = [[train_corpus.vocabulary[index]] for index in gs_anchor_indices]

    return (Q, labels, train_ids, train_corpus,
            test_ids, test_corpus, gs_anchor_vectors,
            gs_anchor_indices, gs_anchor_tokens)

labeled_ids = set(range(prelabeled_size))

(Q, labels, train_ids, train_corpus,
    test_ids, test_corpus, gs_anchor_vectors,
    gs_anchor_indices, gs_anchor_tokens) = load_initial_data()

for doc_id in labeled_ids:
    doc = train_corpus.documents[doc_id]
    # Assign "user_label" to be the correct label
    doc.metadata[user_label] = doc.metadata[attr_name]

#web_unlabeled_ids = set(range(prelabeled_size + STARTING_UNLABELED))
unlabeled_ids = set(range(prelabeled_size, len(train_corpus.documents)))

@app.route('/')
@app.route('/index')
def index():
    return send_from_directory('.','index.html')

# GET - Send the vocabulary to the client
@app.route('/api/vocab')
def api_vocab():
    return jsonify(vocab=corpus.vocabulary)

@app.route('/api/update', methods=['POST'])
def api_update():
    data = request.get_json()
    print(data)

    # Data is expected to come back in this form:
    # data = {anchor_tokens: [[token_str,..],...]
    #         labeled_docs: [{doc_id: number
    #                         user_label: label},...]
    #         unlabeled_docs: [doc_id,...]
    #        }

    anchor_tokens = data.get('anchor_tokens')
    if anchor_tokens is None:
        anchor_tokens, anchor_vectors = gs_anchor_tokens, gs_anchor_indices
    else:
        anchor_vectors = ankura.anchor.tandem_anchors(anchor_tokens, Q,
                                                      train_corpus, epsilon=epsilon)

    newly_labeled_docs = data.get('labeled_docs')

    # Label docs into corpus
    for doc in newly_labeled_docs:
        train_corpus.documents[doc['doc_id'].metadata[user_label] = label
        labeled_ids.add(doc['doc_id'])

    # Fill the unlabeled docs
    remaining_unlabeled_docs = data.get('unlabeled_docs')
    for i in range(UNLABELED_COUNT - len(remaining_unlabeled_docs)):
        remaining_unlabeled_docs.append(unlabeled_ids.pop())

    # QUICK Q update with newly_labeled_docs
    # TODO need to fix quickQ to take into account the number of documents...
    start = time.time()
    Q = quick_Q(Q, train_corpus, user_label, labeled_ids, newly_labeled_docs,
                label_weight=label_weight, smoothing=smoothing)
    print('***Time - quick_Q:', time.time()-start)

    # Get anchor vectors
    start = time.time()
    anchor_vectors = ankura.anchor.tandem_anchors(anchor_tokens, Q,
                                                  train_corpus, epsilon=1e-15)
    print('***Time - tandem_anchors:', time.time()-start)

    start = time.time()
    C, topics = ankura.anchor.recover_topics(Q, anchor_vectors, epsilon=epsilon, get_c=True)
    print('***Time - recover_topics:', time.time()-start)

    start=time.time()
    # TODO learn about the different assign methods
    # TODO assign only for the labeled and visible unlabeled documents
    #   (depending on current speed)
    # TODO Test different assignments for time and see how assignments differ
    ankura.topic.gensim_assign(train_corpus, topics, theta_attr=THETA_ATTR)
    print('***Time - gensim_assign:', time.time()-start, '-Could be optimized')

    start = time.time()
    topic_summary = ankura.topic.topic_summary(topics[:len(train_corpus.vocabulary)], train_corpus)
    print('***Time - topic_summary:', time.time()-start)

    start = time.time()
    # TODO This will be slower than necessary because we will be recounting all
    # the documents that have a specific label every time. Look into changing
    # how we use the prior_attr_name, or maybe manipulate the corpus outside of
    # this function. More thought needs to be put into this first
    clf = ankura.topic.free_classifier_dream(train_corpus, attr_name=user_label,
                                             labeled_docs=labeled_ids, topics=topics,
                                             C=C, labels=labels)
                                             #prior_attr_name=prior_attr_name)
    print('***Time - Get Classifier:', time.time()-start, '-Could be optimized')

    # PREPARE TO SEND OBJECTS BACK

    # TODO Consider redoing this somehow.... I don't like it.
    # It's kind of a mess....
    start=time.time()
    # Classify, getting probabilities
    left_right = lambda arr: -1 if arr[0]>arr[1] else 1
    relative_dif = lambda arr: abs((arr[0]-arr[1])/((arr[0]+arr[1])/2))
    labeled_relative_dif = lambda arr: left_right(arr) * relative_dif

    unlabeled_docs = [
    for doc_id in remaining_unlabeled_docs:
        doc = train_corpus.documents[doc_id]
        predict_probs = clf(doc, get_probabilities=True)
        predict_label = labels[np.argmax(predict_probs)]
        unlabeled_docs.append(
          {'docNum': doc_id,
           'text': doc.text,
           'tokens': [train_corpus.vocabulary[tok.token] for tok in doc.tokens],
           'trueLabel': doc.metadata[attr_name], # FIXME Probably want to take this out
           'prediction': {'label': predict_label,
                          'relativeDif': relative_dif(predit_probs},
           'anchorIdToValue': {i: val for i, val in enumerate(doc.metadata[THETA_ATTR]},
          })
    print('***Time - Classify:', time.time()-start)

    labels_dict = {label: i for i, label in enumerate(labels)}
    # A bit of a complex sort, but gets the job done
    doc_sort = lambda doc: (labels_dict[doc['prediction']['label']],
                            (-1)**(labels_dict[doc['prediction']['relativeDif']] + 1)
                                * doc['prediction']['relativeDif'])
    unlabeled_docs.sort(key=doc_sort)

    # Calculate average for each label
    labeled_topic_total = defaultdict(lambda: np.zeros(len(anchor_tokens)))
    label_count = Counter()
    for doc_id in labeled_ids:
        doc = train_corpus.documents[doc_id]
        label = doc.metadata[user_label]
        labeled_topic_total[label] += doc.metadata[THETA_ATTR]
        label_count[label] += 1

    labeled_averages = {label: list(labeled_topic_total[label]/label_count[label])
        if label_count[label] else [.5]*len(anchor_tokens) for label in labels}

    return_labels = [
        {'labelId': i,
         'label': label,
         'anchorIdToValue': {i, val for i, val in enumerate(labeled_averages[label]}}
        for i, label in enumerate(labels)]

    return_anchors = [
        {'anchorId': i,
         'anchorWords': anchors,
         'topicWords': topic_summary[i]}
        for i, anchors in enumerate(anchor_tokens)]

    return jsonify(anchors=return_anchors,
                   labels=return_labels,
                   unlabeledDocs=unlabeled_docs)


# Maybe
# POST - Something to do with getting more documents?
# @app.route('', methods=['POST'])

# Maybe
# POST - Something about reshuffling unlabelable documents?
# @app.route('', methods=['POST'])



train_size=500

@app.route('/testDocs')
def testDocs():
    res = getDocsLabelsTopics()
    if len(res) == 3:
        train_corpus, anchor_tokens, labels = res
    else:
        train_corpus, anchor_tokens, labels, topic_summary = res


    topics = [{'topic':t, 'topicNum':i, 'relatedWords':summary} for i, (t, summary) in enumerate(zip(anchor_tokens, topic_summary))]
    topic_count=len(topics)

    # List containing the topical content of each document
    docs = []
    for i, doc in enumerate(train_corpus.documents):
        tmp_dict = dict()
        tmp_dict['docNum'] = i
        tmp_dict['label'] = doc.metadata[attr_name]
        tmp_dict['text'] = doc.text
        tmp_dict['trueLabel'] = doc.metadata[attr_name]

        tmp_dict['tokens'] = [train_corpus.vocabulary[tok.token] for tok in doc.tokens]
        probabilities = doc.metadata[THETA_ATTR]
        for topic, prob in zip(topics, probabilities):
            tmp_dict[topic['topic']] = round(prob*100, 1)
        docs.append(tmp_dict)

    if True:
        docs = [doc for d, doc in enumerate(docs) if d%10==0]

    for doc in docs:
        if doc['label']=='religion':
            print(doc)

    return jsonify(docs=docs, labels=labels, topics=topics)


@ankura.util.pickle_cache(f'{dataset_name}_K{num_topics}_train{train_size}_lw{label_weight}.pickle')
def getDocsLabelsTopics():
    print('getDocsLabelsTopics')
    #train_size = 10000
    test_size = 8000

    #print('Importing corpus...')
    #if dataset == 'amazon':
    #    corpus = ankura.corpus.amazon()
    #if dataset == 'yelp':
    #    corpus = ankura.corpus.yelp()

    total_time_start = time.time()

    print('Splitting training, test sets...')
    split = ankura.pipeline.train_test_split(corpus, num_train=train_size, num_test=test_size, return_ids=True)
    (train_ids, train_corpus), (test_ids, test_corpus) = split

    print('Constructing Q...')
    Q, labels = ankura.anchor.build_labeled_cooccurrence(corpus, attr_name, set(train_ids),
                                                         label_weight, smoothing)

    print('Running GramSchmidt')
    anchor_indices = ankura.anchor.gram_schmidt_anchors(corpus, Q, num_topics,
                                                 return_indices=True)

    anchors = Q[anchor_indices]
    anchor_tokens = [corpus.vocabulary[index] for index in anchor_indices]

    print('Recovering topics...')
    anchor_start = time.time()
    C, topics = ankura.anchor.recover_topics(Q, anchors, get_c=True)
    anchor_end = time.time()

    topic_summary = ankura.topic.topic_summary(topics[:len(corpus.vocabulary)], corpus)

    anchor_time = anchor_end - anchor_start

    ankura.topic.variational_assign(train_corpus, topics)

    print('Retrieving free classifier...')
    classifier = ankura.topic.free_classifier_dream(corpus, attr_name, set(train_ids),
                                                    topics, C, labels)

    print('Calculating base accuracy...')
    contingency = ankura.validate.Contingency()
    for i, doc in enumerate(test_corpus.documents):
        contingency[doc.metadata[attr_name], classifier(doc)] += 1

    total_time_end = time.time()
    total_time = total_time_end - total_time_start
    print('****ACCURACY:', contingency.accuracy())

    return train_corpus, anchor_tokens, labels, topic_summary




@app.route('/dist')
def dist():
    docs, labels, topics = get_random_topical_distributions()
    return jsonify(docs=docs, labels=labels, topics=topics)

def get_random_topical_distributions(doc_count=50):
    #List of possible topics
  #  topics = [
  # 'topic':'Agriculture', 'id':'topic0'
  # 'topic':'Amusement', 'id':'topic1'
  # 'topic':'Buisness', 'id':'topic2'
  # 'topic':'Education', 'id':'topic3'
  # 'topic':'Food', 'id':'topic4'
  # 'topic':'Psychology', 'id':'topic5'
  # 'topic':'Politics', 'id':'topic6'
  # 'topic':'Religion', 'id':'topic7'
  # 'topic':'Sports', 'id':'topic8'
  # 'topic':'Topic Modeling', 'id':'topic9'
  # 'topic':'Wildlife', 'id':'topic10'
  # 'topic':'a', 'id':'topic11'
  # 'topic':'b', 'id':'topic12'
  # 'topic':'c', 'id':'topic13'
  # 'topic':'d', 'id':'topic14'
  # 'topic':'e', 'id':'topic15'
  # 'topic':'f', 'id':'topic16'
  # 'topic':'h' 'id':'topic17'
  # ]

    topics = ['Agriculture', 'Amusement', 'Buisness', 'Education', 'Food',
    'Psychology', 'Politics', 'Religion', 'Sports', 'Topic Modelling',
    'Wildlife', 'a', 'b', 'c', 'd', 'e', 'f', 'h']
    topics = ['Row1', 'Row2', 'Row3', 'Row4',
              'Row5', 'Row6', 'Row7', 'Row8',
              'Row9', 'Row10', 'Row11', 'Row12']
    labels = ['Unlabeled', 'Label1', 'Label2', 'Label3', 'Label4', 'Label5', 'Label6',]

    topics = [{'topic':t, 'topicNum':i} for i, t in enumerate(topics)]

    topic_count=len(topics)

    # List containing the topical content of each document
    docs = []
    for i in range(doc_count):
        tmp_dict = dict()
        tmp_dict['docNum'] = i
        tmp_dict['label'] = np.random.choice(labels)
        #Generate random densities/probabilities (Normed so sum is 1)
        probabilities = np.random.random(topic_count)
        probabilities/=sum(probabilities)

        #Convert to percentages and put in dictionary
        for topic, prob in zip(topics,probabilities):
            tmp_dict[topic['topic']] = round(prob*100, 1)

        #Document dictionary is currently {1:{'topic' : 'probability'}}
        docs.append(tmp_dict)
        print(tmp_dict)

    return docs, labels, topics

# FIXME Needs to be fixed to take into account the number of documents that were used
# to build the original Q. Currently, this will be close but not exact to what
# quickQ *should* do.
# TODO After fixing the above, move this into Ankura2
def quick_Q(Q, corpus, attr_name, labeled_docs, newly_labeled_docs,
                               label_weight=1, smoothing=1e-7):
    V = len(corpus.vocabulary)
    '''This part feels a bit shady to me. I don't think sets and dictionaries
     are gaurenteed to iterate in the same order. Perhaps we should cast the
     set to an ordered list first? Will continue to fiddle with it to see how
     it performs.
     Also, could consider returning and holding onto something from the
     construction of Q that would be the label dictionary or something'''
    label_set = set()
    for d, doc in enumerate(corpus.documents):
        if d in labeled_docs:
            label_set.add(doc.metadata[attr_name])
    #label_set = sorted(list(label_set))
    label_set = {l: V + i for i, l in enumerate(label_set)}
    K = len(label_set)
    Q = Q.copy()
    D = len(corpus.documents) # FIXME this line
    H = np.zeros((V+K, V+K))
    for d in newly_labeled_docs:
        doc = corpus.documents[d]
        n_d = len(doc.tokens)

        # Subtract the unlabeled effect of this document
        norm = 1 / (n_d * (n_d - 1) + 2 * n_d * K * smoothing + K * (K - 1) * smoothing**2)
        for i, w_i in enumerate(doc.tokens):
            for j, w_j in enumerate(doc.tokens):
                if i == j:
                    continue
                H[w_i.token, w_j.token] -= norm
            for j in label_set.values():
                H[w_i.token, j] -= norm * smoothing
                H[j, w_i.token] -= norm * smoothing
        for i in label_set.values():
            for j in label_set.values():
                if i == j:
                    continue
                H[i, j] -= norm * smoothing**2

        # Add the labeled effect of this document
        norm = 1 / ((n_d + label_weight) * (n_d + label_weight - 1))
        index = label_set[doc.metadata[attr_name]]
        print(index, end=' ')
        for i, w_i in enumerate(doc.tokens):
            for j, w_j in enumerate(doc.tokens):
                if i == j:
                    continue
                H[w_i.token, w_j.token] += norm
            H[w_i.token, index] += label_weight * norm
            H[index, w_i.token] += label_weight * norm
        H[index, index] += label_weight * (label_weight - 1) * norm
    Q += (H/D)
    return Q

if __name__ =="__main__":
    app.run(debug=False)

################
# ROUGH PROCESS
################
# Get initial stuff (labeled docs, labels, initial anchor words and topics)

# Get unlabeled documents (Maybe 20 to start and then a few more every time
# after that)

# Recalculate updates to document labels
# -quick Q
#   - Needed for changing from unlabeled to labeled
#   - Will need to change labeled to other labeled? or labeled to unlabeled?

# Recalculate everything for anchor changes (TBUIE)

# Label the rest and see accuracy for whole set

# OUTSIDE - Let Dream return probabilities
# OUTSIDE - Something with number of documents Q construction normalizes for # (D)
#  (needed for quick Q)
# OUTSIDE - Single token documents?
################
################


