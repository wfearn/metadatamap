import ankura
from flask import Flask, render_template, jsonify, request, send_from_directory
import numpy as np
import json
import time
import argparse

app = Flask(__name__)
app.secret_key = '-\xc2\xbe6\xeeL\xd0\xa2\x02\x8a\xee\t\xb7.\xa8b\xf0\xf9\xb8f'

Z_ATTR = 'z'
THETA_ATTR = 'theta'


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
args=parser.parse_args()

dataset_name = args.dataset
port = args.port

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


@app.route('/')
@app.route('/index')
def index():
    return send_from_directory('.','index.html')

#Send the vocabulary to the client
@app.route('/api/vocab')
def api_vocab():
    return jsonify(vocab=corpus.vocabulary)




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
        #Generate random densities/probabilities (Normed so sum is 1)
        probabilities = doc.metadata[THETA_ATTR]
        #print(probabilities)

        #Convert to percentages and put in dictionary
        for topic, prob in zip(topics, probabilities):
            tmp_dict[topic['topic']] = round(prob*100, 1)

        #Document dictionary is currently {1:{'topic' : 'probability'}}
        docs.append(tmp_dict)

    print(docs)
    if True:
        docs = [doc for d, doc in enumerate(docs) if d%10==0]

    for doc in docs:
        if doc['label']=='religion':
            print(doc)

    return jsonify(docs=docs, labels=labels, topics=topics)



num_topics = 20
train_size = 1000
label_weight = 1
dataset = 'yelp'
@ankura.util.pickle_cache(f'{dataset}_K{num_topics}_train{train_size}_lw{label_weight}.pickle')
def getDocsLabelsTopics():
    smoothing = 1e-4
    epsilon = 1e-5
    #train_size = 10000
    test_size = 8000

    print('Importing corpus...')
    if dataset == 'amazon':
        corpus = ankura.corpus.amazon()
    if dataset == 'yelp':
        corpus = ankura.corpus.yelp()

    total_time_start = time.time()

    print('Splitting training, test sets...')
    split = ankura.pipeline.train_test_split(corpus, num_train=train_size, num_test=test_size, return_ids=True)
    (train_ids, train_corpus), (test_ids, test_corpus) = split

    print('Constructing Q...')
    Q, labels = ankura.anchor.build_labeled_cooccurrence(corpus, attr_name, set(train_ids), label_weight, smoothing)

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
    classifier = ankura.topic.free_classifier_dream(corpus, attr_name, train_ids, topics, C, labels)

    print('Calculating base accuracy...')
    contingency = ankura.validate.Contingency()
    for i, doc in enumerate(test_corpus.documents):
        contingency[doc.metadata[attr_name], classifier(doc)] += 1

    total_time_end = time.time()
    total_time = total_time_end - total_time_start
    print('****ACCURACY:', contingency.accuracy())
    time.sleep(1)

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


@app.route('/columnChange')
def change_in_columns():
    print('Successfully routed to columnChange')

    #Print out doc and position in sorted order
    #lambda function sorts by the integer at the end of 'doc###'
    for key in sorted(request.args.keys(),key=lambda x : int(x[3:])):
        print("Doc: {:<10}   pos: {:<10}".format(key, request.args[key]))
    return ('',204)

@app.route('/dots')
def dots():
    return render_template('dots.html')

if __name__ =="__main__":
    app.run(debug=True)
