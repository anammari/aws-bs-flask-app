# core dependencies
from gensim.test.utils import datapath
from gensim import models
import spacy
from flair.models import TARSClassifier
from flair.data import Sentence
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Batch, Filter, FieldCondition, Range, MatchValue
import nltk
from sklearn.feature_extraction import text
import re
from sentence_transformers import SentenceTransformer, util

# utility dependencies
import os
import numpy as np
import pandas as pd
import pickle
import joblib
import random

# REST dependencies
import json
import flask

# global parameters
temp_file = datapath(os.path.abspath('models/lda_model'))
dict_file = 'models/topic_names.pkl'
pipe_file = 'models/sklearn_models.joblib'
trent_pipe_file = 'models/trent_sklearn_models.joblib'
bert_model_path = 'models/bert'
doc_top_dist_df = None
onto_path = 'data/'
collection_name = "my_collection"
vector_dim = 384
sim_threshold = 0.5

# Load the SpaCy model
nlp = spacy.load(os.path.abspath('en_core_web_sm-3.3.0'))

# Load the topic model
model = models.ldamodel.LdaModel.load(temp_file)
with open(dict_file, 'rb') as f:
        topic_names = pickle.load(f)

# Load the gender model
modlist = joblib.load(pipe_file)
count_vect, tfidf_transformer, text_clf3, gender_tokens = modlist[0], modlist[1], modlist[2], modlist[3]

# Load the behaviour frequency model
f_modlist = joblib.load(trent_pipe_file)
f_count_vect, f_tfidf_transformer, f_text_clf3 = f_modlist[0], f_modlist[1], f_modlist[2]

# Load the TARS model
tagger = TARSClassifier.load('tars-base')

# Load the Sentence Transformer model
st_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

# Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)

# The flask app for serving predictions
application = flask.Flask(__name__)
@application.route('/ping', methods=['GET'])
def ping():
    # Check if all the models were loaded correctly
    try:
        model
        topic_names
        nlp
        count_vect
        tfidf_transformer
        text_clf3
        gender_tokens
        f_count_vect
        f_tfidf_transformer
        f_text_clf3
        tagger
        st_model
        status = 200
        result = json.dumps({'status': 'OK'})
    except:
        status = 400
        result = json.dumps({'status': 'ERROR'})
    return flask.Response(response=result, status=status, mimetype='application/json' )

@application.route('/topics', methods=['POST'])
def get_topics():
    # Get input JSON data and convert it to a DF
    input_json = flask.request.get_json()
    input_json = json.dumps(input_json['input'])
    input_df = pd.read_json(input_json,orient='list')

    # Text preprocessing
    documents = input_df['text'].tolist()
    texts = []
    for document in documents:
        text = []
        doc = nlp(document)
        person_tokens = []
        for w in doc:
            if w.ent_type_ == 'PERSON':
                person_tokens.append(w.lemma_)
        for w in doc:
            if not w.is_stop and not w.is_punct and not w.like_num and not len(w.text.strip()) == 0 and not w.lemma_ in person_tokens:
                text.append(w.lemma_.lower())
        texts.append(text)
    
    # Produce the predictions
    predictions = []
    for i in range(0, len(documents)):
        tp_lst = [] #topic list
        sc_lst = [] #score list
        bow_doc = model.id2word.doc2bow(texts[i])
        lda_bow_doc = model[bow_doc]
        for tp_tup in lda_bow_doc:
            tp_lst.append(tp_tup[0])
            sc_lst.append(tp_tup[1])
        doc_top_dist_df = pd.DataFrame({'Topic': tp_lst, 'Probability': sc_lst})
        doc_top_dist_df['Score'] = np.where(doc_top_dist_df['Probability']>= 0.5, 1.0, 0.0)
        doc_top_dist_df.sort_values(by='Probability', ascending=False, inplace=True)
        doc_top_dist_df = doc_top_dist_df.replace({"Topic": topic_names})
        doc_top_score_df = doc_top_dist_df[['Topic', 'Score']]
        predictions.append(doc_top_score_df)

    # Transform predictions to JSON
    result = {'output': []}
    list_out = []
    for output_df in predictions:
        row_format = pd.Series(output_df.Score.values,index=output_df.Topic).to_dict()
        list_out.append(row_format)
    result['output'] = list_out
    result = json.dumps(result)
    return flask.Response(response=result, status=200, mimetype='application/json')

@application.route('/age', methods=['POST'])
def get_age():
    # Get input JSON data and convert it to a DF
    input_json = flask.request.get_json()
    input_json = json.dumps(input_json['input'])
    input_df = pd.read_json(input_json,orient='list')

    # Detect age in the text
    documents = input_df['text'].tolist()
    predictions = []
    for line in documents:
        ents = []
        doc = nlp(line)
        ent_d = {}
        for ent in doc.ents:
            if ent.label_ == 'DATE':
                ent_d['word'] = ent.text
                ent_d['entity'] = ent.label_
                ent_d['start'] = ent.start_char
                ent_d['end'] = ent.end_char
                ents.append(ent_d)
        if len(ents) > 0:
            row_format = {'score': 1.0, 'value': ents[0]['word']}
        else:
            row_format = {'score': 0.0, 'value': 'unknown'}
        predictions.append(row_format)

    # Transform predictions to JSON
    result = {'output': []}
    result['output'] = predictions
    result = json.dumps(result)
    return flask.Response(response=result, status=200, mimetype='application/json')

@application.route('/gender', methods=['POST'])
def get_gender():
    # Get input JSON data and convert it to a DF
    input_json = flask.request.get_json()
    input_json = json.dumps(input_json['input'])
    input_df = pd.read_json(input_json,orient='list')

    # Detect gender in the text
    def enrich_text(x):
        dups = 5
        words = []
        for y in x.split(" "):
            if y in gender_tokens:
                yyy = [y] * dups
            else:
                yyy = [y]
            words += yyy
        random.shuffle(words)
        user_input = ' '.join(words)
        return user_input
    input_df['text_enriched'] = input_df['text'].apply(lambda x: enrich_text(x))
    documents = input_df['text_enriched'].tolist()
    g_dict = {0: 'female', 1: 'male', 2: 'unknown'}
    X_test_counts = count_vect.transform(documents)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    pred = text_clf3.predict(X_test_tfidf)
    scores = [0.0 if x == 2 else 1.0 for x in pred]
    values = [g_dict[x] for x in pred]
    predictions = pd.DataFrame({'score': scores, 'value': values})
    
    # Transform predictions to JSON
    result = {'output': []}
    list_out = predictions.to_dict(orient="records")
    result['output'] = list_out
    result = json.dumps(result)
    return flask.Response(response=result, status=200, mimetype='application/json')

@application.route('/frequency', methods=['POST'])
def get_frequency():
    # Get input JSON data and convert it to a DF
    input_json = flask.request.get_json()
    input_json = json.dumps(input_json['input'])
    input_df = pd.read_json(input_json,orient='list')

    # Detect frequency in the text
    documents = input_df['text'].tolist()

    X_test_counts = f_count_vect.transform(documents)
    X_test_tfidf = f_tfidf_transformer.transform(X_test_counts)
    pred = f_text_clf3.predict(X_test_tfidf)
    f_dict = {0: 'No Frequency', 1: 'Frequency Exists'}
    scores = [0.0 if x == 0 else 1.0 for x in pred]
    values = [f_dict[x] for x in pred]
    predictions = pd.DataFrame({'score': scores, 'status': values})
    
    # Transform predictions to JSON
    result = {'output': []}
    list_out = predictions.to_dict(orient="records")
    result['output'] = list_out
    result = json.dumps(result)
    return flask.Response(response=result, status=200, mimetype='application/json')

@application.route('/topics_p2_q1', methods=['POST'])
def get_topics_p2_q1():
    # Initialize variables
    classes = ['speaking',
                'interview',
                'medical',
                'observation',
                'information'
                ]
    classes_dict = {'speaking': 'Speaking with Family',
                    'interview': 'Interview with Stakeholders',
                    'medical': 'File Reviews',
                    'observation': 'Observing the Person with Disability',
                   }
    
    # Get input JSON data and convert it to a DF
    input_json = flask.request.get_json()
    input_json = json.dumps(input_json['input'])
    input_df = pd.read_json(input_json,orient='list')

    # Detect topics in the text
    documents = input_df['text'].tolist()
    document = documents[0]    # Currently this endpoint expects a single text input
    s = Sentence(document)
    tagger.predict_zero_shot(s, classes)
    err_msg = 'No Topic Found'
    err_score = '0%'
    p_classes = []
    score_dict = s.to_dict()
    all_labels = score_dict['all labels']
    p_classes = [x['value'] for x in all_labels]
    if len(all_labels) == 0 or 'information' in p_classes:
        p_classes = [err_msg]
        str_p_scores = [err_score]
    elif len(all_labels) > 0:
        p_classes = [classes_dict[x['value']] for x in all_labels]
        p_scorees = [round(x['confidence']*100.0, 2) for x in all_labels]
        str_p_scores = [str(p_scorees[i])+'%' for i in range(0, len(p_scorees))]
    else:
        pass
    predictions = pd.DataFrame({'score': str_p_scores, 'topic': p_classes})
    
    # Transform predictions to JSON
    result = {'output': []}
    list_out = predictions.to_dict(orient="records")
    result['output'] = list_out
    result = json.dumps(result)
    return flask.Response(response=result, status=200, mimetype='application/json')

@application.route('/topics_p2_q2', methods=['POST'])
def get_topics_p2_q2():
    # Initialize variables
    classes = ['speech',
                'psychiatric',
                'medical',
                'not']
    classes_dict = {'speech': 'Speech / language assessment',
                    'psychiatric': 'Psychiatric assessment',
                    'medical': 'Medical assessment',
                   }
    
    # Get input JSON data and convert it to a DF
    input_json = flask.request.get_json()
    input_json = json.dumps(input_json['input'])
    input_df = pd.read_json(input_json,orient='list')

    # Detect topics in the text
    documents = input_df['text'].tolist()
    document = documents[0]    # Currently this endpoint expects a single text input
    s = Sentence(document)
    tagger.predict_zero_shot(s, classes)
    err_msg = 'No Topic Found'
    err_score = '0%'
    p_classes = []
    score_dict = s.to_dict()
    all_labels = score_dict['all labels']
    p_classes = [x['value'] for x in all_labels]
    if len(all_labels) == 0 or 'not' in p_classes:
        p_classes = [err_msg]
        str_p_scores = [err_score]
    elif len(all_labels) > 0:
        p_classes = [classes_dict[x['value']] for x in all_labels]
        p_scorees = [round(x['confidence']*100.0, 2) for x in all_labels]
        str_p_scores = [str(p_scorees[i])+'%' for i in range(0, len(p_scorees))]
    else:
        pass
    predictions = pd.DataFrame({'score': str_p_scores, 'topic': p_classes})
    
    # Transform predictions to JSON
    result = {'output': []}
    list_out = predictions.to_dict(orient="records")
    result['output'] = list_out
    result = json.dumps(result)
    return flask.Response(response=result, status=200, mimetype='application/json')

@application.route('/topics_p3', methods=['POST'])
def get_topics_p3():
    # Get input JSON data and convert it to a DF
    input_json = flask.request.get_json()
    input_json = json.dumps(input_json['input'])
    input_df = pd.read_json(input_json,orient='list')

    # Get the query parameter value corresponsing to the question number
    q_no = flask.request.args.get("qno")

    # Question - Ontology mapping
    q_onto_dict = {'q1': "behaviours", 
                   'q2': "functional_hypothesis",
                   'q3': "replacement_behaviour"
                  }

    # Detect topics in the text
    documents = input_df['text'].tolist()
    document = documents[0]    # Currently this endpoint expects a single text input

    # Read ontology
    bhvr_onto_df = pd.read_csv(onto_path+f"p3_{q_no}.csv", header=None, encoding="utf-8")
    bhvr_onto_df.columns = ['onto']
    bhvr_onto_lst = bhvr_onto_df['onto'].tolist()
    
    # Text preprocessing
    sw_lst = text.ENGLISH_STOP_WORDS
    def preprocess(onto_lst):
        cleaned_onto_lst = []
        pattern = re.compile(r'^[a-z ]*$')
        for document in onto_lst:
            text = []
            doc = nlp(document)
            person_tokens = []
            for w in doc:
                if w.ent_type_ == 'PERSON':
                    person_tokens.append(w.lemma_)
            for w in doc:
                if not w.is_stop and not w.is_punct and not w.like_num and not len(w.text.strip()) == 0 and not w.lemma_ in person_tokens:
                    text.append(w.lemma_.lower())
            texts = [t for t in text if len(t) > 1 and pattern.search(t) is not None and t not in sw_lst]
            cleaned_onto_lst.append(" ".join(texts))
        return cleaned_onto_lst

    # Compute document embeddings
    def sentence_embeddings(cl_onto_lst):
        emb_onto_lst_temp = st_model.encode(cl_onto_lst)
        emb_onto_lst = [x.tolist() for x in emb_onto_lst_temp]
        return emb_onto_lst

    # Add to qdrant collection
    def add_to_collection(cl_bhvr_onto_lst, emb_bhvr_onto_lst):
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )
        doc_count = len(emb_bhvr_onto_lst)
        ids = list(range(1, doc_count+1))
        payloads = [{"ontology": q_onto_dict[q_no], "phrase": x} for x in cl_bhvr_onto_lst]
        vectors = emb_bhvr_onto_lst
        client.upsert(
            collection_name=f"{collection_name}",
            points=Batch(
                ids=ids,
                payloads=payloads,
                vectors=vectors
            ),
        )

    # Count collection
    def count_collection():
        return len(client.scroll(
                collection_name=f"{collection_name}"
            )[0])

    # Verb phrase extraction
    def extract_vbs(data_chunked):
        for tup in data_chunked:
            if len(tup) > 2:
                yield(str(" ".join(str(x[0]) for x in tup)))

    def get_verb_phrases(nltk_query):
        data_tok = nltk.word_tokenize(nltk_query) #tokenisation
        data_pos = nltk.pos_tag(data_tok) #POS tagging
        cfgs = ["CUSTOMCHUNK: {<VB><.*>{0,3}<NN>}",
                "CUSTOMCHUNK: {<VB><.*>{0,3}<NNP>}",
                "CUSTOMCHUNK: {<VBG><.*>{0,3}<NNP>}",
                "CUSTOMCHUNK: {<VBG><.*>{0,3}<NN>}",
                "CUSTOMCHUNK: {<VBG><.*>{0,3}<PRP><NN>}",
                "CUSTOMCHUNK: {<VB><.*>{0,3}<PRP><NN>}",
                "CUSTOMCHUNK: {<VB><.*>{0,3}<NNPS>}",
                "CUSTOMCHUNK: {<VBG><.*>{0,3}<NNPS>}",
                "CUSTOMCHUNK: {<VBG><.*>{0,3}<NNS>}",
                "CUSTOMCHUNK: {<VBG><.*>{0,3}<PRP><NNS>}",
                "CUSTOMCHUNK: {<VB><.*>{0,3}<PRP><NNS>}",
                "CUSTOMCHUNK: {<VBZ><.*>{0,3}<NNP>}",
                "CUSTOMCHUNK: {<VBZ><.*>{0,3}<NN>}",
                "CUSTOMCHUNK: {<VBZ><.*>{0,3}<PRP><NN>}",
                "CUSTOMCHUNK: {<VBZ><.*>{0,3}<NNPS>}",
                "CUSTOMCHUNK: {<VBZ><.*>{0,3}<NNS>}",
                "CUSTOMCHUNK: {<VBZ><.*>{0,3}<PRP><NNS>}",
                "CUSTOMCHUNK: {<VBZ><.*>{0,3}<PRP><NNP>}",
                "CUSTOMCHUNK: {<VBZ><.*>{0,3}<PRP><NNPS>}"
            ]
        vbs = []
        for cfg_1 in cfgs: 
            chunker = nltk.RegexpParser(cfg_1)
            data_chunked = chunker.parse(data_pos)
            vbs += extract_vbs(data_chunked)
        return vbs

    # Query qdrant and get score
    def sentence_get_query_vector(query):
        query_vector = st_model.encode(query)
        return query_vector

    def search_collection(ontology, query_vector, point_count):
        query_filter=Filter(
            must=[  
                FieldCondition(
                    key='ontology',
                    match=MatchValue(value=ontology)
                )
            ]
        )
        
        hits = client.search(
            collection_name=f"{collection_name}",
            query_vector=query_vector,
            query_filter=query_filter, 
            append_payload=True,  
            limit=point_count 
        )
        return hits

    # Compute the semantic similarity
    cl_bhvr_onto_lst = preprocess(bhvr_onto_lst)
    orig_cl_dict = {x:y for x,y in zip(cl_bhvr_onto_lst, bhvr_onto_lst)}
    emb_bhvr_onto_lst = sentence_embeddings(cl_bhvr_onto_lst)
    add_to_collection(cl_bhvr_onto_lst, emb_bhvr_onto_lst)
    point_count = count_collection()
    query = document
    vbs = get_verb_phrases(query)
    cl_vbs = preprocess(vbs)
    emb_vbs = sentence_embeddings(cl_vbs)
    vb_ind = -1
    highlights = []
    highlight_scores = []
    result_dfs = []
    for query_vector in emb_vbs:
        vb_ind += 1
        hist = search_collection(q_onto_dict[q_no], query_vector, point_count)
        hist_dict = [dict(x) for x in hist]
        scores = [x['score'] for x in hist_dict]
        payloads = [orig_cl_dict[x['payload']['phrase']] for x in hist_dict]
        result_df = pd.DataFrame({'score': scores, 'glossary': payloads})
        result_df = result_df[result_df['score'] >= sim_threshold]
        if len(result_df) > 0:
            highlights.append(vbs[vb_ind])
            highlight_scores.append(result_df.score.max())
            result_df['phrase'] = [vbs[vb_ind]] * len(result_df)
            result_df = result_df.sort_values(by='score', ascending=False).reset_index(drop=True)
            result_dfs.append(result_df)
        else:
            continue
    if len(highlights) > 0:
        result_df = pd.concat(result_dfs).reset_index(drop = True)
        result_df = result_df.sort_values(by='score', ascending=False).reset_index(drop=True)
        predictions = result_df[['phrase', 'glossary', 'score']]
    else:
        predictions = pd.DataFrame({'phrase': [], 'glossary': [], 'score': []})
 
    # Transform predictions to JSON
    result = {'output': []}
    list_out = predictions.to_dict(orient="records")
    result['output'] = list_out
    result = json.dumps(result)
    return flask.Response(response=result, status=200, mimetype='application/json')

# run the application.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be removed before deploying a production application.
    application.debug = True
    application.run()