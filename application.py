# core dependencies
from gensim.test.utils import datapath
from gensim import models
import spacy
from flair.models import TARSClassifier
from flair.data import Sentence
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Batch, Filter, FieldCondition, Range, MatchValue
import nltk
from nltk import sent_tokenize
from sklearn.feature_extraction import text
import re
from sentence_transformers import SentenceTransformer, util
from setfit import SetFitModel
import openai

# utility dependencies
import os
import numpy as np
import pandas as pd
import pickle
import joblib
import random
from dotenv import load_dotenv

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

#Azure OpenAI parameters
load_dotenv()
api_key = os.getenv("API_KEY")
api_base = os.getenv("API_BASE")
api_type = os.getenv("API_TYPE")
api_version = os.getenv("API_VERSION")
deployment_id=os.getenv("DEPLOYMENT_ID")
openai.api_key = api_key
openai.api_base = api_base
openai.api_type = api_type
openai.api_version = api_version

# Load the SpaCy model
#nlp = spacy.load(os.path.abspath('en_core_web_sm-3.3.0'))   #when running in local machine
nlp = spacy.load(os.path.abspath('en_core_web_sm-3.5.0'))    #when running in UoM VM

# Load the LDA topic model
model = models.ldamodel.LdaModel.load(temp_file)
with open(dict_file, 'rb') as f:
    topic_names = pickle.load(f)

# Load the gender model
modlist = joblib.load(pipe_file)
count_vect, tfidf_transformer, text_clf3, gender_tokens = modlist[0], modlist[1], modlist[2], modlist[3]

# Load the behaviour frequency model
f_modlist = joblib.load(trent_pipe_file)
f_count_vect, f_tfidf_transformer, f_text_clf3 = f_modlist[0], f_modlist[1], f_modlist[2]

# Load the TARS model (Zero-shot)
tagger = TARSClassifier.load('tars-base')

# Load the TARS model (Few-shot) used in /topics_p2_q1
tars_model_path = 'few-shot-model-1'
tars = TARSClassifier().load(tars_model_path+'/best-model.pt')

# Load the TARS model (Few-shot) used in /topics_p2_q2
tars_model_path = 'few-shot-model-2'
tars_p2_q2 = TARSClassifier().load(tars_model_path+'/best-model.pt')

# Load the TARS model (Few-shot) used in /topics_p3_function
tars_gain_avoid_model_path = 'few-shot-model-gain-avoid'
tars_gain_avoid = TARSClassifier().load(tars_gain_avoid_model_path+'/best-model.pt')

# Load the Sentence Transformer model
st_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

# Load the SetFit models for PBSP Page 3 (Behaviours, Frequency, Duration, Severity)
sf_bhvr_model_name = "setfit-zero-shot-classification-pbsp-p3-bhvr"
sf_bhvr_model = SetFitModel.from_pretrained(f"aammari/{sf_bhvr_model_name}")
sf_freq_model_name = "setfit-zero-shot-classification-pbsp-p3-freq"
sf_freq_model = SetFitModel.from_pretrained(f"aammari/{sf_freq_model_name}")
sf_dur_model_name = "setfit-zero-shot-classification-pbsp-p3-dur"
sf_dur_model = SetFitModel.from_pretrained(f"aammari/{sf_dur_model_name}")
sf_sev_model_name = "setfit-zero-shot-classification-pbsp-p3-sev"
sf_sev_model = SetFitModel.from_pretrained(f"aammari/{sf_sev_model_name}")

# Load the SetFit models for PBSP Page 3 (Trigger, Consequence)
sf_trig_model_name = "setfit-zero-shot-classification-pbsp-p3-trig"
sf_trig_model = SetFitModel.from_pretrained(f"aammari/{sf_trig_model_name}")
sf_cons_model_name = "setfit-zero-shot-classification-pbsp-p3-cons"
sf_cons_model = SetFitModel.from_pretrained(f"aammari/{sf_cons_model_name}")

# Load the SetFit models for PBSP Page 3 (Function)
sf_func_model_name = "setfit-zero-shot-classification-pbsp-p3-func"
sf_func_model = SetFitModel.from_pretrained(f"aammari/{sf_func_model_name}")

# Load the SetFit models for PBSP Page 4 (S.M.A.R.T. Goals)
model_s_name = "setfit-zero-shot-classification-pbsp-p4-specific"
model_s = SetFitModel.from_pretrained(f"aammari/{model_s_name}")
model_m_name = "setfit-zero-shot-classification-pbsp-p4-meas"
model_m = SetFitModel.from_pretrained(f"aammari/{model_m_name}")
model_a_name = "setfit-zero-shot-classification-pbsp-p4-achiev"
model_a = SetFitModel.from_pretrained(f"aammari/{model_a_name}")
model_r_name = "setfit-zero-shot-classification-pbsp-p4-rel"
model_r = SetFitModel.from_pretrained(f"aammari/{model_r_name}")
model_t_name = "setfit-zero-shot-classification-pbsp-p4-time"
model_t = SetFitModel.from_pretrained(f"aammari/{model_t_name}")

# Load the SetFit models for PBSP Page 4 (Quality of Life Goals)
model_q_name = "setfit-zero-shot-classification-pbsp-p4-quality"
model_q = SetFitModel.from_pretrained(f"{model_q_name}")

# Load the SetFit models for PBSP Page 1
sf_p1_model_name = "setfit-zero-shot-classification-pbsp-p1"
sf_p1_model = SetFitModel.from_pretrained(f"aammari/{sf_p1_model_name}")
sf_p1_comm_model_name = "setfit-zero-shot-classification-pbsp-p1-comm"
sf_p1_comm_model = SetFitModel.from_pretrained(f"aammari/{sf_p1_comm_model_name}")
sf_p1_life_model_name = "setfit-zero-shot-classification-pbsp-p1-life"
sf_p1_life_model = SetFitModel.from_pretrained(f"aammari/{sf_p1_life_model_name}")
sf_p1_likes_model_name = "setfit-zero-shot-classification-pbsp-p1-likes"
sf_p1_likes_model = SetFitModel.from_pretrained(f"aammari/{sf_p1_likes_model_name}")

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
        tars
        tars_p2_q2
        tars_gain_avoid
        st_model
        sf_bhvr_model
        sf_freq_model
        sf_dur_model
        sf_sev_model
        sf_trig_model
        sf_cons_model
        sf_func_model
        sf_p1_model
        sf_p1_comm_model
        sf_p1_life_model
        sf_p1_likes_model
        model_s
        model_m
        model_a
        model_r
        model_t
        model_q
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
    p_classes = {'speak_to_family': 0,
                'stakeholder_interview': 1,
                'case_file_review': 2,
                'observation': 3,
                'no_information_collection': 4}
    ind_topic_dict = {
                    0: 'SPEAK-TO-FAMILY',
                    1: 'STAKEHOLDER-INTERVIEW',
                    2: 'FILE-REVIEW',
                    3: 'OBSERVATION/COMMUNICATION',
                    4: 'NO-COLLECTED-INFO'
                }
    valid_topics = [ind_topic_dict[i] for i in range(0, 4)]
    passing_score = 0.25
    final_passing = 0.0

    # Get input JSON data and convert it to a DF
    input_json = flask.request.get_json()
    input_json = json.dumps(input_json['input'])
    input_df = pd.read_json(input_json,orient='list')

    # Get the query parameter value corresponsing to the output type: phrase | topic_agg | topic_scores
    resp_output = flask.request.args.get("output")

    #sentence extraction
    def extract_sentences(query):
        # Compile the regular expression pattern
        pattern = re.compile(r'[.,;!?]')
        # Split the sentences on the punctuation characters
        sentences = [query]
        split_sentences = [pattern.split(sentence) for sentence in sentences]
        # Flatten the list of split sentences
        flat_list = [item for sublist in split_sentences for item in sublist]
        # Remove empty strings from the list
        filtered_sentences = [sentence.strip() for sentence in flat_list if sentence.strip()]
        return filtered_sentences

    #Text Preprocessing
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
    
    #query and get predicted topic
    def get_topic(sentences):
        preds = []
        for t in sentences:
            sentence = Sentence(t)
            tars.predict(sentence)
            try:
                pred = p_classes[sentence.tag]
            except:
                pred = 4
            preds.append(pred)
        return preds
    def get_topic_scores(sentences):
        preds = []
        for t in sentences:
            sentence = Sentence(t)
            tars.predict(sentence)
            try:
                pred = sentence.score
            except:
                pred = 0.75
            preds.append(pred)
        return preds
    
    # Detect topics in the text
    documents = input_df['text'].tolist()
    document = documents[0]    # Currently this endpoint expects a single text input
    sentences = extract_sentences(document)
    cl_sentences = preprocess(sentences)
    topic_inds = get_topic(cl_sentences)
    topics = [ind_topic_dict[i] for i in topic_inds]
    scores = get_topic_scores(cl_sentences)
    result_df = pd.DataFrame({'phrase': sentences, 'topic': topics, 'score': scores})
    predictions = result_df[(result_df['score'] >= passing_score) & (result_df['topic'] != 'NO-COLLECTED-INFO')]
    
    # required if resp_output is either 'topic_agg' or 'topic_scores'
    def topic_output(predictions, resp_output):
        agg_df = predictions.groupby('topic')['score'].sum()
        agg_df = agg_df.to_frame()
        agg_df.columns = ['Total Score']
        agg_df = agg_df.assign(
            score=lambda x: x['Total Score'] / x['Total Score'].sum()
        )
        agg_df = agg_df.sort_values(by='score', ascending=False)
        agg_df['topic'] = agg_df.index
        rem_topics = [vt for vt in valid_topics if not vt in agg_df.topic.tolist()]
        if len(rem_topics) > 0:
            rem_agg_df = pd.DataFrame({'topic': rem_topics, 'score': 0.0, 'Total Score': 0.0})
            agg_df = pd.concat([agg_df, rem_agg_df])
        # Set the score column to 0 or 1 based on final_passing
        if resp_output == 'topic_scores':
            agg_df['score'] = [1 if score > final_passing else 0 for score in agg_df['score']]

        predictions = agg_df[['topic', 'score']]
        return predictions

    if len(predictions) > 0 and resp_output != 'phrase':
        predictions = topic_output(predictions, resp_output)
    else:
        pass

    # Transform predictions to JSON
    result = {'output': []}
    list_out = predictions.to_dict(orient="records")
    result['output'] = list_out
    result = json.dumps(result)
    return flask.Response(response=result, status=200, mimetype='application/json')

@application.route('/topics_p2_q2', methods=['POST'])
def get_topics_p2_q2():
    # Initialize variables
    p_classes = {'psychiatric_assessment': 0,
                 'medical_assessment': 1,
                 'no_assessment': 2,
                 'speech_and_language_assessment': 3}
    ind_topic_dict = {
                        0: 'PSYCHIATRIC-ASSESSMENT',
                        1: 'MEDICAL-ASSESSMENT',
                        2: 'NO-ASSESSMENT',
                        3: 'SPEECH-AND-LANGUAGE-ASSESSMENT'
                }
    valid_topics = [ind_topic_dict[i] for i in range(0, 4) if i != 2]
    passing_score = 0.25
    final_passing = 0.0

    # Get input JSON data and convert it to a DF
    input_json = flask.request.get_json()
    input_json = json.dumps(input_json['input'])
    input_df = pd.read_json(input_json,orient='list')

    # Get the query parameter value corresponsing to the output type: phrase | topic_agg | topic_scores
    resp_output = flask.request.args.get("output")

    #sentence extraction
    def extract_sentences(query):
        # Compile the regular expression pattern
        pattern = re.compile(r'[.,;!?]')
        # Split the sentences on the punctuation characters
        sentences = [query]
        split_sentences = [pattern.split(sentence) for sentence in sentences]
        # Flatten the list of split sentences
        flat_list = [item for sublist in split_sentences for item in sublist]
        # Remove empty strings from the list
        filtered_sentences = [sentence.strip() for sentence in flat_list if sentence.strip()]
        return filtered_sentences

    #Text Preprocessing
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
    
    #query and get predicted topic
    def get_topic(sentences):
        preds = []
        for t in sentences:
            sentence = Sentence(t)
            tars_p2_q2.predict(sentence)
            try:
                pred = p_classes[sentence.tag]
            except:
                pred = 2
            preds.append(pred)
        return preds
    def get_topic_scores(sentences):
        preds = []
        for t in sentences:
            sentence = Sentence(t)
            tars_p2_q2.predict(sentence)
            try:
                pred = sentence.score
            except:
                pred = 0.75
            preds.append(pred)
        return preds
    
    # Detect topics in the text
    documents = input_df['text'].tolist()
    document = documents[0]    # Currently this endpoint expects a single text input
    sentences = extract_sentences(document)
    cl_sentences = preprocess(sentences)
    topic_inds = get_topic(cl_sentences)
    topics = [ind_topic_dict[i] for i in topic_inds]
    scores = get_topic_scores(cl_sentences)
    result_df = pd.DataFrame({'phrase': sentences, 'topic': topics, 'score': scores})
    predictions = result_df[(result_df['score'] >= passing_score) & (result_df['topic'] != 'NO-ASSESSMENT')]
    
    # required if resp_output is either 'topic_agg' or 'topic_scores'
    def topic_output(predictions, resp_output):
        agg_df = predictions.groupby('topic')['score'].sum()
        agg_df = agg_df.to_frame()
        agg_df.columns = ['Total Score']
        agg_df = agg_df.assign(
            score=lambda x: x['Total Score'] / x['Total Score'].sum()
        )
        agg_df = agg_df.sort_values(by='score', ascending=False)
        agg_df['topic'] = agg_df.index
        rem_topics = [vt for vt in valid_topics if not vt in agg_df.topic.tolist()]
        if len(rem_topics) > 0:
            rem_agg_df = pd.DataFrame({'topic': rem_topics, 'score': 0.0, 'Total Score': 0.0})
            agg_df = pd.concat([agg_df, rem_agg_df])
        # Set the score column to 0 or 1 based on final_passing
        if resp_output == 'topic_scores':
            agg_df['score'] = [1 if score > final_passing else 0 for score in agg_df['score']]

        predictions = agg_df[['topic', 'score']]
        return predictions

    if len(predictions) > 0 and resp_output != 'phrase':
        predictions = topic_output(predictions, resp_output)
    else:
        pass

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
        cfgs = [
                "CUSTOMCHUNK: {<VB><.*>{0,3}<NN>}",
                "CUSTOMCHUNK: {<VB><.*>{0,3}<NNP>}",
                "CUSTOMCHUNK: {<VB><.*>{0,3}<PRP><NN>}",
                "CUSTOMCHUNK: {<VB><.*>{0,3}<PRP><NNS>}",
                "CUSTOMCHUNK: {<VB><.*>{0,3}<NNPS>}",
                "CUSTOMCHUNK: {<VB><.*>{0,3}<NNS>}",
                "CUSTOMCHUNK: {<VB><.*>{0,3}<PRP><NNP>}",
                "CUSTOMCHUNK: {<VB><.*>{0,3}<PRP><NNPS>}",
                "CUSTOMCHUNK: {<VBN><.*>{0,3}<NN>}",
                "CUSTOMCHUNK: {<VBN><.*>{0,3}<NNP>}",
                "CUSTOMCHUNK: {<VBN><.*>{0,3}<PRP><NN>}",
                "CUSTOMCHUNK: {<VBN><.*>{0,3}<PRP><NNS>}",
                "CUSTOMCHUNK: {<VBN><.*>{0,3}<NNPS>}",
                "CUSTOMCHUNK: {<VBN><.*>{0,3}<NNS>}",
                "CUSTOMCHUNK: {<VBN><.*>{0,3}<PRP><NNP>}",
                "CUSTOMCHUNK: {<VBN><.*>{0,3}<PRP><NNPS>}",
                "CUSTOMCHUNK: {<VBG><.*>{0,3}<NN>}",
                "CUSTOMCHUNK: {<VBG><.*>{0,3}<NNP>}",
                "CUSTOMCHUNK: {<VBG><.*>{0,3}<PRP><NN>}",
                "CUSTOMCHUNK: {<VBG><.*>{0,3}<PRP><NNS>}",
                "CUSTOMCHUNK: {<VBG><.*>{0,3}<NNPS>}",
                "CUSTOMCHUNK: {<VBG><.*>{0,3}<NNS>}",
                "CUSTOMCHUNK: {<VBG><.*>{0,3}<PRP><NNP>}",
                "CUSTOMCHUNK: {<VBG><.*>{0,3}<PRP><NNPS>}",
                "CUSTOMCHUNK: {<VBP><.*>{0,3}<NN>}",
                "CUSTOMCHUNK: {<VBP><.*>{0,3}<NNP>}",
                "CUSTOMCHUNK: {<VBP><.*>{0,3}<PRP><NN>}",
                "CUSTOMCHUNK: {<VBP><.*>{0,3}<PRP><NNS>}",
                "CUSTOMCHUNK: {<VBP><.*>{0,3}<NNPS>}",
                "CUSTOMCHUNK: {<VBP><.*>{0,3}<NNS>}",
                "CUSTOMCHUNK: {<VBP><.*>{0,3}<PRP><NNP>}",
                "CUSTOMCHUNK: {<VBP><.*>{0,3}<PRP><NNPS>}",
                "CUSTOMCHUNK: {<VBZ><.*>{0,3}<NN>}",
                "CUSTOMCHUNK: {<VBZ><.*>{0,3}<NNP>}",
                "CUSTOMCHUNK: {<VBZ><.*>{0,3}<PRP><NN>}",
                "CUSTOMCHUNK: {<VBZ><.*>{0,3}<PRP><NNS>}",
                "CUSTOMCHUNK: {<VBZ><.*>{0,3}<NNPS>}",
                "CUSTOMCHUNK: {<VBZ><.*>{0,3}<NNS>}",
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
        result_df = pd.DataFrame({'score': scores, 'topic': ['GLOSSARY'] * len(payloads), 'subtopic': payloads})
        result_df = result_df[result_df['score'] >= sim_threshold]
        result_df = result_df.sort_values(by='score', ascending=False).reset_index(drop=True).head(1)
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
        predictions = result_df[['phrase', 'topic', 'subtopic', 'score']]
    else:
        predictions = pd.DataFrame({'phrase': [], 'topic': [], 'subtopic': [], 'score': []})
 
    # Compute the SetFit models (Behaviours, Frequency, Duration, Severity)
    #setfit sentence extraction
    def extract_sentences(nltk_query):
        sentences = sent_tokenize(nltk_query)
        return sentences
    
    #setfit bhvr query and get predicted topic
    def get_sf_bhvr_topic(sentences):
        preds = list(sf_bhvr_model(sentences))
        return preds
    def get_sf_bhvr_topic_scores(sentences):
        preds = sf_bhvr_model.predict_proba(sentences)
        preds = [max(list(x)) for x in preds]
        return preds

    # setfit bhvr format output
    ind_bhvr_topic_dict = {
            0: 'NO BEHAVIOUR',
            1: 'BEHAVIOUR',
        }

   #setfit freq query and get predicted topic
    def get_sf_freq_topic(sentences):
        preds = list(sf_freq_model(sentences))
        return preds
    def get_sf_freq_topic_scores(sentences):
        preds = sf_freq_model.predict_proba(sentences)
        preds = [max(list(x)) for x in preds]
        return preds

    # setfit freq format output
    ind_freq_topic_dict = {
            0: 'NO FREQUENCY',
            1: 'FREQUENCY',
        }

    #setfit dur query and get predicted topic
    def get_sf_dur_topic(sentences):
        preds = list(sf_dur_model(sentences))
        return preds
    def get_sf_dur_topic_scores(sentences):
        preds = sf_dur_model.predict_proba(sentences)
        preds = [max(list(x)) for x in preds]
        return preds

    # setfit dur format output
    ind_dur_topic_dict = {
            0: 'NO DURATION',
            1: 'DURATION',
        }
    
    #setfit sev query and get predicted topic
    def get_sf_sev_topic(sentences):
        preds = list(sf_sev_model(sentences))
        return preds
    def get_sf_sev_topic_scores(sentences):
        preds = sf_sev_model.predict_proba(sentences)
        preds = [max(list(x)) for x in preds]
        return preds

    # setfit sev format output
    ind_sev_topic_dict = {
            0: 'NO SEVERITY',
            1: 'SEVERITY',
        }
    
    #setfit behaviour
    sentences = extract_sentences(query)
    cl_sentences = preprocess(sentences)
    topic_inds = get_sf_bhvr_topic(cl_sentences)
    topics = [ind_bhvr_topic_dict[i] for i in topic_inds]
    scores = get_sf_bhvr_topic_scores(cl_sentences)
    sf_bhvr_result_df = pd.DataFrame({'phrase': sentences, 'topic': topics, 'subtopic': [''] * len(scores), 'score': scores})
    sf_bhvr_sub_result_df = sf_bhvr_result_df[sf_bhvr_result_df['topic'] == 'BEHAVIOUR']
    if len(sf_bhvr_sub_result_df) > 0:
        predictions = pd.concat([predictions, sf_bhvr_sub_result_df])

    #setfit frequency
    topic_inds = get_sf_freq_topic(cl_sentences)
    topics = [ind_freq_topic_dict[i] for i in topic_inds]
    scores = get_sf_freq_topic_scores(cl_sentences)
    sf_freq_result_df = pd.DataFrame({'phrase': sentences, 'topic': topics, 'subtopic': [''] * len(scores), 'score': scores})
    sf_freq_sub_result_df = sf_freq_result_df[sf_freq_result_df['topic'] == 'FREQUENCY']
    if len(sf_freq_sub_result_df) > 0:
        predictions = pd.concat([predictions, sf_freq_sub_result_df])

    #setfit duration
    topic_inds = get_sf_dur_topic(cl_sentences)
    topics = [ind_dur_topic_dict[i] for i in topic_inds]
    scores = get_sf_dur_topic_scores(cl_sentences)
    sf_dur_result_df = pd.DataFrame({'phrase': sentences, 'topic': topics, 'subtopic': [''] * len(scores), 'score': scores})
    sf_dur_sub_result_df = sf_dur_result_df[sf_dur_result_df['topic'] == 'DURATION']
    if len(sf_dur_sub_result_df) > 0:
        predictions = pd.concat([predictions, sf_dur_sub_result_df])

    #setfit severity
    topic_inds = get_sf_sev_topic(cl_sentences)
    topics = [ind_sev_topic_dict[i] for i in topic_inds]
    scores = get_sf_sev_topic_scores(cl_sentences)
    sf_sev_result_df = pd.DataFrame({'phrase': sentences, 'topic': topics, 'subtopic': [''] * len(scores), 'score': scores})
    sf_sev_sub_result_df = sf_sev_result_df[sf_sev_result_df['topic'] == 'SEVERITY']
    if len(sf_sev_sub_result_df) > 0:
        predictions = pd.concat([predictions, sf_sev_sub_result_df])

    # Transform predictions to JSON
    result = {'output': []}
    list_out = predictions.to_dict(orient="records")
    result['output'] = list_out
    result = json.dumps(result)
    return flask.Response(response=result, status=200, mimetype='application/json')

@application.route('/topics_p1', methods=['POST'])
def get_topics_p1():
    # Get input JSON data and convert it to a DF
    input_json = flask.request.get_json()
    input_json = json.dumps(input_json['input'])
    input_df = pd.read_json(input_json,orient='list')

    # Get the query parameter value corresponsing to the output type: phrase | topic_agg | topic_scores
    resp_output = flask.request.args.get("output")

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

    #sentence extraction
    def extract_sentences(nltk_query):
        sentences = sent_tokenize(nltk_query)
        return sentences

    #query and get predicted topic
    def get_topic(sentences):
        preds = list(sf_p1_model(sentences))
        return preds
    def get_topic_scores(sentences):
        preds = sf_p1_model.predict_proba(sentences)
        preds = [max(list(x)) for x in preds]
        return preds

    # format output
    ind_topic_dict = {
            0: 'FAMILY HISTORY',
            1: 'DISGNOSED DISABILITIES',
            2: 'HEALTH INFO',
            3: 'COMMUNICATION',
            4: 'LIKES',
            5: 'DISLIKES',
            6: 'SENSORY EXPERIENCES',
            7: 'GOALS & ASPIRATIONS'
        }

    passing_score = 0.25

    # Detect topics in the text
    documents = input_df['text'].tolist()
    document = documents[0]    # Currently this endpoint expects a single text input

    # required if resp_output == 'phrase'
    sentences = extract_sentences(document)
    cl_sentences = preprocess(sentences)
    topic_inds = get_topic(cl_sentences)
    topics = [ind_topic_dict[i] for i in topic_inds]
    scores = get_topic_scores(cl_sentences)
    result_df = pd.DataFrame({'phrase': sentences, 'topic': topics, 'score': scores})
    predictions = result_df[result_df['score'] >= passing_score]

    # required if resp_output is either 'topic_agg' or 'topic_scores'
    final_passing = 0.1
    def topic_output(predictions, resp_output):
        agg_df = predictions.groupby('topic')['score'].sum()
        agg_df = agg_df.to_frame()
        agg_df.columns = ['Total Score']
        agg_df = agg_df.assign(
            score=lambda x: x['Total Score'] / x['Total Score'].sum()
        )
        agg_df = agg_df.sort_values(by='score', ascending=False)
        agg_df['topic'] = agg_df.index
        rem_topics = [ind_topic_dict[i] for i in range(0, 8) if not ind_topic_dict[i] in agg_df.topic.tolist()]
        if len(rem_topics) > 0:
            rem_agg_df = pd.DataFrame({'topic': rem_topics, 'score': 0.0, 'Total Score': 0.0})
            agg_df = pd.concat([agg_df, rem_agg_df])
        # Set the score column to 0 or 1 based on final_passing
        if resp_output == 'topic_scores':
            agg_df['score'] = [1 if score >= final_passing else 0 for score in agg_df['score']]

        predictions = agg_df[['topic', 'score']]
        return predictions

    if resp_output != 'phrase':
        predictions = topic_output(predictions, resp_output)
    
    # Transform predictions to JSON
    result = {'output': []}
    list_out = predictions.to_dict(orient="records")
    result['output'] = list_out
    result = json.dumps(result)
    return flask.Response(response=result, status=200, mimetype='application/json')

@application.route('/topics_p3_abc', methods=['POST'])
def topics_p3_abc():
    # Get input JSON data and convert it to a DF
    input_json = flask.request.get_json()
    input_json = json.dumps(input_json['input'])
    input_df = pd.read_json(input_json,orient='list')

    # Get the query parameter value corresponsing to the A-B-C chain item (event, trigger, behaviour, consequence)
    item = flask.request.args.get("item")

    # Item - Ontology mapping
    q_onto_dict = {'behaviour': "behaviours", 
                   'event': "setting_events"
                  }

    # Detect topics in the text
    documents = input_df['text'].tolist()
    document = documents[0]    # Currently this endpoint expects a single text input
    query = document

    # Read ontology
    if item == 'behaviour':
        onto_df = pd.read_csv(onto_path+"ontology_page3_bhvr.csv", header=None, encoding="utf-8").dropna()
        onto_df.columns = ['text']
        onto_lst = onto_df['text'].tolist()
    elif item == 'event':
        onto_df = pd.read_csv(onto_path+"ontology_page3_event.csv", header=None, encoding="utf-8").dropna()
        onto_df.columns = ['text']
        onto_lst = onto_df['text'].tolist()
    else:
        onto_lst = []

    #Text Preprocessing
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
    def add_to_collection(cl_onto_lst, emb_onto_lst):
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )
        doc_count = len(emb_onto_lst)
        ids = list(range(1, doc_count+1))
        payloads = [{"ontology": q_onto_dict[item], "phrase": x} for x in cl_onto_lst]
        vectors = emb_onto_lst
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

    #noun phrase extraction
    def extract_noun_phrases(text):
        # Tokenize the text
        tokens = nltk.word_tokenize(text)

        # Part-of-speech tag the tokens
        tagged_tokens = nltk.pos_tag(tokens)

        # Define the noun phrase grammar
        grammar = r"""
        NP: {<DT|PP\$>?<JJ>*<NN|NNS|NNP|NNPS>+}  # noun phrase with optional determiner and adjectives
            {<NNP>+}                              # proper noun phrase
            {<PRP\$>?<NN|NNS|NNP|NNPS>+}          # noun phrase with optional possessive pronoun
        """

        # Extract the noun phrases
        parser = nltk.RegexpParser(grammar)
        tree = parser.parse(tagged_tokens)

        # Extract the phrase text from the tree
        phrases = []
        for subtree in tree.subtrees():
            if subtree.label() == "NP":
                phrase = " ".join([token[0] for token in subtree.leaves()])
                phrases.append(phrase)
        return phrases

    #verb phrase extraction
    def extract_vbs(data_chunked):
        for tup in data_chunked:
            if len(tup) > 2:
                yield(str(" ".join(str(x[0]) for x in tup)))

    def get_verb_phrases(nltk_query):
        data_tok = nltk.word_tokenize(nltk_query) #tokenisation
        data_pos = nltk.pos_tag(data_tok) #POS tagging
        cfgs = [
            "CUSTOMCHUNK: {<VB><.*>{0,3}<NN>}",
            "CUSTOMCHUNK: {<VB><.*>{0,3}<NNP>}",
            "CUSTOMCHUNK: {<VB><.*>{0,3}<PRP><NN>}",
            "CUSTOMCHUNK: {<VB><.*>{0,3}<PRP><NNS>}",
            "CUSTOMCHUNK: {<VB><.*>{0,3}<NNPS>}",
            "CUSTOMCHUNK: {<VB><.*>{0,3}<NNS>}",
            "CUSTOMCHUNK: {<VB><.*>{0,3}<PRP><NNP>}",
            "CUSTOMCHUNK: {<VB><.*>{0,3}<PRP><NNPS>}",
            "CUSTOMCHUNK: {<VBN><.*>{0,3}<NN>}",
            "CUSTOMCHUNK: {<VBN><.*>{0,3}<NNP>}",
            "CUSTOMCHUNK: {<VBN><.*>{0,3}<PRP><NN>}",
            "CUSTOMCHUNK: {<VBN><.*>{0,3}<PRP><NNS>}",
            "CUSTOMCHUNK: {<VBN><.*>{0,3}<NNPS>}",
            "CUSTOMCHUNK: {<VBN><.*>{0,3}<NNS>}",
            "CUSTOMCHUNK: {<VBN><.*>{0,3}<PRP><NNP>}",
            "CUSTOMCHUNK: {<VBN><.*>{0,3}<PRP><NNPS>}",
            "CUSTOMCHUNK: {<VBG><.*>{0,3}<NN>}",
            "CUSTOMCHUNK: {<VBG><.*>{0,3}<NNP>}",
            "CUSTOMCHUNK: {<VBG><.*>{0,3}<PRP><NN>}",
            "CUSTOMCHUNK: {<VBG><.*>{0,3}<PRP><NNS>}",
            "CUSTOMCHUNK: {<VBG><.*>{0,3}<NNPS>}",
            "CUSTOMCHUNK: {<VBG><.*>{0,3}<NNS>}",
            "CUSTOMCHUNK: {<VBG><.*>{0,3}<PRP><NNP>}",
            "CUSTOMCHUNK: {<VBG><.*>{0,3}<PRP><NNPS>}",
            "CUSTOMCHUNK: {<VBP><.*>{0,3}<NN>}",
            "CUSTOMCHUNK: {<VBP><.*>{0,3}<NNP>}",
            "CUSTOMCHUNK: {<VBP><.*>{0,3}<PRP><NN>}",
            "CUSTOMCHUNK: {<VBP><.*>{0,3}<PRP><NNS>}",
            "CUSTOMCHUNK: {<VBP><.*>{0,3}<NNPS>}",
            "CUSTOMCHUNK: {<VBP><.*>{0,3}<NNS>}",
            "CUSTOMCHUNK: {<VBP><.*>{0,3}<PRP><NNP>}",
            "CUSTOMCHUNK: {<VBP><.*>{0,3}<PRP><NNPS>}",
            "CUSTOMCHUNK: {<VBZ><.*>{0,3}<NN>}",
            "CUSTOMCHUNK: {<VBZ><.*>{0,3}<NNP>}",
            "CUSTOMCHUNK: {<VBZ><.*>{0,3}<PRP><NN>}",
            "CUSTOMCHUNK: {<VBZ><.*>{0,3}<PRP><NNS>}",
            "CUSTOMCHUNK: {<VBZ><.*>{0,3}<NNPS>}",
            "CUSTOMCHUNK: {<VBZ><.*>{0,3}<NNS>}",
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

    # Compute the SetFit models (trigger, consequence)
    #setfit sentence extraction
    def extract_sentences(nltk_query):
        sentences = sent_tokenize(nltk_query)
        return sentences
    
    #setfit trig query and get predicted topic
    def get_sf_trig_topic(sentences):
        preds = list(sf_trig_model(sentences))
        return preds
    def get_sf_trig_topic_scores(sentences):
        preds = sf_trig_model.predict_proba(sentences)
        preds = [max(list(x)) for x in preds]
        return preds

    # setfit trig format output
    ind_trig_topic_dict = {
            0: 'NO TRIGGER',
            1: 'TRIGGER',
        }

    #setfit cons query and get predicted topic
    def get_sf_cons_topic(sentences):
        preds = list(sf_cons_model(sentences))
        return preds
    def get_sf_cons_topic_scores(sentences):
        preds = sf_cons_model.predict_proba(sentences)
        preds = [max(list(x)) for x in preds]
        return preds

    # setfit cons format output
    ind_cons_topic_dict = {
            0: 'NO CONSEQUENCE',
            1: 'CONSEQUENCE',
        }

    # Compute the semantic similarity (for item = event or item = bahaviour)
    if item in ['behaviour', 'event']:
        cl_onto_lst = preprocess(onto_lst)
        orig_cl_dict = {x:y for x,y in zip(cl_onto_lst, onto_lst)}
        emb_onto_lst = sentence_embeddings(cl_onto_lst)
        add_to_collection(cl_onto_lst, emb_onto_lst)
        point_count = count_collection()
        vbs = get_verb_phrases(query)
        cl_vbs = preprocess(vbs)
        sents = vbs
        cl_sents = cl_vbs
        if item == 'event':
            nouns = extract_noun_phrases(query)
            cl_nouns = preprocess(nouns)
            sents = vbs+nouns
            cl_sents = cl_vbs+cl_nouns
        emb_sents = sentence_embeddings(cl_sents)
        vb_ind = -1
        highlights = []
        highlight_scores = []
        result_dfs = []
        for query_vector in emb_sents:
            vb_ind += 1
            if len(sents[vb_ind].split()) <= 1:
                continue
            hist = search_collection(q_onto_dict[item], query_vector, point_count)
            hist_dict = [dict(x) for x in hist]
            scores = [x['score'] for x in hist_dict]
            payloads = [orig_cl_dict[x['payload']['phrase']] for x in hist_dict]
            result_df = pd.DataFrame({'score': scores, 'topic': [q_onto_dict[item].upper()] * len(payloads), 'subtopic': payloads})
            result_df = result_df[result_df['score'] >= sim_threshold]
            result_df = result_df.sort_values(by='score', ascending=False).reset_index(drop=True).head(1)
            if len(result_df) > 0:
                highlights.append(sents[vb_ind])
                highlight_scores.append(result_df.score.max())
                result_df['phrase'] = [sents[vb_ind]] * len(result_df)
                result_df = result_df.sort_values(by='score', ascending=False).reset_index(drop=True)
                result_dfs.append(result_df)
            else:
                continue
        if len(highlights) > 0:
            result_df = pd.concat(result_dfs).reset_index(drop = True)
            result_df = result_df.sort_values(by='score', ascending=False).reset_index(drop=True)
            predictions = result_df[['phrase', 'topic', 'subtopic', 'score']]
        else:
            predictions = pd.DataFrame({'phrase': [], 'topic': [], 'subtopic': [], 'score': []})

    #setfit trigger
    elif item == 'trigger':
        sentences = extract_sentences(query)
        cl_sentences = preprocess(sentences)
        topic_inds = get_sf_trig_topic(cl_sentences)
        topics = [ind_trig_topic_dict[i] for i in topic_inds]
        scores = get_sf_trig_topic_scores(cl_sentences)
        sf_trig_result_df = pd.DataFrame({'phrase': sentences, 'topic': topics, 'subtopic': [''] * len(scores), 'score': scores})
        sf_trig_sub_result_df = sf_trig_result_df[sf_trig_result_df['topic'] == 'TRIGGER']
        if len(sf_trig_sub_result_df) > 0:
            predictions = sf_trig_sub_result_df[['phrase', 'topic', 'subtopic', 'score']]
        else:
            predictions = pd.DataFrame({'phrase': [], 'topic': [], 'subtopic': [], 'score': []})

    #setfit consequence
    elif item == 'consequence':
        sentences = extract_sentences(query)
        cl_sentences = preprocess(sentences)
        topic_inds = get_sf_cons_topic(cl_sentences)
        topics = [ind_cons_topic_dict[i] for i in topic_inds]
        scores = get_sf_cons_topic_scores(cl_sentences)
        sf_cons_result_df = pd.DataFrame({'phrase': sentences, 'topic': topics, 'subtopic': [''] * len(scores), 'score': scores})
        sf_cons_sub_result_df = sf_cons_result_df[sf_cons_result_df['topic'] == 'CONSEQUENCE']
        if len(sf_cons_sub_result_df) > 0:
            predictions = sf_cons_sub_result_df[['phrase', 'topic', 'subtopic', 'score']]
        else:
            predictions = pd.DataFrame({'phrase': [], 'topic': [], 'subtopic': [], 'score': []})

    #case when item is invalid
    else:
        predictions = pd.DataFrame({'phrase': [], 'topic': [], 'subtopic': [], 'score': []})

    # Transform predictions to JSON
    result = {'output': []}
    list_out = predictions.to_dict(orient="records")
    result['output'] = list_out
    result = json.dumps(result)
    return flask.Response(response=result, status=200, mimetype='application/json')

@application.route('/topics_p3_function', methods=['POST'])
def topics_p3_function():
    # Initialize variables
    p_classes = {'gain_attention': 0,
                'avoid_attention': 1,
                'unknown': 2
                }
    ind_topic_dict = {
            0: 'GAIN-ATTENTION',
            1: 'AVOID-ATTENTION',
            2: 'UNKNOWN'
        }
    valid_topics = [ind_topic_dict[i] for i in range(0, 2)]
    passing_score = 0.25
    final_passing = 0.0

    # Get input JSON data and convert it to a DF
    input_json = flask.request.get_json()
    input_json = json.dumps(input_json['input'])
    input_df = pd.read_json(input_json,orient='list')

    # Get the query parameter value corresponsing to the function output type (detect, attention, attention_agg, attention_scores)
    resp_output = flask.request.args.get("output")

    # Detect topics in the text
    documents = input_df['text'].tolist()
    document = documents[0]    # Currently this endpoint expects a single text input
    query = document
    
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
 
    # Compute the SetFit model (Function)
    #setfit sentence extraction
    def extract_sentences(nltk_query):
        sentences = sent_tokenize(nltk_query)
        return sentences
    
    def get_sf_func_topic(sentences):
        preds = list(sf_func_model(sentences))
        return preds
    def get_sf_func_topic_scores(sentences):
        preds = sf_func_model.predict_proba(sentences)
        preds = [max(list(x)) for x in preds]
        return preds

    # setfit sev format output
    ind_func_topic_dict = {
        0: 'NO FUNCTION',
        1: 'FUNCTION',
    }

    #Compute the TARS  model (Gain / Avoid Attention)
    def get_topic(sentences):
        preds = []
        for t in sentences:
            sentence = Sentence(t)
            tars_gain_avoid.predict(sentence)
            try:
                pred = p_classes[sentence.tag]
            except:
                pred = 2
            preds.append(pred)
        return preds
    def get_topic_scores(sentences):
        preds = []
        for t in sentences:
            sentence = Sentence(t)
            tars_gain_avoid.predict(sentence)
            try:
                pred = sentence.score
            except:
                pred = 0.75
            preds.append(pred)
        return preds

    if resp_output == 'detect':
        sentences = extract_sentences(query)
        cl_sentences = preprocess(sentences)
        topic_inds = get_sf_func_topic(cl_sentences)
        topics = [ind_func_topic_dict[i] for i in topic_inds]
        scores = get_sf_func_topic_scores(cl_sentences)
        sf_func_result_df = pd.DataFrame({'phrase': sentences, 'topic': topics, 'subtopic': [''] * len(scores), 'score': scores})
        sf_func_sub_result_df = sf_func_result_df[sf_func_result_df['topic'] == 'FUNCTION']
        if len(sf_func_sub_result_df) > 0:
            predictions = sf_func_sub_result_df[['phrase', 'topic', 'subtopic', 'score']]
        else:
            predictions = pd.DataFrame({'phrase': [], 'topic': [], 'subtopic': [], 'score': []})
    
    else:
        sentences = extract_sentences(document)
        cl_sentences = preprocess(sentences)
        topic_inds = get_topic(cl_sentences)
        topics = [ind_topic_dict[i] for i in topic_inds]
        scores = get_topic_scores(cl_sentences)
        result_df = pd.DataFrame({'phrase': sentences, 'topic': topics, 'score': scores})
        predictions = result_df[(result_df['score'] >= passing_score) & (result_df['topic'] != 'UNKNOWN')]
        
        # required if resp_output is either 'attention_agg' or 'attention_scores'
        def topic_output(predictions, resp_output):
            agg_df = predictions.groupby('topic')['score'].sum()
            agg_df = agg_df.to_frame()
            agg_df.columns = ['Total Score']
            agg_df = agg_df.assign(
                score=lambda x: x['Total Score'] / x['Total Score'].sum()
            )
            agg_df = agg_df.sort_values(by='score', ascending=False)
            agg_df['topic'] = agg_df.index
            rem_topics = [vt for vt in valid_topics if not vt in agg_df.topic.tolist()]
            if len(rem_topics) > 0:
                rem_agg_df = pd.DataFrame({'topic': rem_topics, 'score': 0.0, 'Total Score': 0.0})
                agg_df = pd.concat([agg_df, rem_agg_df])
            # Set the score column to 0 or 1 based on final_passing
            if resp_output == 'attention_scores':
                agg_df['score'] = [1 if score > final_passing else 0 for score in agg_df['score']]

            predictions = agg_df[['topic', 'score']]
            return predictions

        if len(predictions) > 0 and resp_output != 'attention':
            predictions = topic_output(predictions, resp_output)
        else:
            pass
        
    # Transform predictions to JSON
    result = {'output': []}
    list_out = predictions.to_dict(orient="records")
    result['output'] = list_out
    result = json.dumps(result)
    return flask.Response(response=result, status=200, mimetype='application/json')

@application.route('/topics_p1_comm', methods=['POST'])
def get_topics_p1_comm():
    # Get input JSON data and convert it to a DF
    input_json = flask.request.get_json()
    input_json = json.dumps(input_json['input'])
    input_df = pd.read_json(input_json,orient='list')

    # Get the query parameter value corresponsing to the output type: phrase | topic_agg | topic_scores
    resp_output = flask.request.args.get("output")

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

    #sentence extraction
    def extract_sentences(nltk_query):
        sentences = sent_tokenize(nltk_query)
        return sentences

    #query and get predicted topic
    def get_topic(sentences):
        preds = list(sf_p1_comm_model(sentences))
        return preds
    def get_topic_scores(sentences):
        preds = sf_p1_comm_model.predict_proba(sentences)
        preds = [max(list(x)) for x in preds]
        return preds

    # format output
    ind_topic_dict = {
                    0: 'VERBAL COMMUNICATION',
                    1: 'NONVERBAL COMMUNICATION',
                    2: 'AUGMENTATIVE AND ALTERNATIVE COMMUNICATION',
                    3: 'WRITTEN COMMUNICATION',
                    4: 'ASSISTIVE TECHNOLOGY'
        }

    passing_score = 0.25

    # Detect topics in the text
    documents = input_df['text'].tolist()
    document = documents[0]    # Currently this endpoint expects a single text input

    # required if resp_output == 'phrase'
    sentences = extract_sentences(document)
    cl_sentences = preprocess(sentences)
    topic_inds = get_topic(cl_sentences)
    topics = [ind_topic_dict[i] for i in topic_inds]
    scores = get_topic_scores(cl_sentences)
    result_df = pd.DataFrame({'phrase': sentences, 'topic': topics, 'score': scores})
    predictions = result_df[result_df['score'] >= passing_score]

    # required if resp_output is either 'topic_agg' or 'topic_scores'
    final_passing = 0.0
    def topic_output(predictions, resp_output):
        agg_df = predictions.groupby('topic')['score'].sum()
        agg_df = agg_df.to_frame()
        agg_df.columns = ['Total Score']
        agg_df = agg_df.assign(
            score=lambda x: x['Total Score'] / x['Total Score'].sum()
        )
        agg_df = agg_df.sort_values(by='score', ascending=False)
        agg_df['topic'] = agg_df.index
        rem_topics = [ind_topic_dict[i] for i in range(0, 5) if not ind_topic_dict[i] in agg_df.topic.tolist()]
        if len(rem_topics) > 0:
            rem_agg_df = pd.DataFrame({'topic': rem_topics, 'score': 0.0, 'Total Score': 0.0})
            agg_df = pd.concat([agg_df, rem_agg_df])
        # Set the score column to 0 or 1 based on final_passing
        if resp_output == 'topic_scores':
            agg_df['score'] = [1 if score > final_passing else 0 for score in agg_df['score']]

        predictions = agg_df[['topic', 'score']]
        return predictions

    if resp_output != 'phrase':
        predictions = topic_output(predictions, resp_output)
    
    # Transform predictions to JSON
    result = {'output': []}
    list_out = predictions.to_dict(orient="records")
    result['output'] = list_out
    result = json.dumps(result)
    return flask.Response(response=result, status=200, mimetype='application/json')

@application.route('/topics_p1_life', methods=['POST'])
def get_topics_p1_life():
    # Get input JSON data and convert it to a DF
    input_json = flask.request.get_json()
    input_json = json.dumps(input_json['input'])
    input_df = pd.read_json(input_json,orient='list')

    # Get the query parameter value corresponsing to the output type: phrase | topic_agg | topic_scores
    resp_output = flask.request.args.get("output")

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

    #sentence extraction
    def extract_sentences(nltk_query):
        sentences = sent_tokenize(nltk_query)
        return sentences

    #query and get predicted topic
    def get_topic(sentences):
        preds = list(sf_p1_life_model(sentences))
        return preds
    def get_topic_scores(sentences):
        preds = sf_p1_life_model.predict_proba(sentences)
        preds = [max(list(x)) for x in preds]
        return preds

    # format output
    ind_topic_dict = {
                    0: 'TRAUMA',
                    1: 'LOSS',
                    2: 'MEDICAL ISSUE',
                    3: 'CHANGE IN LIVING',
                    4: 'CHANGE IN ABILITIES'
        }

    passing_score = 0.25

    # Detect topics in the text
    documents = input_df['text'].tolist()
    document = documents[0]    # Currently this endpoint expects a single text input

    # required if resp_output == 'phrase'
    sentences = extract_sentences(document)
    cl_sentences = preprocess(sentences)
    topic_inds = get_topic(cl_sentences)
    topics = [ind_topic_dict[i] for i in topic_inds]
    scores = get_topic_scores(cl_sentences)
    result_df = pd.DataFrame({'phrase': sentences, 'topic': topics, 'score': scores})
    predictions = result_df[result_df['score'] >= passing_score]

    # required if resp_output is either 'topic_agg' or 'topic_scores'
    final_passing = 0.0
    def topic_output(predictions, resp_output):
        agg_df = predictions.groupby('topic')['score'].sum()
        agg_df = agg_df.to_frame()
        agg_df.columns = ['Total Score']
        agg_df = agg_df.assign(
            score=lambda x: x['Total Score'] / x['Total Score'].sum()
        )
        agg_df = agg_df.sort_values(by='score', ascending=False)
        agg_df['topic'] = agg_df.index
        rem_topics = [ind_topic_dict[i] for i in range(0, 5) if not ind_topic_dict[i] in agg_df.topic.tolist()]
        if len(rem_topics) > 0:
            rem_agg_df = pd.DataFrame({'topic': rem_topics, 'score': 0.0, 'Total Score': 0.0})
            agg_df = pd.concat([agg_df, rem_agg_df])
        # Set the score column to 0 or 1 based on final_passing
        if resp_output == 'topic_scores':
            agg_df['score'] = [1 if score > final_passing else 0 for score in agg_df['score']]

        predictions = agg_df[['topic', 'score']]
        return predictions

    if resp_output != 'phrase':
        predictions = topic_output(predictions, resp_output)
    
    # Transform predictions to JSON
    result = {'output': []}
    list_out = predictions.to_dict(orient="records")
    result['output'] = list_out
    result = json.dumps(result)
    return flask.Response(response=result, status=200, mimetype='application/json')

@application.route('/topics_p1_likes', methods=['POST'])
def get_topics_p1_likes():
    # Get input JSON data and convert it to a DF
    input_json = flask.request.get_json()
    input_json = json.dumps(input_json['input'])
    input_df = pd.read_json(input_json,orient='list')

    # Get the query parameter value corresponsing to the output type: phrase | topic_agg | topic_scores
    resp_output = flask.request.args.get("output")

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

    #sentence extraction
    def extract_sentences(nltk_query):
        sentences = sent_tokenize(nltk_query)
        return sentences

    #query and get predicted topic
    def get_topic(sentences):
        preds = list(sf_p1_likes_model(sentences))
        return preds
    def get_topic_scores(sentences):
        preds = sf_p1_likes_model.predict_proba(sentences)
        preds = [max(list(x)) for x in preds]
        return preds

    # format output
    ind_topic_dict = {
                        0: 'LIKES',
                        1: 'DISLIKES',
                        2: 'NEUTRAL'
        }

    passing_score = 0.75

    # Detect topics in the text
    documents = input_df['text'].tolist()
    document = documents[0]    # Currently this endpoint expects a single text input

    # required if resp_output == 'phrase'
    sentences = extract_sentences(document)
    cl_sentences = preprocess(sentences)
    topic_inds = get_topic(cl_sentences)
    topics = [ind_topic_dict[i] for i in topic_inds]
    scores = get_topic_scores(cl_sentences)
    result_df = pd.DataFrame({'phrase': sentences, 'topic': topics, 'score': scores})
    predictions = result_df[(result_df['score'] >= passing_score) & (result_df['topic'] != 'NEUTRAL')]

    # required if resp_output is either 'topic_agg' or 'topic_scores'
    final_passing = 0.0
    def topic_output(predictions, resp_output):
        agg_df = predictions.groupby('topic')['score'].sum()
        agg_df = agg_df.to_frame()
        agg_df.columns = ['Total Score']
        agg_df = agg_df.assign(
            score=lambda x: x['Total Score'] / x['Total Score'].sum()
        )
        agg_df = agg_df.sort_values(by='score', ascending=False)
        agg_df['topic'] = agg_df.index
        rem_topics = [ind_topic_dict[i] for i in range(0, 2) if not ind_topic_dict[i] in agg_df.topic.tolist()]
        if len(rem_topics) > 0:
            rem_agg_df = pd.DataFrame({'topic': rem_topics, 'score': 0.0, 'Total Score': 0.0})
            agg_df = pd.concat([agg_df, rem_agg_df])
        # Set the score column to 0 or 1 based on final_passing
        if resp_output == 'topic_scores':
            agg_df['score'] = [1 if score > final_passing else 0 for score in agg_df['score']]

        predictions = agg_df[['topic', 'score']]
        return predictions

    if resp_output != 'phrase':
        predictions = topic_output(predictions, resp_output)
    
    # Transform predictions to JSON
    result = {'output': []}
    list_out = predictions.to_dict(orient="records")
    result['output'] = list_out
    result = json.dumps(result)
    return flask.Response(response=result, status=200, mimetype='application/json')

@application.route('/topics_p4_q1', methods=['POST'])
def get_topics_p4_q1():
    # Get input JSON data and convert it to a DF
    input_json = flask.request.get_json()
    input_json = json.dumps(input_json['input'])
    input_df = pd.read_json(input_json,orient='list')

    # Get the query parameter value corresponsing to the output type: phrase | topic_agg | topic_scores
    resp_output = flask.request.args.get("output")

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

    #sentence extraction
    def extract_sentences(paragraph):
        symbols = ['\\.', '!', '\\?', ';', ':', ',', '\\_', '\n', '\\-']
        pattern = '|'.join([f'{symbol}' for symbol in symbols])
        sentences = re.split(pattern, paragraph)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        return sentences

    #query and get predicted topic
    def get_topic(model, sentences):
        preds = list(model(sentences))
        return preds
    def get_topic_scores(model, sentences):
        preds = model.predict_proba(sentences)
        preds = [max(list(x)) for x in preds]
        return preds

    # format output
    ind_s_topic_dict = {
            0: 'NONE',
            1: 'SPECIFIC',
        }

    ind_m_topic_dict = {
            0: 'NONE',
            1: 'MEASURABLE',
        }

    ind_a_topic_dict = {
            0: 'NONE',
            1: 'ACHIEVABLE',
        }

    ind_r_topic_dict = {
            0: 'NONE',
            1: 'RELEVANT',
        }

    ind_t_topic_dict = {
            0: 'NONE',
            1: 'TIMELY',
        }

    passing_score = 0.70

    # Detect topics in the text
    documents = input_df['text'].tolist()
    document = documents[0]    # Currently this endpoint expects a single text input

    # required if resp_output == 'phrase'
    sentences = extract_sentences(document)
    cl_sentences = preprocess(sentences)
    #specific
    topic_inds = get_topic(model_s, cl_sentences)
    topics_s = [ind_s_topic_dict[i] for i in topic_inds]
    scores_s = get_topic_scores(model_s, cl_sentences)
    #measurable
    topic_inds = get_topic(model_m, cl_sentences)
    topics_m = [ind_m_topic_dict[i] for i in topic_inds]
    scores_m = get_topic_scores(model_m, cl_sentences)
    #achievable
    topic_inds = get_topic(model_a, cl_sentences)
    topics_a = [ind_a_topic_dict[i] for i in topic_inds]
    scores_a = get_topic_scores(model_a, cl_sentences)
    #relevant
    topic_inds = get_topic(model_r, cl_sentences)
    topics_r = [ind_r_topic_dict[i] for i in topic_inds]
    scores_r = get_topic_scores(model_r, cl_sentences)
    #timely
    topic_inds = get_topic(model_t, cl_sentences)
    topics_t = [ind_t_topic_dict[i] for i in topic_inds]
    scores_t = get_topic_scores(model_t, cl_sentences)
    all_sentences = sentences * 5
    topics = topics_s + topics_m + topics_a + topics_r + topics_t
    scores = scores_s + scores_m + scores_a + scores_r + scores_t
    result_df = pd.DataFrame({'phrase': all_sentences, 'topic': topics, 'score': scores})
    topic_list = ['SPECIFIC', 'MEASURABLE', 'ACHIEVABLE', 'RELEVANT', 'TIMELY']
    sub_result_df = result_df[(result_df['score'] >= passing_score) & (result_df['topic'].isin(topic_list))].reset_index(drop=True)
    # Identify duplicate phrases and keep the record with the highest score
    agg_df = sub_result_df.groupby(sub_result_df.phrase).max()
    agg_df['phrase'] = agg_df.index
    agg_df = agg_df.reset_index(drop=True)
    agg_df = agg_df.drop(columns=['topic'])
    predictions = pd.merge(sub_result_df, agg_df, 'inner', ['phrase', 'score'])

    # required if resp_output is either 'topic_agg' or 'topic_scores'
    final_passing = 0.0
    def topic_output(predictions, resp_output):
        agg_df = predictions.groupby('topic')['score'].sum()
        agg_df = agg_df.to_frame()
        agg_df.columns = ['Total Score']
        agg_df = agg_df.assign(
            score=lambda x: x['Total Score'] / x['Total Score'].sum()
        )
        agg_df = agg_df.sort_values(by='score', ascending=False)
        agg_df['topic'] = agg_df.index
        rem_topics = [x for x in topic_list if not x in agg_df.topic.tolist()]
        if len(rem_topics) > 0:
            rem_agg_df = pd.DataFrame({'topic': rem_topics, 'score': 0.0, 'Total Score': 0.0})
            agg_df = pd.concat([agg_df, rem_agg_df])
        # Set the score column to 0 or 1 based on final_passing
        if resp_output == 'topic_scores':
            agg_df['score'] = [1 if score > final_passing else 0 for score in agg_df['score']]

        predictions = agg_df[['topic', 'score']]
        return predictions

    if resp_output != 'phrase':
        predictions = topic_output(predictions, resp_output)
    
    # Transform predictions to JSON
    result = {'output': []}
    list_out = predictions.to_dict(orient="records")
    result['output'] = list_out
    result = json.dumps(result)
    return flask.Response(response=result, status=200, mimetype='application/json')

@application.route('/topics_p4_q2', methods=['POST'])
def get_topics_p4_q2():
    # Get input JSON data and convert it to a DF
    input_json = flask.request.get_json()
    input_json = json.dumps(input_json['input'])
    input_df = pd.read_json(input_json,orient='list')

    # Get the query parameter value corresponsing to the output type: phrase | topic_agg | topic_scores
    resp_output = flask.request.args.get("output")

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

    #sentence extraction
    def extract_sentences(nltk_query):
        sentences = sent_tokenize(nltk_query)
        return sentences

    #query and get predicted topic
    def get_topic(sentences):
        preds = list(model_q(sentences))
        return preds
    def get_topic_scores(sentences):
        preds = model_q.predict_proba(sentences)
        preds = [max(list(x)) for x in preds]
        return preds

    # format output
    ind_topic_dict = {
                0: 'PHYSICAL / MENTAL', 
                1: 'RELATIONSHIPS', 
                2: 'PERSONAL FULFILLMENT', 
                3: 'FINANCIAL STABILITY', 
                4: 'PURPOSE / IMPACT', 
                5: 'NO_GOAL'
        }

    passing_score = 0.70

    # Detect topics in the text
    documents = input_df['text'].tolist()
    document = documents[0]    # Currently this endpoint expects a single text input

    # required if resp_output == 'phrase'
    sentences = extract_sentences(document)
    cl_sentences = preprocess(sentences)
    topic_inds = get_topic(cl_sentences)
    topics = [ind_topic_dict[i] for i in topic_inds]
    scores = get_topic_scores(cl_sentences)
    result_df = pd.DataFrame({'phrase': sentences, 'topic': topics, 'score': scores})
    predictions = result_df[(result_df['score'] >= passing_score) & (result_df['topic'] != 'NO_GOAL')]

    # required if resp_output is either 'topic_agg' or 'topic_scores'
    final_passing = 0.0
    def topic_output(predictions, resp_output):
        agg_df = predictions.groupby('topic')['score'].sum()
        agg_df = agg_df.to_frame()
        agg_df.columns = ['Total Score']
        agg_df = agg_df.assign(
            score=lambda x: x['Total Score'] / x['Total Score'].sum()
        )
        agg_df = agg_df.sort_values(by='score', ascending=False)
        agg_df['topic'] = agg_df.index
        rem_topics = [ind_topic_dict[i] for i in range(0, 5) if not ind_topic_dict[i] in agg_df.topic.tolist()]
        if len(rem_topics) > 0:
            rem_agg_df = pd.DataFrame({'topic': rem_topics, 'score': 0.0, 'Total Score': 0.0})
            agg_df = pd.concat([agg_df, rem_agg_df])
        # Set the score column to 0 or 1 based on final_passing
        if resp_output == 'topic_scores':
            agg_df['score'] = [1 if score > final_passing else 0 for score in agg_df['score']]

        predictions = agg_df[['topic', 'score']]
        return predictions

    if resp_output != 'phrase':
        predictions = topic_output(predictions, resp_output)
    
    # Transform predictions to JSON
    result = {'output': []}
    list_out = predictions.to_dict(orient="records")
    result['output'] = list_out
    result = json.dumps(result)
    return flask.Response(response=result, status=200, mimetype='application/json')

@application.route('/topics_p4_q5', methods=['POST'])
def get_topics_p4_q5():
    # Get input JSON data and convert it to a DF
    input_json = flask.request.get_json()
    input_json = json.dumps(input_json['input'])
    input_df = pd.read_json(input_json,orient='list')

    # Get the query parameter value corresponsing to the output type: phrase | topic_agg | topic_scores
    resp_output = flask.request.args.get("output")

    def process_response(response):
        sentences = []
        topics = []
        scores = []
        lines = response.strip().split("\n")
        topic = "REPLACEMENT BEHAVIOUR"
        for line in lines:
            try:
                phrase = line.split("(Confidence Score:")[0].strip()
                score = float(line.split("(Confidence Score:")[1].strip().replace(")", ""))
                sentences.append(phrase)
                topics.append(topic)
                scores.append(score)
            except:
                pass
        result_df = pd.DataFrame({'phrase': sentences, 'topic': topics, 'score': scores})
        if len(result_df) > 0:
            result_df['phrase'] = result_df['phrase'].str.replace('\d+\.', '', regex=True)
            result_df['phrase'] = result_df['phrase'].str.replace('^\s', '', regex=True)
        return result_df
    
    def get_prompt(query):
        with open("data/p4_q5_prompt.txt", "r") as file:
            static_prompt = file.read()
        prompt = static_prompt.format(query=query)
        return prompt
    
    def get_response(prompt):
        response = openai.Completion.create(engine=deployment_id, 
                                        prompt=prompt, 
                                        temperature=1, 
                                        max_tokens=2048,
                                        top_p=0.5,
                                        frequency_penalty=0,
                                        presence_penalty=0,
                                        best_of=1,
                                        stop=None)
        text = response['choices'][0]['text']
        return text

    # format output
    passing_score = 0.75

    # Detect topics in the text
    documents = input_df['text'].tolist()
    document = documents[0]    # Currently this endpoint expects a single text input

    # required if resp_output == 'phrase'
    prompt = get_prompt(document)
    response = get_response(prompt)
    result_df = process_response(response)
    if len(result_df) > 0:
        predictions = result_df[result_df['score'] >= passing_score]
    else:
        predictions = pd.DataFrame({'phrase': [], 'topic': [], 'score': []})

    # required if resp_output is 'topic_scores'
    def topic_output(predictions):
        if len(predictions) > 0:
            predictions = pd.DataFrame({'topic': ["REPLACEMENT BEHAVIOUR"], 'score': [1.0]})
        else:
            predictions = pd.DataFrame({'topic': ["REPLACEMENT BEHAVIOUR"], 'score': [0.0]})
        return predictions

    if resp_output != 'phrase':
        predictions = topic_output(predictions)
    
    # Transform predictions to JSON
    result = {'output': []}
    list_out = predictions.to_dict(orient="records")
    result['output'] = list_out
    result = json.dumps(result)
    return flask.Response(response=result, status=200, mimetype='application/json')

# run the application.
if __name__ == "__main__":
    os.environ['SOCKET_TIMEOUT'] = '120'
    # Setting debug to True enables debug output. This line should be removed before deploying a production application.
    application.debug = True
    application.run()