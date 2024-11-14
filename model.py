import os.path
import re

import numpy as np
import pandas as pd
import spacy
from nltk import sent_tokenize
from gensim.models import Word2Vec
from sklearn import metrics
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss, make_scorer, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neural_network import MLPClassifier
import joblib
from sklearn.svm import SVC
from transformers import DistilBertTokenizer, DistilBertModel, RobertaTokenizer, RobertaModel
from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib.request
from tqdm import tqdm


def load_data():
    preprocessed_data = pd.read_csv('labeled_data.csv')
    return preprocessed_data.dropna(subset=['data'])


class Word2VecVectorizer:
    def __init__(self, size=300, window=5, min_count=1, workers=4):
        self.size = size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def fit_transform(self, data):
        sentences = [sentence.split() for sentence in data]
        self.model = Word2Vec(sentences, window=self.window, min_count=self.min_count, workers=self.workers, sg=1,
                              vector_size=self.size)
        return self.transform(data)

    def transform(self, data):
        sentences = [sentence.split() for sentence in data]
        vectors = []
        for sentence in sentences:
            sentence_vectors = [self.model.wv[word] for word in sentence if word in self.model.wv]
            if not sentence_vectors:
                vectors.append(np.zeros(self.size))
                continue
            # average of all word vectors in sentence
            averaged_vector = np.mean(sentence_vectors, axis=0)
            vectors.append(averaged_vector)
        return np.array(vectors)


class BertVectorizer:
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    def fit_transform(self, data):
        embeddings = []
        if not isinstance(data, list):
            data = data.tolist()
        for sentence in tqdm(data, desc='Vectorizing', unit='sentence'):
            inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model(**inputs)
            # extract values from last hidden state to use as input to a model
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy())
        return embeddings

    def transform(self, data):
        return self.fit_transform(data)


class RoBERTAVectorizer:
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
        self.model = RobertaModel.from_pretrained('distilroberta-base')

    def fit_transform(self, data):
        embeddings = []
        if not isinstance(data, list):
            data = data.tolist()
        for sentence in tqdm(data, desc='Vectorizing', unit='sentence'):
            inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model(**inputs)
            # extract values from last hidden state to use as input to a model
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy())
        return embeddings

    def transform(self, data):
        return self.fit_transform(data)


class SpacyVec:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')

    def fit_transform(self, data):
        features = np.array([sentence.vector.reshape(1, -1).flatten() for sentence in self.nlp.pipe(data)])
        return features

    def transform(self, data):
        return self.fit_transform(data)


class Classifier:
    def __init__(self, model, vectorizer=None):
        self.__vectorizer = vectorizer
        self.testVectorized = None
        self.trainVectorized = None
        self.preprocessedData = load_data()
        self.trainInput, self.testInput, self.trainLabels, self.testLabels = self.train_test_split()
        self.__model = model

    def train_test_split(self):
        return train_test_split(self.preprocessedData['data'], self.preprocessedData['label'],
                                test_size=0.2, random_state=42)

    def vectorize_input(self):
        if self.__vectorizer is None:
            self.__vectorizer = TfidfVectorizer()
            print('default vectorizer used')
        self.trainVectorized = self.__vectorizer.fit_transform(self.trainInput)
        self.testVectorized = self.__vectorizer.transform(self.testInput)

    def train_model(self):
        if self.trainVectorized is not None and self.trainLabels is not None and self.__model is not None:
            self.__model.fit(self.trainVectorized, self.trainLabels)

    def cross_validate(self, folds=5):
        scorer = make_scorer(f1_score, pos_label='chatgpt')
        self.trainVectorized = self.__vectorizer.fit_transform(self.trainInput)
        if self.trainVectorized is not None and self.trainLabels is not None and self.__model is not None:
            scores = cross_val_score(self.__model, self.trainVectorized, self.trainLabels, cv=folds, scoring=scorer)
            ret_str = ''
            for iter, score in enumerate(scores, start=1):
                ret_str += f'Fold {iter}: f1 = {score}\n'
            return ret_str
        else:
            return 'training data not vectorized'

    def dump_model_and_vectorizer(self, file_name):
        joblib.dump([self.__model, self.__vectorizer], file_name, compress=1)

    def load_model_and_vectorizer(self, file_name):
        self.__model, self.__vectorizer = joblib.load(file_name)

    def create_classification_report(self, digits=2):
        if self.testVectorized is None:
            self.testVectorized = self.__vectorizer.transform(self.testInput)
        predictions = self.__model.predict(self.testVectorized)
        return "Classification Report:\n" + metrics.classification_report(self.testLabels, predictions, digits=digits)

    def predict_new(self, text, is_split=False):
        input_str = None
        if not is_split:
            new_arr = []
            input_str = sent_tokenize(text)
            for sentence in input_str:
                sentence = sentence.strip('\n').lower()
                cleaned_input = re.sub(r'[^\w\s]', '', sentence)
                new_arr.append(cleaned_input)
            vectorized_input = self.__vectorizer.transform(new_arr)
        else:
            cleaned_input = text.strip('\n').lower()
            cleaned_input = re.sub(r'[^\w\s]', '', cleaned_input)
            vectorized_input = self.__vectorizer.transform(cleaned_input)
        return (self.__model.predict(vectorized_input), 1 - self.__model.predict_proba(vectorized_input),
                (text if input_str is None else input_str))

    def get_log_loss(self):
        if self.testVectorized is None:
            self.testVectorized = self.__vectorizer.transform(self.testInput)
        probs = self.__model.predict_proba(self.testVectorized)
        return log_loss(self.testLabels, probs)

    # gets all visible text from a webpage and puts it through predict new
    def predict_webpage(self, url):
        def tag_visible(element):
            if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
                return False
            if isinstance(element, Comment):
                return False
            return True

        bsoup = BeautifulSoup(urllib.request.urlopen(url).read(), 'html.parser')
        strings = bsoup.findAll(string=True)
        visible_text = filter(tag_visible, strings)
        text_for_model = [line.strip() for line in visible_text if len(line.split()) > 4]
        probs = []
        labels = []
        text = []
        for line in text_for_model:
            l, p, t = self.predict_new(line)
            probs.extend(p)
            labels.extend(l)
            text.extend(t)
        return labels, probs, text


# default pipeline for creating and training a new model
def setup_classifier(model, vectorizer=None):
    classifier = Classifier(model, vectorizer)
    classifier.vectorize_input()
    classifier.train_model()
    return classifier


# formatting for output of Classifier.predict_new
def print_predictions(predictions, probabs, text):
    for pred, probs, line in zip(predictions, probabs, text):
        if pred == 'human':
            print(f'Prediction: {pred}   => Probability: {probs[0]}: {line}')
        else:
            print(f'Prediction: {pred} => Probability: {probs[1]}: {line}')


w2v = Word2VecVectorizer()
spacy_vec = SpacyVec()
bert = BertVectorizer()
roberta = RoBERTAVectorizer()
tfidf = TfidfVectorizer(encoding="utf-8",
                        decode_error="strict",
                        strip_accents=None,
                        lowercase=True,
                        preprocessor=None,
                        tokenizer=None,
                        analyzer="word",
                        stop_words=None,
                        token_pattern=r"(?u)\b\w\w+\b",
                        ngram_range=(1, 2),
                        max_df=1.0,
                        min_df=1,
                        max_features=None,
                        vocabulary=None,
                        binary=False,
                        dtype=np.float64,
                        norm="l2",
                        use_idf=True,
                        smooth_idf=True,
                        sublinear_tf=False)
bow = CountVectorizer(input="content",
                      encoding="utf-8",
                      decode_error="strict",
                      strip_accents=None,
                      lowercase=True,
                      preprocessor=None,
                      tokenizer=None,
                      stop_words=None,
                      token_pattern=r"(?u)\b\w\w+\b",
                      ngram_range=(1, 1),
                      analyzer="word",
                      max_df=1.0,
                      min_df=1,
                      max_features=None,
                      vocabulary=None,
                      binary=False,
                      dtype=np.int64)

mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100, 100),
                    activation="relu",
                    solver="adam",
                    alpha=0.0001,
                    batch_size="auto",
                    learning_rate="constant",
                    learning_rate_init=0.001,
                    power_t=0.5,
                    max_iter=500,
                    shuffle=True,
                    random_state=18,
                    tol=1e-4,
                    verbose=True,
                    warm_start=False,
                    momentum=0.9,
                    nesterovs_momentum=True,
                    early_stopping=False,
                    validation_fraction=0.1,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-8,
                    n_iter_no_change=10,
                    max_fun=15000)
svc = SVC(C=1.0,
          kernel="poly",
          degree=3,
          gamma="scale",
          coef0=0.0,
          shrinking=True,
          probability=True,
          tol=1e-4,
          cache_size=200,
          class_weight=None,
          verbose=True,
          max_iter=-1,
          decision_function_shape="ovr",
          break_ties=False,
          random_state=18)

rforest = RandomForestClassifier(n_estimators=100,
                                 criterion="gini",
                                 max_depth=None,
                                 min_samples_split=2,
                                 min_samples_leaf=1,
                                 min_weight_fraction_leaf=0.0,
                                 max_features="sqrt",
                                 max_leaf_nodes=None,
                                 min_impurity_decrease=0.0,
                                 bootstrap=True,
                                 oob_score=False,
                                 n_jobs=2,
                                 random_state=18,
                                 verbose=True,
                                 warm_start=False,
                                 class_weight=None,
                                 ccp_alpha=0.0,
                                 max_samples=None)

mlp_eight = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100, 100, 100, 100),
                          verbose=True,
                          random_state=18,
                          max_iter=500,
                          )

mlp_bagged = BaggingClassifier(estimator=mlp_eight, n_jobs=5, random_state=18, verbose=True)

models = [(svc, 'svc'), (rforest, 'rforest'), (mlp_eight, 'mlp_eight')]
vectorizers = [(bow, 'bow'), (tfidf, 'tfidf_2'), (w2v, 'w2v'),
               (spacy_vec, 'svec'), (bert, 'bert'), (roberta, 'roberta')]

# main loop to train all combinations of models
for m in models:
    for v in vectorizers:
        model_name = m[1] + '_' + v[1] + '_final.model'
        if not os.path.isfile(model_name):
            print('Begin training ' + m[1] + ' with ' + v[1])
            classifier = setup_classifier(m[0], v[0])
            classifier.dump_model_and_vectorizer(model_name)
            print('Model Dumped')
            file_name = model_name.split('.')[0] + '.txt'
            file_str = (classifier.cross_validate() + f'\nLog loss = {classifier.get_log_loss()}\n'
                        + classifier.create_classification_report(digits=4))
            with open(file_name, 'w') as file:
                file.write(file_str)
        else:
            print(model_name, 'already exists')
