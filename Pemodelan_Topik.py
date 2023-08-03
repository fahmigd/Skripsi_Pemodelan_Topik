import nltk
import re
import string
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic import BERTopic
import pandas as pd
import numpy as np
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import seaborn as sns

class Pemodelan_Topik:
    def __init__(self):
        self.data = None
        self.tweets = None
        self.BERTopic_model = None
        self.jml_topik = None
        self.average_coherence_score = None
        self.coherence_score_topics = None
        self.min_coherence_score = None
        self.min_coherence_topic_id = None
        self.min_coherence_daftar_kata = None
        self.max_coherence_score = None
        self.max_coherence_topic_id = None
        self.max_coherence_daftar_kata = None
    
    def load_data(self, uploaded_file):
        # Use pandas to read the uploaded file
        self.data = pd.read_csv(uploaded_file, engine='python')
    
    def preprocess_text(self, text):
        # mengubah tweet menjadi huruf kecil
        text = text.lower()
        # menghilangkan url
        text = re.sub(r'https?:\/\/\S+','',text)
        # menghilangkan mention, link, hastag
        text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
        #menghilangkan karakter byte (b')
        text = re.sub(r'(b\'{1,2})',"", text)
        # menghilangkan yang bukan huruf
        text = re.sub('[^a-zA-Z]', ' ', text)
        # menghilangkan digit angka
        text = re.sub(r'\d+', '', text)
        #menghilangkan tanda baca
        text = text.translate(str.maketrans("","",string.punctuation))
        # menghilangkan whitespace berlebih
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def preprocess_tweets(self):
        self.tweets = self.data['Tweet'].apply(self.preprocess_text).to_list()
    
    def load_pemodelan_topik(self):
        self.BERTopic_model = BERTopic.load("my_model")

    def transform_pemodelan_topik(self):
        # Fit the model
        topics, probs = self.BERTopic_model.transform(self.tweets)
        return topics
    
    def evaluate_pemodelan_topik(self):
        topics = self.transform_pemodelan_topik()
        self.tweets = self.BERTopic_model._preprocess_text(self.tweets)
        # Extract vectorizer and tokenizer from BERTopic
        vectorizer = self.BERTopic_model.vectorizer_model
        tokenizer = vectorizer.build_tokenizer()

        # Extract features for Topic Coherence evaluation
        words = vectorizer.get_feature_names_out()
        tokens = [tokenizer(doc) for doc in self.tweets]
        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(token) for token in tokens]
        topic_words = [[words for words, _ in self.BERTopic_model.get_topic(topic)]
                    for topic in range(len(set(topics))-1)]

        # Evaluate
        coherence_model_cv = CoherenceModel(topics=topic_words,
                                texts=tokens,
                                corpus=corpus,
                                dictionary=dictionary,
                                coherence='c_v')

        self.average_coherence_score = coherence_model_cv.get_coherence()

        topic_coherence_cv = coherence_model_cv.get_coherence_per_topic(segmented_topics=None, with_std=False, with_support=False)

        x = len(topic_words)
        y = len(topic_words[0])
        z = len(topic_words[0][0])

        daftar_topik = []
        daftar_kata_arr = []
        for i in range(x):
            daftar_kata = ""
            for j in range(y):
                o = topic_words[i][j]
                daftar_kata_arr.append(o)
            daftar_kata = "_".join(daftar_kata_arr)
            daftar_topik.append(daftar_kata)
            daftar_kata_arr.clear()

        topic_id = []
        topic_coherence = []
        # topic_words = []

        i = 0
        for topics_coherence in topic_coherence_cv:
            topic_id.append(i)
            topic_coherence.append(topics_coherence)
            i += 1

        self.coherence_score_topics = pd.DataFrame(
                    {
                        "Topic": topic_id,
                        "Daftar_Kata": daftar_topik,
                        "Cohrence Score CV": topic_coherence,
                    }
                )
        id_min = topic_coherence.index(min(topic_coherence))
        id_max = topic_coherence.index(max(topic_coherence))
        self.jml_topik = len(topic_id)+1

        self.min_coherence_topic_id = topic_id[id_min]
        self.min_coherence_score = topic_coherence[id_min]
        self.min_coherence_daftar_kata = daftar_topik[id_min]
        self.max_coherence_topic_id = topic_id[id_max]
        self.max_coherence_score = topic_coherence[id_max]
        self.max_coherence_daftar_kata = daftar_topik[id_max]
        