# Author: Giorgos Myrianthous (BSc Computer Science, MSc Data Analytics | Machine Learning) 

from utils.dataset import DataSet
from utils.generate_test_splits import split
from os import path
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pylab as py
from scipy.sparse import hstack
from scipy.sparse import coo_matrix
from tqdm import tqdm
from scipy import sparse
import csv, random, numpy, score, os, re, nltk, scipy, gensim
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from langdetect import detect
from sklearn.ensemble import RandomForestClassifier

dataset = DataSet()
lemmatizer = nltk.WordNetLemmatizer()

# Get the bodies of training data points
def get_bodies(data):
	bodies = []
	for i in range(len(data)):
		bodies.append(dataset.articles[data[i]['Body ID']])	
	return bodies

# Get the headlines of training data points
def get_headlines(data):
	headlines = []
	for i in range(len(data)):
		headlines.append(data[i]['Headline'])
	return headlines

# Function for extracting tf-idf vectors (for both the bodies and the headlines).
def extract_tfidf(training_headlines, training_bodies, dev_headlines, dev_bodies, test_headlines, test_bodies):
	# Body vectorisation
	body_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english')#, max_features=1024)
	bodies_tfidf = body_vectorizer.fit_transform(training_bodies)

	# Headline vectorisation
	headline_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english')#, max_features=1024)
	headlines_tfidf = headline_vectorizer.fit_transform(training_headlines)

	# Tranform dev/test bodies and headlines using the trained vectorizer (trained on training data)
	bodies_tfidf_dev = body_vectorizer.transform(dev_bodies)
	headlines_tfidf_dev = headline_vectorizer.transform(dev_headlines)

	bodies_tfidf_test = body_vectorizer.transform(test_bodies)
	headlines_tfidf_test = headline_vectorizer.transform(test_headlines)

	# Combine body_tfdif with headline_tfidf for every data point. 
	training_tfidf = scipy.sparse.hstack([bodies_tfidf, headlines_tfidf])
	dev_tfidf = scipy.sparse.hstack([bodies_tfidf_dev, headlines_tfidf_dev])
	test_tfidf = scipy.sparse.hstack([bodies_tfidf_test, headlines_tfidf_test])

	return training_tfidf, dev_tfidf, test_tfidf

# Function for extracting features
# Feautres: 1) Word Overlap, 2) TF-IDF vectors, 3) Cosine similarity, 4) Word embeddings
def extract_features(train, dev, test):
	# Get bodies and headlines for dev and training data
	training_bodies = get_bodies(training_data)
	training_headlines = get_headlines(training_data)
	dev_bodies = get_bodies(dev_data)
	dev_headlines = get_headlines(dev_data)
	test_bodies = get_bodies(test_data)
	test_headlines = get_headlines(test_data)

	# Extract tfidf vectors
	print("\t-Extracting tfidf vectors..")
	training_tfidf, dev_tfidf, test_tfidf = extract_tfidf(training_headlines, training_bodies, dev_headlines, dev_bodies, test_headlines, test_bodies)

	return training_tfidf, dev_tfidf, test_tfidf

if __name__ == '__main__':
	##############################################################################

	# Load the data
	print("\n[1] Loading data..")
	data_splits = split(dataset)

	# in the format: Stance, Headline, BodyID
	training_data = data_splits['training']
	dev_data = data_splits['dev']
	test_data = data_splits['test'] # currently 0 test points

	# Change the number of training examples used.
	N = int(len(training_data) * 1.0)
	training_data = training_data[:N]

	print("\t-Training size:\t", len(training_data))
	print("\t-Dev size:\t", len(dev_data))
	print("\t-Test data:\t", len(test_data))

	##############################################################################

	# Feature extraction
	print("[2] Extracting features.. ")
	training_features, dev_features, test_features = extract_features(training_data, dev_data, test_data)

	##############################################################################

	# Fitting model
	print("[3] Fitting model..")
	print("\t-Logistic Regression")

	#lr = LogisticRegression(C = 1.0, class_weight='balanced', solver="lbfgs", max_iter=150) 
	#lr = MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)
	lr = RandomForestClassifier(n_estimators=10, random_state=12345)

	targets_tr = [a['Stance'] for a in training_data]
	targets_dev = [a['Stance'] for a in dev_data]
	targets_test = [a['Stance'] for a in test_data]

	y_pred = lr.fit(training_features, targets_tr).predict(test_features)

	##############################################################################

	# Evaluation
	print("[4] Evaluating model..")
	score.report_score(targets_test, y_pred)
