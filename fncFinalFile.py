# Importing all the libraries
from utils.dataset import DataSet
from utils.generate_test_splits import split, comp_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import hstack
from tqdm import tqdm
from scipy import sparse
import numpy, score, os, re, nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pickle
from bert_serving.client import BertClient
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, Activation
from keras import optimizers
import scipy
import numpy as np
from sklearn import feature_extraction


_wnl = nltk.WordNetLemmatizer()

# Defining Functions for Pre-Processing
###############################################################################################################################################
def normalize_word(w):
    #lemmatize the words
    return _wnl.lemmatize(w).lower()


def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]


def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric
    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()


def remove_stopwords(l):
    # Removes stopwords from a list of tokens
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]


def gen_or_load_feats(feat_fn, headlines, bodies, dataset_type):
    #Calls the feature function with headlines and bodies
    feats = feat_fn(headlines, bodies, dataset_type)
    return feats
###################################################################################################################################################

# Defining Functions for Feature Creation
###################################################################################################################################################

# Define: This feature finds the refuting_words in sentences and create a sparse matrix for that
def refuting_features(headlines, bodies, dataset_type):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        # 'refute',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_headline = get_tokenized_lemmas(clean_headline)
        features = [1 if word in clean_headline else 0 for word in _refuting_words]
        X.append(features)

    X_sparse = scipy.sparse.coo_matrix(numpy.array(X))
    return X_sparse

# Define: bert feature vector of constant length from string with headline and body concatenated as given in bert documentation
def bert_features(headlines, bodies, dataset_type):


    #bert service needs to be running to get the features
    data_combined = []
    bc = BertClient(check_length=False)
    print("Starting bert train")

    #Saving and loading data if already saved
    if not os.path.isfile("saved_concatenated_headline_body_text_"+dataset_type):
    
        for sen in range(0 , len (headlines)):
            
            hl = headlines[sen]
            bdy = bodies[sen]
            concat_hl_bd = hl + ' ||| ' + bdy
            data_combined.append(concat_hl_bd)

        file = open('saved_concatenated_headline_body_text_'+dataset_type, 'wb')
        pickle.dump(data_combined, file)
        file.close()

    else:
        
        file = open('saved_concatenated_headline_body_text_'+dataset_type, 'rb')
        data_combined = pickle.load(file)
        file.close()

    
    if not os.path.isfile("saved_bert_combined_features_"+dataset_type):
    
        feat_final= bc.encode(data_combined)
        print("Saved bert train")
        file = open('saved_bert_combined_features_'+dataset_type, 'wb')
        pickle.dump(feat_final, file)
        file.close()


    else:
        
        file = open('saved_bert_combined_features_'+dataset_type, 'rb')
        feat_final = pickle.load(file)
        file.close()

    
    
    bert_feat_sparse = scipy.sparse.coo_matrix(numpy.array(feat_final))
    
    return bert_feat_sparse

#Define: This function calculates polarity within sentences
def polarity_features(headlines, bodies, dataset_type):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]

    def calculate_polarity(text):
        tokens = get_tokenized_lemmas(text)
        return sum([t in _refuting_words for t in tokens]) % 2
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        features = []
        features.append(calculate_polarity(clean_headline))
        features.append(calculate_polarity(clean_body))
        X.append(features)
    X_sparse = scipy.sparse.coo_matrix(numpy.array(X))
    return X_sparse

# Define: This function calculates the number of words in a sentence
def ngrams(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output

# Define: This function calculates the number of characters in a word/sentence
def chargrams(input, n):
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output

# Define: This function appends the characters as obtained from chargrams function
def append_chargrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in chargrams(" ".join(remove_stopwords(text_headline.split())), size)]
    grams_hits = 0
    grams_early_hits = 0
    grams_first_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
        if gram in text_body[:100]:
            grams_first_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    features.append(grams_first_hits)
    return features

# Define: This function appends the ngrams as obtained from ngrams function
def append_ngrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in ngrams(text_headline, size)]
    grams_hits = 0
    grams_early_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    return features

# Define: Get the occurances of a word
def hand_features(headlines, bodies, dataset_type):

    def binary_co_occurence(headline, body):
        # Count how many times a token in the title
        # appears in the body text.
        bin_count = 0
        bin_count_early = 0
        for headline_token in clean(headline).split(" "):
            if headline_token in clean(body):
                bin_count += 1
            if headline_token in clean(body)[:255]:
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def binary_co_occurence_stops(headline, body):
        # Count how many times a token in the title
        # appears in the body text. Stopwords in the title
        # are ignored.
        bin_count = 0
        bin_count_early = 0
        for headline_token in remove_stopwords(clean(headline).split(" ")):
            if headline_token in clean(body):
                bin_count += 1
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def count_grams(headline, body):
        clean_body = clean(body)
        clean_headline = clean(headline)
        features = []
        features = append_chargrams(features, clean_headline, clean_body, 2)
        features = append_chargrams(features, clean_headline, clean_body, 8)
        features = append_chargrams(features, clean_headline, clean_body, 4)
        features = append_chargrams(features, clean_headline, clean_body, 16)
        features = append_ngrams(features, clean_headline, clean_body, 2)
        features = append_ngrams(features, clean_headline, clean_body, 3)
        features = append_ngrams(features, clean_headline, clean_body, 4)
        features = append_ngrams(features, clean_headline, clean_body, 5)
        features = append_ngrams(features, clean_headline, clean_body, 6)
        return features

    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        X.append(binary_co_occurence(headline, body)
                 + binary_co_occurence_stops(headline, body)
                 + count_grams(headline, body))

    X_sparse = scipy.sparse.coo_matrix(numpy.array(X))
    return X_sparse

# Get the bodies of data points
def get_bodies(data):
    bodies = []
    for i in range(len(data)):
        bodies.append(dataset.articles[data[i]['Body ID']])
    return bodies

# Get the bodies of competition data points
def get_comp_bodies(data):
	bodies = []
	for i in range(len(data)):
		bodies.append(comp_dataset.articles[data[i]['Body ID']])
	return bodies



# Get the headlines of training data points
def get_headlines(data):
    headlines = []
    for i in range(len(data)):
        headlines.append(data[i]['Headline'])
    return headlines

# Tokenisation, Normalisation, removing Non-alphanumeric, Stemming & Lemmatization
def preprocess(string):
    # to lowercase, non-alphanumeric removal
    step1 = " ".join(re.findall(r'\w+', string, flags=re.UNICODE)).lower()
    step2 = [lemmatizer.lemmatize(t).lower() for t in nltk.word_tokenize(step1)]

    return step2


# Function for getting intersection over union
def extract_word_overlap(headlines, bodies, dataset_type):
    word_overlap = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        preprocess_headline = preprocess(headline)
        preprocess_body = preprocess(body)
        features = len(set(preprocess_headline).intersection(preprocess_body)) / float(len(set(preprocess_headline).union(preprocess_body)))
        word_overlap.append(features)

        # Convert the list to a sparse matrix (in order to concatenate the cos sim with other features)
        word_overlap_sparse = scipy.sparse.coo_matrix(numpy.array(word_overlap))

    return word_overlap_sparse

# Function for getting tf idf feature vectors .
def extract_tfidf(training_headlines, training_bodies, dev_headlines, dev_bodies, test_headlines, test_bodies):
    # Body vectorisation
    body_vect = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english')
    body_tfidf = body_vect.fit_transform(training_bodies)

    # Headline
    headline_vect = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english')
    headlines_tfidf = headline_vect.fit_transform(training_headlines)

    # Tranform  bodies and headlines using the trained vectorizer (this was trained on training data so as to avoid data leak)
    bodies_tfidf_dev = body_vect.transform(dev_bodies)
    headlines_tfidf_dev = headline_vect.transform(dev_headlines)

    bodies_tfidf_test = body_vect.transform(test_bodies)
    headlines_tfidf_test = headline_vect.transform(test_headlines)

    # Combine body_tf idf and headline_tf idf
    training_tfidf = scipy.sparse.hstack([body_tfidf, headlines_tfidf])
    dev_tfidf = scipy.sparse.hstack([bodies_tfidf_dev, headlines_tfidf_dev])
    test_tfidf = scipy.sparse.hstack([bodies_tfidf_test, headlines_tfidf_test])

    return training_tfidf, dev_tfidf, test_tfidf

# This extracts the cos similarity between body and headline
def extract_cosine_similarity(headlines, bodies, dataset_type):

    vectorizer = TfidfVectorizer(ngram_range=(1,2), lowercase=True, stop_words='english')

    cos_similarity_feats = []
    for itr in range(0, len(bodies)):

        body_headline = []
        body_headline.append(bodies[itr])
        body_headline.append(headlines[itr])
        tfidf = vectorizer.fit_transform(body_headline)

        cosine_similarity = (tfidf * tfidf.T).A
        cos_similarity_feats.append(cosine_similarity[0][1])

    cos_sim_array = scipy.sparse.coo_matrix(numpy.array(cos_similarity_feats))

    return cos_sim_array

# Function for counting words
def extract_word_counts(headlines, bodies):
    word_counts = []
    for i in range(0, len(headlines)):
        features = []
        features.append(len(headlines[i].split(" ")))
        features.append(len(bodies[i].split(" ")))
        word_counts.append(features)
    word_counts_array = scipy.sparse.coo_matrix(numpy.array(word_counts))

    return word_counts_array


#This combines the features ndarrays for different type of features
def combine_features(tfidf_vectors, cosine_similarity, word_overlap, refuting, polarity, hand):
    combined_features =  sparse.bmat([[tfidf_vectors, word_overlap.T, cosine_similarity.T, refuting, polarity, hand]])
    return combined_features


#This call different feature computing functions to get features to train and test
def extract_features(train, dev, test):
    training_bodies = get_bodies(training_data)
    training_headlines = get_headlines(training_data)
    dev_bodies = get_bodies(dev_data)
    dev_headlines = get_headlines(dev_data)
    test_bodies = get_comp_bodies(test_data)
    test_headlines = get_headlines(test_data)

    #print("\t-Extracting BERT features..")
    #X_bert_feats_training = gen_or_load_feats(bert_features, training_headlines, training_bodies, "training")
    #X_bert_feats_dev = gen_or_load_feats(bert_features, dev_headlines, dev_bodies, "dev")
    #X_bert_feats_test = gen_or_load_feats(bert_features, test_headlines, test_bodies, "test")
    
    print("\t-Extracting refuting features..")
    X_refuting_training = gen_or_load_feats(refuting_features, training_headlines, training_bodies, "training")
    X_refuting_dev = gen_or_load_feats(refuting_features, dev_headlines, dev_bodies, "dev")
    X_refuting_test = gen_or_load_feats(refuting_features, test_headlines, test_bodies, "test")
    
    print("\t-Extracting polarity vectors..")
    X_polarity_training = gen_or_load_feats(polarity_features, training_headlines, training_bodies, "training")
    X_polarity_dev = gen_or_load_feats(polarity_features, dev_headlines, dev_bodies, "dev")
    X_polarity_test = gen_or_load_feats(polarity_features, test_headlines, test_bodies, "test")

    print("\t-Extracting hand feature vectors..")
    X_hand_training = gen_or_load_feats(hand_features, training_headlines, training_bodies, "training")
    X_hand_dev = gen_or_load_feats(hand_features, dev_headlines, dev_bodies, "dev")
    X_hand_test = gen_or_load_feats(hand_features, test_headlines, test_bodies, "test")

    print("\t-Extracting tfidf vectors..")
    training_tfidf, dev_tfidf, test_tfidf = extract_tfidf(training_headlines, training_bodies, dev_headlines, dev_bodies, test_headlines, test_bodies)

    print("\t-Extracting word overlap..")
    training_overlap = extract_word_overlap(training_headlines, training_bodies, "training")
    dev_overlap = extract_word_overlap(dev_headlines, dev_bodies, "dev")
    test_overlap = extract_word_overlap(test_headlines, test_bodies, "test")

    print("\t-Extracting cosine similarity..")
    training_cos = extract_cosine_similarity(training_headlines, training_bodies, "training")
    dev_cos = extract_cosine_similarity(dev_headlines, dev_bodies, "dev")
    test_cos = extract_cosine_similarity(test_headlines, test_bodies, "test")

    print("\t-Combining features")
    training_features = combine_features(training_tfidf, training_cos, training_overlap, X_refuting_training, X_polarity_training, X_hand_training)
    dev_features = combine_features(dev_tfidf, dev_cos, dev_overlap, X_refuting_dev, X_polarity_dev, X_hand_dev)
    test_features = combine_features(test_tfidf, test_cos, test_overlap, X_refuting_test, X_polarity_test, X_hand_test)

    #print("\t-Combining features")
    #training_features = combine_features(X_bert_feats_training, training_tfidf, training_cos, training_overlap)
    #dev_features = combine_features(X_bert_feats_dev, dev_tfidf, dev_cos, dev_overlap)
    #test_features = combine_features(X_bert_feats_test, test_tfidf, test_cos, test_overlap)

    return training_features, dev_features, test_features


if __name__ == '__main__':

    # Loading training and competition datasets
    dataset = DataSet()
    comp_dataset = DataSet("competition_test")
    lemmatizer = nltk.WordNetLemmatizer()

    print("\n[1] Loading data..")
    data_splits = split(dataset)
    training_data = data_splits['training']
    dev_data = data_splits['dev']
    test_data = comp_split(comp_dataset)

    N = int(len(training_data) * 1.0)
    training_data = training_data[:N]

    print("\t-Training size:\t", len(training_data))
    print("\t-Dev size:\t", len(dev_data))
    print("\t-Test data:\t", len(test_data))
    print("[2] Extracting features.. ")

    # Saving and loading saved features to save time and computations
    if not os.path.isfile("saved_train_feature") or not os.path.isfile("saved_train_feature") or not os.path.isfile(
            "saved_test_feature"):

        training_features, dev_features, test_features = extract_features(training_data, dev_data, test_data)
        file = open('saved_train_feature', 'wb')
        pickle.dump(training_features, file)
        file.close()

        file = open('saved_dev_feature', 'wb')
        pickle.dump(dev_features, file)
        file.close()

        file = open('saved_test_feature', 'wb')
        pickle.dump(test_features, file)
        file.close()

    else:

        file = open('saved_train_feature', 'rb')
        training_features = pickle.load(file)
        file.close()

        file = open('saved_dev_feature', 'rb')
        dev_features = pickle.load(file)
        file.close()

        file = open('saved_test_feature', 'rb')
        test_features = pickle.load(file)
        file.close()

    # Fitting model
    print("[3] Fitting model..")

    #Getting stance vectors
    targets_tr = [a['Stance'] for a in training_data]
    targets_dev = [a['Stance'] for a in dev_data]
    targets_test = [a['Stance'] for a in test_data]

    if not os.path.isfile("logistic") or not os.path.isfile("randomForest") or not os.path.isfile("MNB"):

        #Getting different types of models so that ensemble can be applied later
        lr = LogisticRegression(C=1.0, class_weight='balanced', solver="lbfgs", max_iter=400)
        lr1 = RandomForestClassifier(n_estimators=10, random_state=12345)
        lr2 = MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)
        y_pred_logistic = lr.fit(training_features, targets_tr).predict(test_features)  # Logistic
        y_pred_random = lr1.fit(training_features, targets_tr).predict(test_features)  # Random Forest
        y_pred_mnb = lr2.fit(training_features, targets_tr).predict(test_features)  # Multinomial
        file = open('logistic', 'wb')
        pickle.dump(y_pred_logistic, file)
        file.close()

        file = open('randomForest', 'wb')
        pickle.dump(y_pred_random, file)
        file.close()

        file = open('MNB', 'wb')
        pickle.dump(y_pred_mnb, file)
        file.close()

    else:

        file = open('logistic', 'rb')
        y_pred_logistic = pickle.load(file)
        file.close()

        file = open('randomForest', 'rb')
        y_pred_random = pickle.load(file)
        file.close()

        file = open('logistic', 'rb')
        y_pred_mnb = pickle.load(file)
        file.close()

    # For Training Target

    for i in range(len(targets_tr)):
        if targets_tr[i] == 'agree':
            targets_tr[i] = 0
        elif targets_tr[i] == 'disagree':
            targets_tr[i] = 1
        elif targets_tr[i] == 'discuss':
            targets_tr[i] = 2
        else:
            targets_tr[i] = 3

    # For Dev Target
    for i in range(len(targets_dev)):
        if targets_dev[i] == 'agree':
            targets_dev[i] = 0
        elif targets_dev[i] == 'disagree':
            targets_dev[i] = 1
        elif targets_dev[i] == 'discuss':
            targets_dev[i] = 2
        else:
            targets_dev[i] = 3

    # For Test Target
    for i in range(len(targets_test)):
        if targets_test[i] == 'agree':
            targets_test[i] = 0
        elif targets_test[i] == 'disagree':
            targets_test[i] = 1
        elif targets_test[i] == 'discuss':
            targets_test[i] = 2
        else:
            targets_test[i] = 3

    stance_dict = {
        0: "agree",
        1: "disagree",
        2: "discuss",
        3: "unrelated"
    }


    # Deep Neural Network
    X_train_sparse = sparse.csr_matrix(training_features)
    X_dev_sparse = sparse.csr_matrix(dev_features)
    X_test_sparse = sparse.csr_matrix(test_features)

    model = Sequential()
    model.add(Dense(512, input_shape=(199291,)))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Dropout(0.2))
    model.add(Dense(16))
    model.add(Dropout(0.2))
    model.add(Dense(4))
    model.add(Activation('softmax'))

    learning_rate = 0.006  # Defining the learning rate
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=learning_rate, beta_1=0.65, beta_2=0.75, epsilon=1e-8, decay=0.07,
                                            amsgrad=False), metrics=['accuracy'])


    def batch_generator(X, y, batch_size):
        n_batches_for_epoch = X.shape[0] // batch_size
        for i in range(n_batches_for_epoch):
            index_batch = range(X.shape[0])[batch_size * i:batch_size * (i + 1)]
            X_batch = X[index_batch, :].todense()
            y_batch = np.array(y)[index_batch]
            yield (np.array(X_batch), y_batch)


    if not os.path.isfile("neural_model_2.h5"):
        model.fit_generator(generator=batch_generator(X_train_sparse, targets_tr, 1000),
                            validation_data=(X_dev_sparse, targets_dev),
                            nb_epoch=5,
                            samples_per_epoch=X_train_sparse.shape[0] / 6000)
        model.save("neural_model_2.h5")


    else:
        model = load_model("neural_model_2.h5")

    print("Predicting")
    if not os.path.isfile("neural_pred"):

        y_pred = model.predict(X_test_sparse)
        file = open('neural_pred', 'wb')
        pickle.dump(y_pred, file)
        file.close()

    else:

        file = open('neural_pred', 'rb')
        y_pred = pickle.load(file)
        file.close()

    stance_dict = {
        0: "agree",
        1: "disagree",
        2: "discuss",
        3: "unrelated"
    }

    # Code for ensembling results based on voting criteria
    final_prediction = [stance_dict[np.argmax(val)] for val in y_pred]  # Neural Network
    final = []
    for i in range(len(final_prediction)):

        voting_result = []
        temp_count = 0
        temp_res = ""
        voting_result.append(final_prediction[i])
        voting_result.append(y_pred_logistic[i])
        voting_result.append(y_pred_random[i])
        voting_result.append(y_pred_mnb[i])
        if (voting_result.count("agree") > temp_count):
            temp_count = voting_result.count("agree")
            temp_res = "agree"
        if (voting_result.count("disagree") > temp_count):
            temp_count = voting_result.count("disagree")
            temp_res = "disagree"

        if (voting_result.count("discuss") > temp_count):
            temp_count = voting_result.count("discuss")
            temp_res = "discuss"

        if (voting_result.count("unrelated") > temp_count):
            temp_count = voting_result.count("unrelated")
            temp_res = "unrelated"
        final.append(temp_res)

    #saving the predicted stances for submission purpose
    np.savetxt("stance.csv", np.array(final), delimiter="\n", fmt='%s')
    targets_test = [a['Stance'] for a in test_data]
    # Evaluation
    print("[4] Evaluating model..")

    score.report_score(targets_test, final)
