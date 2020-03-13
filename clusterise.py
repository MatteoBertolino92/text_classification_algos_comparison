from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import os

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    var = True;
    classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    #if metrics.accuracy_score(predictions, valid_y) < 0.7:
    #print (metrics.accuracy_score(predictions, valid_y)+ " " + predictions + " " + valid_y)
    return metrics.accuracy_score(predictions, valid_y)

data = open('/home/tobi/Desktop/Cybmap/AbstractsProject/WOS46985/X.txt').read()
rawClusters = open('/home/tobi/Desktop/Cybmap/AbstractsProject/WOS46985/YL1.txt').read()    #WOS11967

labels, texts = [], []
for i, line in enumerate(data.split("\n")):
    content = line.split()
    #texts.append(" ".join(content[1:]))
    texts.append(content)

for i, line in enumerate(rawClusters.split("\n")):
    content = line.split()
    labels.append(content)

print(len(texts))
print(len(labels))

#nt = []
#i=0
#for element in texts:
#    if i<=46985:
#        nt.append(texts[i])
#        i=i+1
#flat_texts  = [item for sublist in texts for item in sublist]
#texts.clear()
#for element in nt:
#    texts.append(element)


flat_texts = []

for i in range (len(texts)):
    tmp = ''
    for value in texts[i]:
        tmp+=value
        tmp+=' '
    flat_texts.append(tmp)

print(len(flat_texts))

flag = False
if flag:
    import nltk
    import re
    from nltk.stem.wordnet import WordNetLemmatizer
    lem = WordNetLemmatizer()
    newtexts = []
    stopwords = []
    #type(texts) is list
    for x in flat_texts:
        tokens = nltk.word_tokenize(x)
        tagged = nltk.pos_tag(tokens)
        newText = x
        for el in tagged:
            if re.match(r'^SY', el[1]):
            #if  re.match(r'^CC', el[1]) or re.match(r'^I', el[1])  or re.match(r'^SY', el[1]):
                stopwords.append(el[0])
        newText = ' '.join( [lem.lemmatize(word.lower(), "v") for word in re.split("\W+", x) if word not in stopwords]) #newText = ' '.join([word for word in re.split("\W+", x) if word not in stopwords]) #newText = ''.join( lem.lemmatize(x, "v"))
        newtexts.append(newText)
        stopwords = []
        newText = ""

if flag == False:
    newtexts=flat_texts

trainDF = pandas.DataFrame()
flat_labels = [item for sublist in labels for item in sublist]
trainDF['text'] = newtexts
trainDF['label'] = flat_labels


# split the dataset into training and validation datasets
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])

# label encode the target variable
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

#########################################################################################################################
#########################################################################################################################
                                                    #FEATURES
#########################################################################################################################
#########################################################################################################################
############################################### COUNT VECTOR #########################################################
# create a count vectorizer object
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['text'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)
#########################################################################################################################

############################################### TF-IDF #########################################################

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(trainDF['text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

# ngram level tf-idf
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(trainDF['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x)
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)


#########################################################################################################################
#########################################################################################################################
                                                    #MODELLI
#########################################################################################################################
#########################################################################################################################

#################################################### NAIVE BAYES CLASSIFIER #####################################################################
# Naive Bayes on Count Vectors
accuracyCountBAYES = round(100*train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count), 2)
# Naive Bayes on Word Level TF IDF Vectors
accuracyWordBAYES = round(100*train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf), 2)
# Naive Bayes on Ngram Level TF IDF Vectors
accuracyNGramBAYES = round(100*train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram), 2)
# Naive Bayes on Character Level TF IDF Vectors
accuracyCharBAYES = round(100*train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars), 2)
#########################################################################################################################

#################################################### Lin. REGRESSION ######################################################
# Linear Classifier on Count Vectors
accuracyCountREGRESSION = round(100*train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count),2)
# Linear Classifier on Word Level TF IDF Vectors
accuracyWordREGRESSION = round(100*train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf), 2)
# Linear Classifier on Ngram Level TF IDF Vectors
accuracyNGramREGRESSION = round(100*train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram), 2)
# Linear Classifier on Character Level TF IDF Vectors
accuracyCharREGRESSION = round(100*train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars), 2)
###########################################################################################################################

#################################################### RF ######################################################
# RF on Count Vectors
accuracyRFCount = round(100*train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count), 2)
# RF on Word Level TF IDF Vectors
accuracyRFWord = round(100*train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf), 2)
###########################################################################################################################""

# Extreme Gradient Boosting on Count Vectors
accuracyXGBCount = round(100*train_model(xgboost.XGBClassifier(), xtrain_count.tocsc(), train_y, xvalid_count.tocsc()), 2)
# Extreme Gradient Boosting on Word Level TF IDF Vectors
accuracyXGBWord =round(100* train_model(xgboost.XGBClassifier(), xtrain_tfidf.tocsc(), train_y, xvalid_tfidf.tocsc()), 2)

print ("Naive Bayes classifier, Count Vectors Accuracy: ", accuracyCountBAYES, "%")
print ("Naive Bayes classifier, WordLevel TF-IDF Accuracy: ", accuracyWordBAYES, "%")
print ("Naive Bayes classifier, N-Gram Vectors Accuracy: ", accuracyNGramBAYES, "%")
print ("Naive Bayes classifier, CharLevel Vectors Accuracy: ", accuracyCharBAYES, "%")
print ("Lin. Regression classifier, Count Vectors Accuracy: ", accuracyCountREGRESSION, "%")
print ("Lin. Regression classifier, WordLevel TF-IDF Accuracy: ", accuracyWordREGRESSION, "%")
print ("Lin. Regression classifier, N-Gram Vectors Accuracy: ", accuracyNGramREGRESSION, "%")
print ("Lin. Regression classifier, CharLevel Vectors Accuracy: ", accuracyCharREGRESSION, "%")
print ("Random Forest, Count Vectors: ", accuracyRFCount, "%")
print ("Random Forest, WordLevel TF-IDF: ", accuracyRFWord, "%")
print ("Xgb, Count Vectors: ", accuracyXGBCount, "%")
print ("Xgb, WordLevel TF-IDF: ", accuracyXGBWord, "%")



#################################################### SHALLOW NN ######################################################
def create_model_architecture(input_size):
    # create input layer
    input_layer = layers.Input((input_size, ), sparse=True)
    # create hidden layer
    hidden_layer = layers.Dense(100, activation="relu")(input_layer)
    # create output layer
    output_layer = layers.Dense(1, activation="sigmoid")(hidden_layer)
    classifier = models.Model(inputs = input_layer, outputs = output_layer)
    classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    return classifier

classifier = create_model_architecture(xtrain_tfidf.shape[1])
accuracy = round(100*train_model(classifier, xtrain_tfidf, train_y, xvalid_tfidf, is_neural_net=True))
print ("NN, Ngram Level TF IDF Vectors",  accuracy)
###########################################################################################################################
from keras.layers import Embedding
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(flat_texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
vocab_size = len (word_index)+1
embedding_matrix = np.zeros((vocab_size, 300))
###################################################### CNN #####################################################################
def create_cnn():
    # Add an Input Layer
    input_layer = layers.Input((70, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the convolutional Layer
    conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)

    # Add the pooling Layer
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return model

classifier = create_cnn()
accuracy = train_model(classifier, train_x, train_y, valid_x, is_neural_net=True)
print ("CNN, Word Embeddings",  accuracy)
###########################################################################################################################


###################################################### RNN #####################################################################
def create_rnn_lstm():

    # Add an Input Layer
    input_layer = layers.Input((70, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the LSTM Layer
    lstm_layer = layers.LSTM(100)(embedding_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return model

classifier = create_rnn_lstm()
accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print ("RNN-LSTM, Word Embeddings",  accuracy)
###########################################################################################################################




###################################################### G-RNN #####################################################################

def create_rnn_gru():

    # Add an Input Layer
    input_layer = layers.Input((70, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the GRU Layer
    lstm_layer = layers.GRU(100)(embedding_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return model

classifier = create_rnn_gru()
accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print ("RNN-GRU, Word Embeddings",  accuracy)
###########################################################################################################################



###################################################### Bi-RNN #####################################################################

def create_bidirectional_rnn():

    # Add an Input Layer
    input_layer = layers.Input((70, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the LSTM Layer
    lstm_layer = layers.Bidirectional(layers.GRU(100))(embedding_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return model

classifier = create_bidirectional_rnn()
accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print ("RNN-Bidirectional, Word Embeddings",  accuracy)
###########################################################################################################################



###################################################### C-RNN #####################################################################

def create_rcnn():

    # Add an Input Layer
    input_layer = layers.Input((70, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the recurrent layer
    rnn_layer = layers.Bidirectional(layers.GRU(50, return_sequences=True))(embedding_layer)

    # Add the convolutional Layer
    conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)

    # Add the pooling Layer
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return model

classifier = create_rcnn()
accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print ("CNN, Word Embeddings",  accuracy)
###########################################################################################################################
