# Import general libraries
import pickle
import numpy as np
import json
import random

# Import WordNet for lemmatization
import nltk
from nltk.stem import WordNetLemmatizer

# Import tensorflow for loading the model
from tensorflow.keras.models import load_model

# Enable GPU for prediction faster
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# Import word embedding library
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim import models
w2v_model = models.KeyedVectors.load_word2vec_format('glove-wiki-gigaword-50.txt', binary=False)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the model
model = load_model('chatbot_model.h5')

# Load data from the pickle files
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

# Fuction calculate the error matching between two words
def error_matching(word, word2):
    word = list(word.lower())
    word2 = list(word2.lower())
    # calculate the number of different characters based on the length of the word or word2
    error = np.intersect1d(word, word2) # the characters that are in both words
    if len(word) > len(word2):
        return 1-len(error)/len(word)
    else:
        return 1-len(error)/len(word2)

# Function tokelize the sentence and lemmatize it
def clean_up_sentence(sentence):
    # tokenize the pattern - splitting words into array
    sentence_words = nltk.word_tokenize(sentence)

    # stemming every word - reducing to base form
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

    return sentence_words

# Function to generate vector of words
def bag_of_words(sentence, show_details=True):
    # tokenizing patterns
    sentence_words = clean_up_sentence(sentence)

    # bag of words - vocabulary matrix
    bag = [0]*len(words)

    # filling the bag with 1 for each word in the pattern
    for s in sentence_words:
            if (s in words):
                bag[words.index(s)] = 1
                if show_details:
                    print("found in bag (standard matching): %s" % words[words.index(s)])
            else:
                # try to find the best match for the word
                word_match = []

                for i, word in enumerate(words):
                    # if word is not in the vocabulary, try to calculate the error matching for matching the word
                    error = error_matching(word, s)
                    
                    if error <= 0.5:
                        word_match.append((word, error))
                    else:
                        # if error matching is not less than 0.5, try to calculate the similarity between two words
                        try:
                            similarity = w2v_model.similarity(word, s)
                            if similarity >= 0.5:
                                word_match.append((word, 1-similarity))
                        except:
                            pass
                
                # sort the word_match list by the second element of the tuple (i.e., the error matching)
                sort_word_match = sorted(word_match, key=lambda x: x[1])
                if sort_word_match:
                    bag[words.index(sort_word_match[0][0])] = 1
                    if show_details:
                        print("found in bag (advanced matching): %s" % sort_word_match[0][0])
    return np.array(bag)

# Define the predict_class function
def predict_class(sentence):
    # filter below  threshold predictions
    bow = bag_of_words(sentence)

    # model.predict will return probabilities
    res = model.predict(np.array([bow]))[0]

    # remove all predictions below threshold
    ERROR_THRESHOLD = 0.25

    # return tag with max probability
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # sorting strength probability
    results.sort(key=lambda x: x[1], reverse=True)

    # return list of intent and probability
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Define the get_response function
def getResponse(intents_list, intents_json):
    # if there is no intent in the intents_list, return the default response
    if len(intents_list) == 0:
        return "I am not sure what you mean."

    # get the tag with the highest probability
    tag = intents_list[0]['intent']

    # return a random response based on the intent
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

#Check on run and terminal
while True:
    message = input("You: ")
    ints = predict_class(message)
    res = getResponse(ints, intents)
    print("NickBot: {}".format(res))
