import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import random
import numpy

with open('model.pkl', 'rb') as file:
    loaded_model_pickle = file.read()
model = pickle.loads(loaded_model_pickle)
print(pickle.__version__)

st.title("Streamlit ML App")
user_input = st.text_input("Enter some data:")
word_size=330
import nltk
import json
with open ('C:\\Users\\ashutosh\\OneDrive\\Desktop\\Paritosh_learning_stuff\\Knowcode_hackathon\\database_large_new.json') as json_data:
    intents=json.load(json_data)
words = []
classes = []
documents = []
ignore = ['I,?']
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])    
def sentence_preprocessing(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [ps.stem(word.lower()) for word in sentence_words]
    return sentence_words
def bag_of_words(sentence, words):
    sentence_words = sentence_preprocessing(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
    bag=np.array(bag)
    return(bag)
ERROR_THRESHOLD = 0.10
def Predict(sentence):
    bag = bag_of_words(sentence, words)
    results = model.predict(np.array([bag]))
    results = [[i,r] for i,r in enumerate(results[0]) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list
    
def response(sentence):
    results = Predict(sentence)
    if results:
        while results:
            for i in intents['intents']:
                if i['tag'] == results[0][0]:
                    return print(random.choices(i['responses']))

            results.pop(0)
# Display results
if st.button("Predict"):
    # Perform prediction using the model
    a=response(user_input)
    st.write(f"Prediction:{a}")
