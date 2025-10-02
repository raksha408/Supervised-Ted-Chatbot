import nltk
from nltk.stem import WordNetLemmatizer
import json
import numpy as np
import random

# Ensure required packages are downloaded
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load dataset
with open('intents.json', 'r') as file:
    data = json.load(file)

words = []
classes = []
documents = []

# Tokenizing words
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)  # Tokenizing the sentence
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))

        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lemmatize and clean words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w.isalpha()]
words = sorted(set(words))
classes = sorted(set(classes))

print("Words:", words)
print("Classes:", classes)
