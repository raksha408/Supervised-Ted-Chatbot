import json
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import random

# Load the dataset
dataset_path = r"D:\Internship\Programs\Chatbot_recent\intents.json"

with open(dataset_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Initialize NLP tools
nltk.download('punkt')
stemmer = PorterStemmer()

# Prepare data
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

# Tokenize words and categorize intents
for intent in data['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Stemming and removing duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))
classes = sorted(set(classes))

# Create training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    pattern_words = document[0]
    pattern_words = [stemmer.stem(w.lower()) for w in pattern_words]

    for w in words:
        bag.append(1 if w in pattern_words else 0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])

# Shuffle and convert to NumPy arrays
random.shuffle(training)
training = np.array(training, dtype=object)
X = np.array(list(training[:, 0]))  # Features
y = np.array(list(training[:, 1]))  # Labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define MLP Model
model = Sequential([
    Dense(128, input_shape=(len(X_train[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

# Compile Model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")

# Save the Model and Data
model.save("chatbot_model.h5")
np.save("words.npy", words)
np.save("classes.npy", classes)

print("\nTraining complete. Model saved as 'chatbot_model.h5'.")
