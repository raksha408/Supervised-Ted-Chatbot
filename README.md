# Supervised Ted Chatbot – Internship Project

##  Project Overview

During my internship, I built a simple chatbot that understands what the user wants and gives the correct reply.

The main problem I wanted to solve was this:

Computers normally understand only exact commands. But people speak in many different ways.  
For example, “Hi”, “Hello”, and “Namaste” all mean the same thing — but a basic program cannot understand that.

So I built a supervised machine learning model that learns from examples and predicts the correct intent behind a user’s message.

---

##  Problem Statement

To build a chatbot that can:

- Understand user input in different wordings  
- Classify the correct intent  
- Generate an appropriate response  

---

##  Dataset

I used a dataset called `intents.json`.

In this file:
- Each intent has a name (e.g., greeting, goodbye, help, etc.)
- Each intent contains example sentences
- Each intent also has possible responses

The model learns from these examples.

---

##  How the Chatbot Works

### 1️⃣ Text Preprocessing
- Cleaned the text
- Removed unnecessary characters
- Converted words into numerical form

### 2️⃣ Feature Conversion
- Converted text into numbers using TF-IDF vectorization

### 3️⃣ Model Training
- Used a supervised machine learning approach
- Trained a Neural Network (MLP model)
- Split data into:
  - 80% training
  - 20% testing

### 4️⃣ Evaluation
- Tested the model on unseen data
- Observed prediction accuracy
- Tested with new sentences not in dataset

After training, the chatbot could correctly predict user intent and generate appropriate responses.

---

##  What I Learned

Through this project, I learned:

- How supervised machine learning works
- Importance of clean and quality data
- Text preprocessing techniques
- Model evaluation and performance checking
- How chatbots understand user input

---

##  Technologies Used

- Python
- scikit-learn
- TensorFlow / Keras
- TF-IDF Vectorization
- JSON dataset
- Tkinter (for basic interface)

---

##  Future Improvements

- Improve conversational flow
- Add more intents and training data
- Deploy as a web-based chatbot
- Use advanced NLP models

---

##  Author

Shriraksha Kulkarni

---

 This project was built as part of my internship learning journey.
