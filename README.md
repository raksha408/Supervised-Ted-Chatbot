# Supervised Ted Chatbot
Ted! Chatbot is an intelligent conversational agent built in Python that can understand user queries, classify their intents, and generate contextually appropriate responses. It leverages **machine learning** for intent classification and **natural language processing (NLP)** techniques such as TF-IDF vectorization and text preprocessing. The chatbot includes a **Tkinter-based GUI** for an interactive, user-friendly chat experience, making it suitable for applications like customer support, personal assistants, or educational tools.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Folder Structure](#folder-structure)  
3. [Features](#features)  
4. [Technologies Used](#technologies-used)  
5. [Installation](#installation)  
6. [Usage](#usage)  
7. [Training Details](#training-details)  
8. [Results](#results)  
9. [Conclusion](#conclusion)  
10. [Future Work](#future-work)  
11. [Acknowledgements](#acknowledgements)  
12. [Author](#author)  

---

## Features

- **Intent Classification:** Recognizes user intents using a trained neural network.  
- **Dynamic Responses:** Provides varied responses for the same intent to make conversations more natural.  
- **Text Preprocessing:** Cleans and stems user input for better model predictions.  
- **Class Imbalance Handling:** Uses random oversampling to balance training data.  
- **TF-IDF Vectorization:** Converts textual input into numerical features for the model.  
- **Console & GUI Interfaces:** Offers both command-line and Tkinter-based graphical chat interfaces.  
- **Fallback Mechanism:** Provides a default response when the chatbot is unsure about the intent.  
- **Interactive Chat Bubbles:** Tkinter GUI displays user and bot messages in a visually appealing format.  
- **Trained Model Integration:** Supports using pre-trained models (`.h5`, `.pkl`) for quick deployment.


## Project Overview

Ted! Chatbot is a Python-based conversational agent capable of understanding user queries and responding appropriately. It leverages:

- **TF-IDF vectorization** for converting text into numeric features  
- **Neural network classification** for intent recognition  
- **Random oversampling** to handle class imbalance  
- **Tkinter GUI** for a friendly chat interface  

This project can serve as a foundation for customer support automation, personal assistants, or interactive learning tools.

---

# Folder Structure - Chatbot Project

**Supervised-Ted-Chatbot/**  
Main project directory containing all code, data, models, and results.

---

## 1. Data
**data/**  
Contains sample intent.json dataset. 

- `intents.json` → Sample dataset containing intents, patterns, and responses.  

---

## 2. Code
- `preprocess.py` → Script for text preprocessing (tokenization, stemming, cleaning).  
- `model.py` → Script to define and train the neural network model.  
- `app.py` → Main application script to run the chatbot (console or GUI).  
- `chatbot_model.h5` → Trained Keras model for predicting intents.  
- `vectorizer.pkl` → Saved TF-IDF vectorizer for transforming text inputs.  
- `label_encoder.pkl` → Label encoder mapping tags to numeric labels. 

---

## 3. Environment
**requirements.txt**  

- Python dependencies needed to run the chatbot project.  
- Example: `tensorflow`, `nltk`, `numpy`, `sklearn`, `imblearn`, `matplotlib`, `seaborn`, `tkinter` (comes with Python).  

---

## Features

- Understands and classifies user intents accurately  
- Provides dynamic, context-aware responses  
- Handles class imbalance with random oversampling  
- Offers both **console-based** and **GUI-based** chat interfaces  
- Easy to extend with new intents and responses  

---

## Technologies Used

- **Python 3.10+**  
- **TensorFlow / Keras** for neural network modeling  
- **NLTK** for text preprocessing and tokenization  
- **Scikit-learn** for TF-IDF vectorization and label encoding  
- **Imbalanced-learn** for oversampling  
- **Tkinter** for GUI development  
- **Matplotlib & Seaborn** for evaluation and plotting  

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Supervised-Ted-Chatbot.git
cd Supervised-Ted-Chatbot

  ```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

  ```

3. Install dependencies:

```bash
pip install -r requirements.txt

  ```
4. Ensure the trained model (chatbot_model.h5), vectorizer (vectorizer.pkl), and encoder (label_encoder.pkl) are in the data/ folder.

  

## Usage ##

### 1. Console Interface

Run the chatbot directly in the terminal:

```bash
python app.py

  ```

- **Type your message and press Enter**  

- **Chatbot responds based on the trained model**  

- **Commands to end the conversation:** `bye`, `quit`, `exit`

**Example Interaction:**

You: Hi
Bot: Hey there! What can I do for you?
You: How to start a startup?
Bot: Starting a business requires careful planning and market research...
You: bye
Bot: Goodbye!

## 2. GUI Interface (Tkinter) ##

Run the graphical interface using:

```bash
python chatbot_gui.py

  ```
## Training Details##

The chatbot model is trained on a supervised dataset (`intents.json`) containing user intents, sample patterns, and corresponding responses. The training process includes:

- **Text Preprocessing:** Tokenization, punctuation removal, and stemming using NLTK.  
- **Feature Extraction:** TF-IDF vectorization converts text patterns into numeric features.  
- **Label Encoding:** Intent tags are converted into numeric labels using `LabelEncoder`.  
- **Handling Imbalance:** Random oversampling ensures all classes have sufficient samples.  
- **Model Architecture:** A feedforward neural network with the following layers:
  - Input layer matching TF-IDF feature size  
  - Dense layer with 128 neurons, ReLU activation  
  - Dropout layer (0.5) for regularization  
  - Dense layer with 64 neurons, ReLU activation  
  - Dropout layer (0.5)  
  - Output layer with softmax activation corresponding to the number of intents  
- **Training Parameters:**  
  - Optimizer: Adam  
  - Loss: Categorical crossentropy  
  - Epochs: 100  
  - Batch size: 8  
- **Evaluation:** Accuracy and loss monitored on validation set; classification report and confusion matrix generated.

---

## Models Used

- **TF-IDF Vectorizer:** Converts text input to numeric vectors.  
- **Label Encoder:** Encodes intent tags as numerical labels.  
- **Feedforward Neural Network:** Classifies user intents based on processed input.  
- **Random Oversampler:** Balances class distribution in training data.

---

## Results

- **Classification Accuracy:** Achieved high accuracy (>95%) on the validation set.  
- **Confusion Matrix:** Most intents correctly classified; minor confusion among similar intents.  
- **Example Interaction:**

```text
You: Hi
Bot: Hey there! What can I do for you?
You: How to start a startup?
Bot: Starting a business requires careful planning and market research...
You: bye
Bot: Goodbye!

  ```
## Conclusion

The supervised Ted! Chatbot effectively identifies user intents and generates appropriate responses. By combining TF-IDF vectorization, neural network classification, and random oversampling, the chatbot achieves high accuracy and handles class imbalance efficiently. The Tkinter GUI enhances user interaction, providing a visually intuitive chat experience.

---

## Future Work

- Introduce contextual understanding for multi-turn conversations.  
- Integrate voice input and output for a more natural conversational interface.  
- Expand the dataset to include more diverse intents and variations.  
- Incorporate advanced NLP models such as BERT or GPT for improved comprehension.  
- Implement continuous learning to update the chatbot based on new user interactions.

---

## Acknowledgements

- NLTK for natural language processing tools.  
- TensorFlow and Keras for neural network modeling.  
- scikit-learn for preprocessing, encoding, and evaluation utilities.  
- imbalanced-learn for handling class imbalance via oversampling.  
- Python open-source community for libraries and support.

---

## Author

**Shriraksha Kulkarni**  

