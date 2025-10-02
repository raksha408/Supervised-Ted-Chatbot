import tkinter as tk
from tkinter import Canvas, Frame, Label, Scrollbar
import json
import random
import string
import nltk
import datetime
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# ========== FILE PATHS ==========
MODEL_PATH = "chatbot_model.h5"
VECTORIZER_PATH = "vectorizer.pkl"
ENCODER_PATH = "label_encoder.pkl"
INTENTS_PATH = r"D:\Internship\Programs\Chatbot_recent\intents.json"

# ========== LOAD INTENTS ==========
with open(INTENTS_PATH, 'r') as file:
    data = json.load(file)
responses = {intent['tag']: intent['responses'] for intent in data['intents']}

# ========== NLTK SETUP ==========
nltk.download('punkt')
stemmer = nltk.PorterStemmer()

# ========== LOAD MODEL & TOOLS ==========
if not (MODEL_PATH and VECTORIZER_PATH and ENCODER_PATH):
    print("Please ensure the model and tools are saved.")
    exit()

model = load_model(MODEL_PATH)
with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)
with open(ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)

# ========== PREPROCESS FUNCTION ==========
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    return " ".join([stemmer.stem(word) for word in tokens])

# ========== CHATBOT RESPONSE (NO FALLBACK) ==========
def get_bot_response(user_input):
    processed = preprocess_text(user_input)
    X = vectorizer.transform([processed]).toarray()
    prediction = model.predict(X)[0]
    tag_index = np.argmax(prediction)
    tag = le.inverse_transform([tag_index])[0]
    return random.choice(responses.get(tag, ["I'm not sure how to respond."]))

# ========== UTILITY TO DRAW ROUNDED RECTANGLES ON CANVAS ==========
def round_rect(canvas, x1, y1, x2, y2, r=25, **kwargs):
    """
    Draw a rounded rectangle on 'canvas' from (x1,y1) to (x2,y2) with corner radius r.
    """
    points = [
        x1 + r, y1,
        x1 + r, y1,
        x2 - r, y1,
        x2 - r, y1,
        x2, y1,
        x2, y1 + r,
        x2, y2 - r,
        x2, y2,
        x2 - r, y2,
        x2 - r, y2,
        x1 + r, y2,
        x1 + r, y2,
        x1, y2,
        x1, y2 - r,
        x1, y1 + r,
        x1, y1
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)

# ========== MAIN BUBBLE CHAT GUI ==========
class BubbleChat(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ted! Chatbot")
        self.geometry("520x600")
        self.configure(bg="white")
        
        # TOP BAR
        top_bar = tk.Frame(self, bg="#007bff", height=60)
        top_bar.pack(side="top", fill="x")
        
        title_label = tk.Label(top_bar, text="Ted! Chatbot", bg="#007bff", fg="white",
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # SCROLLABLE CANVAS FOR MESSAGES
        self.canvas_frame = tk.Frame(self, bg="white")
        self.canvas_frame.pack(side="top", fill="both", expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="white", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        
        self.scrollbar = Scrollbar(self.canvas_frame, command=self.canvas.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.bubble_area = tk.Frame(self.canvas, bg="white")
        self.canvas.create_window((0,0), window=self.bubble_area, anchor="nw")
        self.bubble_area.bind("<Configure>", self.on_configure)
        
        # BOTTOM BAR
        self.bottom_bar = tk.Frame(self, bg="white", height=60)
        self.bottom_bar.pack(side="bottom", fill="x")
        
        # Placeholder text
        self.placeholder = "Enter your message. Type 'help' to know all my capabilities..."
        
        self.user_input = tk.StringVar()
        
        # Canvas for entry background (rounded rectangle)
        self.entry_canvas = tk.Canvas(self.bottom_bar, width=400, height=40, bg="white", highlightthickness=0)
        self.entry_canvas.pack(side="left", padx=(10,5), pady=5)
        
        # Draw a rounded rectangle for the entry background
        round_rect(self.entry_canvas, 0, 0, 400, 40, r=20, fill="white", outline="#ccc")
        
        # Actual entry inside
        self.entry = tk.Entry(self.bottom_bar, textvariable=self.user_input,
                              font=("Arial", 12), bd=0, fg="gray")
        self.entry_canvas.create_window(10, 5, anchor="nw", window=self.entry, width=380, height=30)
        
        self.entry.insert(0, self.placeholder)
        self.entry.bind("<FocusIn>", self.clear_placeholder)
        self.entry.bind("<FocusOut>", self.restore_placeholder)
        self.entry.bind("<Return>", self.send_message)
        
        # Canvas for send button (circle)
        self.send_canvas = tk.Canvas(self.bottom_bar, width=40, height=40, bg="white", highlightthickness=0)
        self.send_canvas.pack(side="left", padx=(5,10), pady=5)
        
        # Draw a green circle as the send button
        self.send_btn_id = self.send_canvas.create_oval(0,0,40,40, fill="#4CAF50", outline="#4CAF50")
        # Create a '>' text or triangle for the arrow
        self.send_canvas.create_text(20, 20, text="➤", fill="white", font=("Arial", 14, "bold"))
        self.send_canvas.tag_bind(self.send_btn_id, "<Button-1>", self.send_message)
        self.send_canvas.bind("<Button-1>", self.send_message)
        
        # Initial welcome message
        self.add_bot_bubble("Hey, I am Ted! Type below to talk to me...")

    def on_configure(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def clear_placeholder(self, event=None):
        if self.entry.get() == self.placeholder:
            self.entry.delete(0, tk.END)
            self.entry.config(fg="black")

    def restore_placeholder(self, event=None):
        if self.entry.get().strip() == "":
            self.entry.config(fg="gray")
            self.entry.insert(0, self.placeholder)

    # BOT BUBBLE (gray, left)
    def add_bot_bubble(self, text):
        bubble_frame = Frame(self.bubble_area, bg="white")
        bubble_frame.pack(anchor="w", pady=5, padx=10, fill="x")
        
        time_str = datetime.datetime.now().strftime("%I:%M %p")
        Label(bubble_frame, text=f"Ted {time_str}", bg="white",
              fg="#555", font=("Arial", 8, "bold")).pack(anchor="w")
        
        msg_label = tk.Label(bubble_frame, text=text, font=("Arial", 11),
                             bg="#e8e8e8", fg="black", wraplength=300, justify="left")
        msg_label.pack(anchor="w", padx=5, pady=2)
        
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)

    # USER BUBBLE (green, right)
    def add_user_bubble(self, text):
        bubble_frame = Frame(self.bubble_area, bg="white")
        bubble_frame.pack(anchor="e", pady=5, padx=10, fill="x")
        
        time_str = datetime.datetime.now().strftime("%I:%M %p")
        Label(bubble_frame, text=f"You {time_str}", bg="white",
              fg="#555", font=("Arial", 8, "bold")).pack(anchor="e")
        
        msg_label = tk.Label(bubble_frame, text=text, font=("Arial", 11),
                             bg="#c3f7c3", fg="black", wraplength=300, justify="left")
        msg_label.pack(anchor="e", padx=5, pady=2)
        
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)

    def send_message(self, event=None):
        user_msg = self.entry.get().strip()
        if user_msg == "" or user_msg == self.placeholder:
            return
        self.add_user_bubble(user_msg)
        self.entry.delete(0, tk.END)
        
        # Manual exit check
        if user_msg.lower() in ["quit", "exit", "bye"]:
            self.add_bot_bubble("Goodbye!")
            self.after(1000, self.destroy)
            return
        
        # Bot response
        bot_msg = get_bot_response(user_msg)
        self.add_bot_bubble(bot_msg)

        # If user input or bot_msg suggests 'goodbye', end session
        if user_msg.lower() in ["bye", "goodbye"]:
            self.add_bot_bubble("(Session Ended)")
            self.after(1000, self.destroy)

def main():
    app = BubbleChat()
    app.mainloop()

if __name__ == "__main__":
    main()
