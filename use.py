import tkinter as tk
from tkinter import scrolledtext
from tensorflow.keras.models import load_model
import pickle
import json
import random
import nltk
import numpy as np
from nltk.stem.lancaster import LancasterStemmer

class Chatbot:
    ERROR_THRESHOLD = 0.25

    def __init__(self, model_path, data_path, intents_path):
        self.stemmer = LancasterStemmer()
        self.model = load_model(model_path)
        self.data = pickle.load(open(data_path, "rb"))
        self.words = self.data['words']
        self.classes = self.data['classes']
        with open(intents_path) as json_data:
            self.intents = json.load(json_data)

    def _clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        return [self.stemmer.stem(word.lower()) for word in sentence_words]

    def _bow(self, sentence):
        sentence_words = self._clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for s in sentence_words:
            for i, w in enumerate(self.words):
                if w == s: bag[i] = 1
        return np.array(bag)

    def classify(self, sentence):
        results = self.model.predict(np.array([self._bow(sentence)]))[0]
        results = [[i, r] for i, r in enumerate(results) if r > self.ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return [(self.classes[r[0]], r[1]) for r in results]

    def get_response(self, sentence):
        results = self.classify(sentence)
        if not results:
            return "I'm sorry, I didn't understand that."
        for i in self.intents['intents']:
            if i['tag'] == results[0][0]:
                return random.choice(i['responses'])

class ChatbotUI:
    def __init__(self, chatbot):
        self.chatbot = chatbot
        self.root = tk.Tk()
        self.root.title("Chatbot")
        self._setup_ui()

    def _setup_ui(self):
        self.chat_history = scrolledtext.ScrolledText(self.root, state='disabled', width=80, height=20, wrap='word', font=("Arial", 12))
        self.chat_history.tag_config('user', foreground='blue')
        self.chat_history.tag_config('bot', foreground='green')
        self.chat_history.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        self.user_text = tk.Text(self.root, height=1, width=60, font=("Arial", 12))
        self.user_text.grid(row=1, column=0, padx=1, pady=1)
        self.user_text.bind("<Return>", self._send)

        send_button = tk.Button(self.root, text="Send", command=self._send, font=("Arial", 12))
        send_button.grid(row=1, column=1, padx=10, pady=10)

    def _send(self, event=None):
        user_input = self.user_text.get("1.0", 'end-1c').strip()
        if not user_input:
            return
        self.user_text.delete("1.0", tk.END)
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.insert(tk.END, "You: " + user_input + '\n', 'user')
        bot_response = self.chatbot.get_response(user_input)
        self.chat_history.insert(tk.END, "Bot: " + bot_response + '\n', 'bot')
        self.chat_history.config(state=tk.DISABLED)
        self.chat_history.yview(tk.END)
        if user_input.lower() == 'quit':
            self.root.quit()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    chatbot = Chatbot('model.keras', 'training_data', 'intents.json')
    chatbot_ui = ChatbotUI(chatbot)
    chatbot_ui.run()