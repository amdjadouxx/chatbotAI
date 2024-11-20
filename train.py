import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow as tf
import random
import pickle
import json
from tensorflow.keras.callbacks import TensorBoard
from time import time

tensorboard=TensorBoard(log_dir="logs/{}".format(time()))

class DataPreprocessor:
    def __init__(self, intents_file):
        self.stemmer = LancasterStemmer()
        self.intents_file = intents_file
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_words = ['?', '!', '.', ',', ';', ':', '-', '_', '(', ')', '[', ']', '{', '}', '&', '@', '#', '$', '%', '^', '*', '+', '=', '<', '>', '/', '\\', '|', '`', '~']
        self.load_intents()
        self.preprocess_data()

    def load_intents(self):
        with open(self.intents_file) as data:
            self.intents = json.load(data)

    def preprocess_data(self):
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                w = nltk.word_tokenize(pattern)
                self.words.extend(w)
                self.documents.append((w, intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])
        self.words = [self.stemmer.stem(w.lower()) for w in self.words if w not in self.ignore_words]
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))

    def create_training_data(self):
        training = []
        output_empty = [0] * len(self.classes)
        for doc in self.documents:
            bag = []
            pattern_words = doc[0]
            pattern_words = [self.stemmer.stem(word.lower()) for word in pattern_words]
            for w in self.words:
                bag.append(1) if w in pattern_words else bag.append(0)
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            training.append([bag, output_row])
        random.shuffle(training)
        train_x = np.array([np.array(t[0]) for t in training])
        train_y = np.array([np.array(t[1]) for t in training])
        return train_x, train_y

class ModelBuilder:
    @staticmethod
    def build_model(input_shape, output_shape):
        tf.keras.backend.clear_session()
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(8, input_shape=(input_shape,), activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(output_shape, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

class Trainer:
    def __init__(self, model, train_x, train_y):
        self.model = model
        self.train_x = train_x
        self.train_y = train_y

    def train(self):
        self.model.fit(np.array(self.train_x), np.array(self.train_y), epochs=1000, batch_size=8, verbose=1, callbacks=[tensorboard])
        self.model.save('model.keras')

def main():
    data_preprocessor = DataPreprocessor('intents.json')
    train_x, train_y = data_preprocessor.create_training_data()
    model = ModelBuilder.build_model(len(train_x[0]), len(train_y[0]))
    trainer = Trainer(model, train_x, train_y)
    trainer.train()
    pickle.dump({'words': data_preprocessor.words, 'classes': data_preprocessor.classes, 'train_x': train_x, 'train_y': train_y}, open("training_data", "wb"))

if __name__ == "__main__":
    main()