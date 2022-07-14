from tensorflow import keras
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def main():

    model = keras.models.load_model("LSTM_model")

    MAX_SEQUENCE_LENGTH=200

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    user_input=input("Enter the news to classify:")
    input_news = [user_input]
    seq = tokenizer.texts_to_sequences(input_news)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded)
    labels = ["finance","news","sports"]
    print(pred, labels[np.argmax(pred)])

if __name__=="__main__":
    main()