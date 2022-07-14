import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM,Dense,Embedding,SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from keras.callbacks import EarlyStopping
from sklearn import metrics
import pickle


MAX_WORDS = 50000
MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 100


def data_preprocessing(data):
    corpus = []
    stop_words = set(stopwords.words('english'))
    for i in range(0, len(data)):
        preprossed_text = re.sub('[^a-zA-Z]', ' ', str(data[i]))
        preprossed_text = preprossed_text.lower()
        preprossed_text = preprossed_text.split()
        preprossed_text = [w for w in preprossed_text if not w in stop_words]
        preprossed_text = ' '.join(preprossed_text)
        corpus.append(preprossed_text)
    return corpus

def define_tokenizer(texttotokenize):
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(texttotokenize)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    return tokenizer

def build_model():
    model = Sequential()
    model.add(Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=200))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def main():
    df=pd.read_csv("news_text.csv",delimiter="\t")
    final_df=df[df["category"].isin(["news","sports","finance"])]
    final_df["text"]=final_df["title"]+" "+final_df["abstract"]
    final_df=final_df.drop(["news_id","title","abstract"],axis=1)
    final_df=final_df.dropna()
    preprocessed_text=data_preprocessing(final_df["text"].values)
    final_df["preprocessed_text"]=preprocessed_text
    final_df=final_df.drop(["text"],axis=1)

    tokenizer=define_tokenizer(final_df["preprocessed_text"].values)

    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    X = tokenizer.texts_to_sequences(final_df["preprocessed_text"].values)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    Y = pd.get_dummies(final_df["category"]).values

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)

    model=build_model()

    epochs = 5
    batch_size = 64
    history = model.fit(X_train, Y_train, epochs=epochs,
                        batch_size=batch_size,validation_split=0.1,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

    model.save("LSTM_model")

    accr = model.evaluate(X_test,Y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


    input_news = ['Repayment pressure to test banks loan underwriting quality: Fitch']
    seq = tokenizer.texts_to_sequences(input_news)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded)
    labels = ["finance","news","sports"]
    print(pred, labels[np.argmax(pred)])

if __name__=="__main__":
    main()
