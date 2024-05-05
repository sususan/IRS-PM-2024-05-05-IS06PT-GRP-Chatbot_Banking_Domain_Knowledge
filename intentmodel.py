import numpy as np
import pandas as pd
import json

def clear_screen():
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

def check_and_download_embeddings():
    from pathlib import Path
    url ='https://www.dropbox.com/s/a247ju2qsczh0be/glove.6B.100d.txt?dl=1'
    if Path('./glove.6B.100d.txt').exists() is not True:
        import wget
        wget.download(url)
    clear_screen()

check_and_download_embeddings()
embeddings_index={}
with open('glove.6B.100d.txt', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow info and warning messages
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Activation, Bidirectional,Embedding

def save_model(model, classes, tokenizer, label_encoder):
    print('Started Save the model')
    import pickle
    import json
    model.save('intents.h5', overwrite=True, save_format='h5')
    #model.export('intents.h5', overwrite=True, save_format='h5')

    with open('classes.pkl','wb') as file:
        pickle.dump(classes,file)

    with open('tokenizer.pkl','wb') as file:
        pickle.dump(tokenizer,file)

    with open('label_encoder.pkl','wb') as file:
        pickle.dump(label_encoder,file)

def train_model():
    print('Started Train the model')
    df = pd.read_csv("new_intents_utterances.csv")

    text = df['utterance']
    labels = df['intent']


    train_txt,test_txt,train_label,test_labels = train_test_split(text,labels,test_size = 0.3)

    max_num_words = 40000
    classes = np.unique(labels)

    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(train_txt)
    word_index = tokenizer.word_index

    ls=[]
    for c in train_txt:
        ls.append(len(c.split()))
    maxLen=int(np.percentile(ls, 98))
    train_sequences = tokenizer.texts_to_sequences(train_txt)
    train_sequences = pad_sequences(train_sequences, maxlen=maxLen,              padding='post')
    test_sequences = tokenizer.texts_to_sequences(test_txt)
    test_sequences = pad_sequences(test_sequences, maxlen=maxLen, padding='post')

    

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(classes)

    onehot_encoder = OneHotEncoder(sparse_output=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoder.fit(integer_encoded)
    train_label_encoded = label_encoder.transform(train_label)
    train_label_encoded = train_label_encoded.reshape(len(train_label_encoded), 1)
    train_label = onehot_encoder.transform(train_label_encoded)
    test_labels_encoded = label_encoder.transform(test_labels)
    test_labels_encoded = test_labels_encoded.reshape(len(test_labels_encoded), 1)
    test_labels = onehot_encoder.transform(test_labels_encoded)

    # print (type(embeddings_index.values()))
    embeddings = list(embeddings_index.values())
    all_embs = np.stack(embeddings)
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    num_words = min(max_num_words, len(word_index))+1
    embedding_dim=len(embeddings_index['the'])
    embedding_matrix = np.random.normal(emb_mean, emb_std, (num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= max_num_words:
            break
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    model = Sequential()

    model.add(Embedding(num_words, 100, trainable=False,input_length=train_sequences.shape[1], weights=[embedding_matrix]))
    # model.add(Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.1, dropout=0.1), 'concat'))
    model.add(Dropout(0.7))
    model.add(LSTM(256, return_sequences=False, recurrent_dropout=0.1, dropout=0.1))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes.shape[0], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    history = model.fit(train_sequences, train_label, epochs = 450,
            batch_size = 64, shuffle=True,
            validation_data=[test_sequences, test_labels])

    save_model(model, classes, tokenizer, label_encoder)

    nlu = IntentClassifier(classes,model,tokenizer,label_encoder)
    return nlu



class IntentClassifier:
    def __init__(self,classes,model,tokenizer,label_encoder):
        self.classes = classes
        self.classifier = model
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder

    def get_intent(self,text):
        self.text = [text]
        self.test_keras = self.tokenizer.texts_to_sequences(self.text)
        self.test_keras_sequence = pad_sequences(self.test_keras, maxlen=16, padding='post')
        self.pred = self.classifier.predict(self.test_keras_sequence, verbose=None)
        return self.label_encoder.inverse_transform(np.argmax(self.pred,1))[0]
    

def load_model():
    import pickle
    from tensorflow.keras.models import load_model
    #model = load_model('intents.h5',  compile=True)
    model = load_model('intents.h5')

    print("model loaded")
    with open('classes.pkl','rb') as file:
        classes = pickle.load(file)

    with open('tokenizer.pkl','rb') as file:
        tokenizer = pickle.load(file)

    with open('label_encoder.pkl','rb') as file:
        label_encoder = pickle.load(file)

    nlu = IntentClassifier(classes,model,tokenizer,label_encoder)
    return nlu




from pathlib import Path

def all_artifacts_exist()->bool:
    model_file =Path("intents.h5")
    classes_file = Path("classes.pkl")
    Labelencoder_file = Path("label_encoder.pkl")
    tokenizer_file = Path('tokenizer.pkl')
    if model_file.exists() and classes_file.exists() and Labelencoder_file.exists() and tokenizer_file.exists():
        return True
    else:
        return False

nlu = load_model() if all_artifacts_exist() else train_model()