import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dropout, Dense, Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam

import nltk
from nltk.stem import PorterStemmer
# import spellchecker 
# from spellchecker import SpellChecker

import spacy
import gensim.downloader as api
from neo4j import GraphDatabase


# Initialize NLTK resources
# nltk.download('punkt')

# Initialize Stemmer
stemmer = PorterStemmer()

# Initialize Spell Checker
# spell = SpellChecker()


# Initialize SpaCy
# spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

# Download pre-trained Word2Vec model
w2v_model = api.load("word2vec-google-news-300")


# Neo4j connection details
uri = "neo4j+s://c22b0b3c.databases.neo4j.io:7687"
username = "neo4j"
password = "akterzObCsJTwulDEG7AFlCkDcMyGC7RblEnmXbg7aE"

# Function to connect to Neo4j
def connect_to_neo4j(uri, username, password):
    return GraphDatabase.driver(uri, auth=(username, password))

# Function to retrieve nodes and attributes from Neo4j
def retrieve_nodes_and_attributes(driver):
    with driver.session() as session:
        result = session.run(
            "MATCH (n:Intent)-[r:HAS_ANSWER]->(a:Attribute) RETURN n.name AS intent, collect(a.value) AS answers"
        )
        nodes_and_attributes = {record['intent']: {"answer": record['answers']} for record in result}
    return nodes_and_attributes

# Function to preprocess text
def preprocess_text(text):
    # Lemmatization
    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc])
    
    # Spell Correction
    corrected_text = [spell.correction(word) for word in lemmatized_text.split()]
    corrected_text = " ".join(corrected_text)
    
    return corrected_text

# Function to generate utterances in Singaporean accent
def singaporean_accent(text):
    # Example implementation
    singaporean_phrases = {
        "Hello": "Alamak, how can I help you today?",
        "account balance": "Your current account balancy is $5000 lah."
        # Add more phrases as needed
    }
    for phrase, replacement in singaporean_phrases.items():
        text = text.replace(phrase, replacement)
    return text


# Download pre-trained Word2Vec model
w2v_model = api.load("word2vec-google-news-300")

# Function to calculate average Word2Vec embeddings for an utterance
def get_avg_word2vec_embedding(text):
    words = text.split()
    embeddings = [w2v_model[word] for word in words if word in w2v_model.vocab]
    if embeddings:
        return sum(embeddings) / len(embeddings)
    else:
        return None



# Function to preprocess text and generate Word2Vec embeddings (if needed)
def preprocess_text_and_embeddings(utterances):
    processed_utterances = [preprocess_text(utterance) for utterance in utterances]
    embeddings = [get_avg_word2vec_embedding(utterance) for utterance in processed_utterances]
    return np.array(embeddings)

# Function to train the LSTM model
def train_lstm_model(X, y):
    # Define LSTM model architecture
    model = Sequential([
        Embedding(input_dim=X.shape[0], output_dim=300, input_length=X.shape[1]),
        LSTM(128),
        Dense(64, activation='relu'),
        Dense(len(set(y)), activation='softmax')
    ])
    
    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train the model
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
    
    return model


# Function to preprocess text
def preprocess_text(utterances):
    return [preprocess_text(utterance) for utterance in utterances]

# Function to fine-tune BERT for intent classification
def train_bert_model(X, y):
    # Load BERT from TensorFlow Hub
    bert_module = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3", trainable=True)
    
    # Define model architecture
    input_word_ids = Input(shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_seq_length,), dtype=tf.int32, name="segment_ids")
    
    pooled_output, _ = bert_module([input_word_ids, input_mask, segment_ids])
    dropout = Dropout(0.1)(pooled_output)
    output = Dense(len(set(y)), activation='softmax')(dropout)
    
    model = tf.keras.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=output)
    
    # Compile the model
    optimizer = Adam(lr=2e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    # Train the model
    model.fit(X, y, epochs=3, batch_size=32, validation_split=0.2)
    
    return model

# Main function
def main():
    # Connect to Neo4j and retrieve nodes and attributes
    driver = connect_to_neo4j(uri, username, password)
    nodes_and_attributes = retrieve_nodes_and_attributes(driver)
    
    # Extract intents and utterances
    intents = []
    utterances = []
    
    for intent, attributes in nodes_and_attributes.items():
        intent = intent.lower()  # Lowercase intent
        intents.append(intent)
        
        answers = attributes.get("answer", [])
        for answer in answers:
            utterances.append(answer)
    
    # Preprocess utterances
    preprocessed_utterances = preprocess_text(utterances)
    
    # Encode intents
    label_encoder = LabelEncoder()
    encoded_intents = label_encoder.fit_transform(intents)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(preprocessed_utterances, encoded_intents, test_size=0.2, random_state=42)
    
    # Fine-tune BERT model
    bert_model = train_bert_model(X_train, y_train)
    
    # Evaluate the model
    loss, accuracy = bert_model.evaluate(X_test, y_test)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)

if __name__ == "__main__":
    main()