import argparse
import pandas as pd
import re
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from nltk.corpus import stopwords
from nltk import SnowballStemmer
from sklearn.utils import shuffle

# Function to clean and preprocess text
def clean_text(text, stopwords_set, stemmer):
    words = str(text).strip().split()
    words = [word for word in words if word not in stopwords_set]
    words = [re.sub(r'[^\u0600-\u06FF]|\u061F|\u060C', '', word) for word in words]
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

# Function to build CNN-LSTM model
def build_cnn_lstm_model(max_words, embedding_dim, max_length):
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=max_length))
    model.add(Conv1D(128, 3, activation='relu', padding='same'))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(128, 3, activation='relu', padding='same'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(5))
    model.add(LSTM(128, dropout=0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Main function to handle training
def train_model(model_type):
    # Load and preprocess the dataset
    print("Loading dataset...")
    df = pd.read_excel('Dataset/tunisian_youtube_comments_dataset.xlsx')
    df.drop_duplicates(subset='Comment', inplace=True)
    df = shuffle(df, random_state=5).reset_index(drop=True)
    
    print("Preprocessing data...")
    arabic_stopwords = set(stopwords.words('arabic'))
    stemmer = SnowballStemmer("arabic")
    df['Comment'] = df['Comment'].apply(lambda x: clean_text(x, arabic_stopwords, stemmer))

    # Encode labels
    label_encoder = LabelEncoder()
    df['Label'] = label_encoder.fit_transform(df['Label'])

    # Tokenize and pad sequences
    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(df['Comment'])
    sequences = tokenizer.texts_to_sequences(df['Comment'])
    max_length = 50
    X = pad_sequences(sequences, maxlen=max_length)
    y = df['Label']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the selected model
    print(f"Building model: {model_type}")
    if model_type == 'cnn-lstm':
        model = build_cnn_lstm_model(max_words=1000, embedding_dim=100, max_length=max_length)
    else:
        raise ValueError(f"Model type {model_type} not supported.")

    # Define callbacks
    filepath = "weights_best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only=True)
    lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=5, verbose=1, factor=0.5, min_lr=0.00001)

    # Train the model
    print("Training model...")
    history = model.fit(X_train, y_train, 
                        epochs=30, 
                        batch_size=32, 
                        validation_data=(X_test, y_test), 
                        callbacks=[checkpoint, lr_reduction],
                        verbose=1)
    
    # Save the final model
    model.save("created_model")
    print("Model training complete. Created model saved as created_model.")

# Entry point for script execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a deep learning model for abusive language detection.")
    parser.add_argument("--model", type=str, required=True, help="Specify the model type (e.g., 'cnn-lstm').")
    args = parser.parse_args()
    train_model(args.model)
