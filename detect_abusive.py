import argparse
import numpy as np
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
import pickle

# Load the tokenizer and label encoder
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('encoder.pkl', 'rb') as handle:
    label_encoder = pickle.load(handle)

# Load the trained model
model = load_model("best_model")

# Function to predict abusive language
def detect_abusive(comment, tokenizer, model, label_encoder, max_length=50):
    # Preprocess and tokenize the input comment
    text_sequence = tokenizer.texts_to_sequences([comment])
    text_padded = pad_sequences(text_sequence, maxlen=max_length)
    
    # Make prediction
    pred = model.predict(text_padded)
    pred = int(np.round(np.squeeze(pred)))
    return label_encoder.inverse_transform([pred])[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect abusive language in comments.")
    parser.add_argument("--input", type=str, required=True, help="Comment to classify.")
    args = parser.parse_args()
    
    result = detect_abusive(args.input, tokenizer, model, label_encoder)
    print(f"Comment: {args.input}\nPrediction: {result}")
