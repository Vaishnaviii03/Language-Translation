from flask import Flask, render_template, request
from keras.models import load_model
import json
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np

app = Flask(__name__)

# Load the model
model = load_model('english_to_french_model.keras')

# Load the tokenizers
with open('english_tokenizer.json', 'r', encoding='utf8') as f:
    english_tokenizer = tokenizer_from_json(json.load(f))
with open('french_tokenizer.json', 'r', encoding='utf8') as f:
    french_tokenizer = tokenizer_from_json(json.load(f))

# Load max sequence length
with open('sequence_length.json', 'r', encoding='utf8') as f:
    max_french_sequence_length = json.load(f)

def translate_sentence(english_sentence):
    try:
        # Tokenize and pad the input sentence
        seq = english_tokenizer.texts_to_sequences([english_sentence])
        padded_seq = pad_sequences(seq, maxlen=max_french_sequence_length, padding='post')
        
        # Predict the French sentence
        pred = model.predict(padded_seq)
        pred = np.argmax(pred, axis=-1)
        
        # Convert token IDs to words
        french_sentence = ' '.join(french_tokenizer.index_word.get(i, '<UNK>') for i in pred[0] if i != 0)
        
        return french_sentence
    except Exception as e:
        print(f"Error during translation: {e}")
        return "Translation Error"

@app.route('/', methods=['GET', 'POST'])
def home():
    french_sentence = ""
    if request.method == 'POST':
        english_sentence = request.form['english_sentence']
        if english_sentence:
            french_sentence = translate_sentence(english_sentence)
    
    return render_template('index.html', french_sentence=french_sentence)

if __name__ == "__main__":
    app.run(debug=True)
