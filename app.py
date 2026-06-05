from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

app = Flask(__name__)

# Load model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    message = None

    if request.method == 'POST':
        message = request.form['message']
        cleaned = clean_text(message)
        vectorized = tfidf.transform([cleaned])
        result = model.predict(vectorized)[0]
        prediction = 'SPAM' if result == 1 else 'HAM'

    return render_template('index.html',
                           prediction=prediction,
                           message=message)

if __name__ == '__main__':
    app.run(debug=True)