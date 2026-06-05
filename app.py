from flask import Flask, render_template, request
import pickle
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)

nltk.download('stopwords', quiet=True)

app = Flask(__name__)

# ---------- Load model and vectorizer ----------
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# ---------- Clean text ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# ---------- Compute model results ----------
def get_model_results():
    df = pd.read_csv(
        r'C:\Users\ADMIN\Desktop\spam_detector\fakemail\SMSSpamCollection',
        sep='\t', header=None, names=['label', 'message']
    )
    df['cleaned'] = df['message'].apply(clean_text)
    df['label_enc'] = df['label'].map({'ham': 0, 'spam': 1})

    tv = TfidfVectorizer(max_features=3000)
    X = tv.fit_transform(df['cleaned'])
    y = df['label_enc']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        'Naive Bayes':         MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'SVM':                 SVC(kernel='linear')
    }

    results = []
    for name, m in models.items():
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        results.append({
            'model':     name,
            'accuracy':  round(accuracy_score(y_test, y_pred) * 100, 2),
            'precision': round(precision_score(y_test, y_pred) * 100, 2),
            'recall':    round(recall_score(y_test, y_pred) * 100, 2),
            'f1':        round(f1_score(y_test, y_pred) * 100, 2)
        })
    return results

# ---------- Routes ----------
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
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

@app.route('/dashboard')
def dashboard():
    results = get_model_results()
    return render_template('dashboard.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)