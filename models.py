import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

nltk.download('stopwords', quiet=True)

# ---------- 1. Load & clean ----------
df = pd.read_csv(
    r'C:\Users\ADMIN\Desktop\spam_detector\fakemail\SMSSpamCollection',
    sep='\t',
    header=None,
    names=['label', 'message']
)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

df['cleaned'] = df['message'].apply(clean_text)
df['label_enc'] = df['label'].map({'ham': 0, 'spam': 1})

# ---------- 2. TF-IDF + split ----------
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['cleaned'])
y = df['label_enc']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- 3. Train models ----------
models = {
    'Naive Bayes':         MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM':                 SVC(kernel='linear')
}

results = {}

print("Training models...\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'Accuracy':  round(accuracy_score(y_test, y_pred) * 100, 2),
        'Precision': round(precision_score(y_test, y_pred) * 100, 2),
        'Recall':    round(recall_score(y_test, y_pred) * 100, 2),
        'F1 Score':  round(f1_score(y_test, y_pred) * 100, 2)
    }
    print(f"{name} trained ✓")

# ---------- 4. Print results ----------
print("\n--- Model Comparison ---")
print(f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print("-" * 60)
for name, metrics in results.items():
    print(f"{name:<22} {metrics['Accuracy']:>8}% {metrics['Precision']:>9}% {metrics['Recall']:>7}% {metrics['F1 Score']:>7}%")

# ---------- 5. Save best model ----------
best_model_name = max(results, key=lambda x: results[x]['F1 Score'])
best_model = models[best_model_name]

print(f"\nBest model: {best_model_name} (F1: {results[best_model_name]['F1 Score']}%)")

with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("Best model saved as model.pkl")
print("TF-IDF vectorizer saved as tfidf.pkl")
print("\nStep 3 complete!")