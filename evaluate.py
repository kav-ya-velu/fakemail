import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

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
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    results[name] = {
        'Accuracy':  round(accuracy_score(y_test, y_pred) * 100, 2),
        'Precision': round(precision_score(y_test, y_pred) * 100, 2),
        'Recall':    round(recall_score(y_test, y_pred) * 100, 2),
        'F1 Score':  round(f1_score(y_test, y_pred) * 100, 2)
    }

# ---------- 4. Plot confusion matrices ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')

for ax, (name, y_pred) in zip(axes, predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=['Ham', 'Spam'],
        yticklabels=['Ham', 'Spam']
    )
    ax.set_title(name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.show()

# ---------- 5. Plot model comparison bar chart ----------
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
model_names = list(results.keys())

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle('Model Comparison', fontsize=14, fontweight='bold')

colors = ['steelblue', 'tomato', 'seagreen']

for ax, metric in zip(axes, metrics):
    values = [results[m][metric] for m in model_names]
    bars = ax.bar(model_names, values, color=colors)
    ax.set_title(metric)
    ax.set_ylim(60, 101)
    ax.set_ylabel('Score (%)')
    ax.tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f'{val}%', ha='center', va='bottom', fontsize=9
        )

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()

print("Confusion matrices saved as confusion_matrices.png")
print("Model comparison saved as model_comparison.png")
print("\nStep 4 complete!")