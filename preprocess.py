import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# ---------- 1. Download NLTK stopwords ----------
nltk.download('stopwords')

# ---------- 2. Load dataset ----------
df = pd.read_csv(
    r'C:\Users\ADMIN\Desktop\spam_detector\fakemail\SMSSpamCollection',
    sep='\t',
    header=None,
    names=['label', 'message']
)

# ---------- 3. Clean text function ----------
def clean_text(text):
    text = text.lower()                          # lowercase
    text = re.sub(r'[^a-z\s]', '', text)         # remove punctuation & numbers
    tokens = text.split()                         # split into words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]  # remove stopwords
    return ' '.join(tokens)

# ---------- 4. Apply cleaning ----------
df['cleaned'] = df['message'].apply(clean_text)

print("Original message:")
print(df['message'][2])
print("\nCleaned message:")
print(df['cleaned'][2])

# ---------- 5. Encode labels ----------
df['label_enc'] = df['label'].map({'ham': 0, 'spam': 1})

# ---------- 6. TF-IDF Vectorization ----------
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['cleaned'])
y = df['label_enc']

print("\nTF-IDF matrix shape:", X.shape)

# ---------- 7. Train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

print("\nStep 2 complete! Data is ready for model training.")