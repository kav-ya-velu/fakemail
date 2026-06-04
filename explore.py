import pandas as pd
import matplotlib.pyplot as plt

# ---------- 1. Load the dataset ----------
df = pd.read_csv(
    'SMSSpamCollection',
    sep='\t',
    header=None,
    names=['label', 'message']
)

# ---------- 2. Basic info ----------
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nLabel counts:")
print(df['label'].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

# ---------- 3. Add message length column ----------
df['msg_length'] = df['message'].apply(len)

print("\nAverage message length by label:")
print(df.groupby('label')['msg_length'].mean())

# ---------- 4. Visualise class distribution ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

df['label'].value_counts().plot(kind='bar', ax=axes[0], color=['steelblue', 'tomato'])
axes[0].set_title('Spam vs Ham count')
axes[0].set_xlabel('Label')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=0)

df[df['label'] == 'ham']['msg_length'].plot(
    kind='hist', bins=40, alpha=0.6, label='Ham', color='steelblue', ax=axes[1]
)
df[df['label'] == 'spam']['msg_length'].plot(
    kind='hist', bins=40, alpha=0.6, label='Spam', color='tomato', ax=axes[1]
)
axes[1].set_title('Message length distribution')
axes[1].set_xlabel('Character count')
axes[1].legend()

plt.tight_layout()
plt.savefig('step1_eda.png')
plt.show()
print("\nPlot saved as step1_eda.png")