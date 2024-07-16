import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the dataset, trying different encodings
try:
    df = pd.read_csv('spam.csv', encoding='utf-8')  # Try UTF-8 first
except UnicodeDecodeError:
    df = pd.read_csv('spam.csv', encoding='latin-1')  # Try Latin-1 if UTF-8 fails

# Preprocess the data
df['v2'] = df['v2'].str.lower()
df['v2'] = df['v2'].str.replace('[^\w\s]', '')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['v2'], df['v1'], test_size=0.2)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)