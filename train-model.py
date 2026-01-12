# Import libraries
import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight
import numpy as np


# Load datasets
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true], ignore_index=True)

# Function for text cleaning
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove punctuation & numbers
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text

# Apply cleaning
data["text"] = data["text"].apply(clean_text)

# Split into features and labels
X = data["text"]
y = data["label"]

# this line to handle class imbalance
weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weights = {0: weights[0], 1: weights[1]}

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7,min_df=5, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)


# Train Logistic Regression model with class weights (this is the key fix)
model = LogisticRegression(max_iter=1000, class_weight=class_weights)
model.fit(X_train_vec, y_train)


# Save the model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# Test accuracy
X_test_vec = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print("Testing accuracy:", accuracy)

#training accuracy

X_train_vec = vectorizer.transform(X_train)  # you already have this
y_train_pred = model.predict(X_train_vec)
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy:", train_accuracy)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Fake News Detection")
plt.show()



print("Model trained and saved successfully.")
