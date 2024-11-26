CHATGPT CONVERSATION

-----------


You said:
create a PYTHON code that visualizes machine learning of a text
ChatGPT said:
ChatGPT
To visualize the machine learning process of a text, we can use Natural Language Processing (NLP) to train a text classifier and then display important aspects of the process. Here's a Python script that demonstrates this using the following steps:

Preprocessing the text data.
Training a simple classifier (e.g., Logistic Regression).
Visualizing key insights like word frequency and feature importance.
Below is the code:

Failed to generate
Here's the Python script:

python
Copy code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Sample text data (you can replace this with your dataset)
data = {
    "text": [
        "I love programming in Python",
        "Python is an amazing language",
        "I dislike debugging code",
        "Debugging is so frustrating",
        "I enjoy solving problems with Python",
        "Sometimes coding can be challenging",
    ],
    "label": [1, 1, 0, 0, 1, 0],  # 1: Positive sentiment, 0: Negative sentiment
}

# Load the data into a DataFrame
df = pd.DataFrame(data)

# Step 1: Text Preprocessing
vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train a Classifier
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 4: Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualizations
# 1. Word Frequency
word_freq = np.array(X.sum(axis=0)).flatten()
words = vectorizer.get_feature_names_out()
word_freq_df = pd.DataFrame({"word": words, "frequency": word_freq})
word_freq_df = word_freq_df.sort_values(by="frequency", ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x="frequency", y="word", data=word_freq_df, palette="viridis")
plt.title("Top 10 Most Frequent Words")
plt.xlabel("Frequency")
plt.ylabel("Words")
plt.show()

# 2. Feature Importance (Weights of Logistic Regression)
feature_importance = model.coef_[0]
importance_df = pd.DataFrame({"word": words, "importance": feature_importance})
importance_df = importance_df.sort_values(by="importance", ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x="importance", y="word", data=importance_df, palette="coolwarm")
plt.title("Top 10 Important Words for Prediction")
plt.xlabel("Importance")
plt.ylabel("Words")
plt.show()

# 3. Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
Features of the Code:
Text Preprocessing: Uses a CountVectorizer to transform text into numerical data.
Visualization:
Top 10 most frequent words.
Top 10 important words based on the classifier's weights.
Confusion matrix to visualize prediction accuracy.
Let me know if you need help adapting this for your dataset! 😊📊✨





