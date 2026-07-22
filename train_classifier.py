import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

print("Loading labeled reviews...")
df = pd.read_csv("labeled_reviews.csv")

# Filter out rows where category is missing or empty
df = df.dropna(subset=["category"])
df = df[df["category"].str.strip() != ""]

X = df["text"].astype(str)
y = df["category"].astype(str)

print(f"Training model on {len(df)} samples across {y.nunique()} categories.")

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text into numerical TF-IDF features
vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Naive Bayes Classifier (bypasses Windows DLL security blocks)
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# Evaluate the model
y_pred = clf.predict(X_test_vec)
print("\n--- Model Evaluation Report ---")
print(classification_report(y_test, y_pred, zero_division=0))

# Save the trained model and vectorizer for your Streamlit app
joblib.dump(clf, "complaint_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print(
    "\nSaved trained model ('complaint_model.pkl') and vectorizer"
    " ('vectorizer.pkl') successfully!"
)