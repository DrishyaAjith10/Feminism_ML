import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset
data = pd.read_csv("dataset.csv")

# Split features and labels
X = data["sentence"]
y = data["label"]

# Convert text to numbers
vectorizer = TfidfVectorizer(
    ngram_range=(1,3),
    stop_words="english",
    lowercase=True,
    min_df=5
)
X_vectorized = vectorizer.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = LogisticRegression(
    class_weight="balanced",
    max_iter=2000
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Test custom input
def predict_sentence(sentence):
    vec = vectorizer.transform([sentence])
    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec)[0]

    print("Probability Not Feminist:", round(probability[0], 3))
    print("Probability Feminist:", round(probability[1], 3))

    return "Feminist" if prediction == 1 else "Not Feminist"


while True:
    user_input = input("\nEnter a sentence (type 'exit' to quit): ")

    if user_input.lower() == "exit":
        print("Exiting...")
        break

    vec = vectorizer.transform([user_input])
    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec)[0]

    print("\nProbability Not Feminist:", round(probability[0], 3))
    print("Probability Feminist:", round(probability[1], 3))

    if prediction == 1:
        print("Prediction: Feminist (Equality-aligned)")
    else:
        print("Prediction: Not Feminist (Gender-biased)")

