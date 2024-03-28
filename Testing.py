import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for index2, char2 in enumerate(s2):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(s1):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1], distances[index1 + 1], new_distances[-1])))
        distances = new_distances
    return distances[-1]


def is_almost_match(s1, s2, threshold=5):
    distance = levenshtein_distance(s1, s2)
    print(distance)
    return distance <= threshold


# Test
s1 = "non current liabilities".lower()
s2 = "non-Current-liability".lower()

if is_almost_match(s1, s2):
    print("The strings almost match.")
else:
    print("The strings do not almost match.")


'''class CashDebtClassifier:
    def __init__(self, max_iter=1000, tol=1e-4):
        # Initialize components: TF-IDF Vectorizer, Label Encoder, and SGD Classifier
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.label_encoder = LabelEncoder()
        self.model = make_pipeline(self.tfidf_vectorizer,
                                   SGDClassifier(loss='log', max_iter=max_iter, tol=tol))

    def fit(self, X, y):
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.model.fit(X, y_encoded)

    def predict(self, new_texts):
        # Predict and return the label names
        predictions = self.model.predict(new_texts)
        return self.label_encoder.inverse_transform(predictions)


# Example usage:
# Sample labeled dataset with 'cash', 'debt', and 'unrelated' categories
data = {'text': ['I withdrew cash from the ATM',  # cash
                 'I paid off my credit card debt',  # debt
                 'I received $100 as a cashback reward',  # cash
                 'I have a mortgage debt',  # debt
                 'I deposited a cheque into my bank account',  # unrelated
                 'I bought groceries'],  # unrelated
        'category': ['cash', 'debt', 'cash', 'debt', 'unrelated', 'unrelated']}

X = data['text']
y = data['category']

# Initialize and train the model
cash_debt_classifier = CashDebtClassifier()
cash_debt_classifier.fit(X, y)

# Predictions
new_texts = ["I need to withdraw some cash",
             "I'm considering a loan for a new car",
             "He paid for dinner with his credit card"]
predictions = cash_debt_classifier.predict(new_texts)

# Output predictions
for text, prediction in zip(new_texts, predictions):
    print(f"'{text}' is classified as '{prediction}'")
'''

# Sample labeled dataset
'''data = {'text': ['I withdrew cash from the ATM',
                 'I made a payment with my credit card',
                 'I received $100 as a cashback reward',
                 'I deposited a cheque into my bank account'],
        'is_cash_related': [1, 0, 1, 0]}  # 1 for cash-related, 0 otherwise

df = pd.DataFrame(data)
print(df)
print()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['is_cash_related'], test_size=0.2, random_state=42)

# Vectorize text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train Logistic Regression model
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train_tfidf, y_train)

# Evaluate model
y_pred = logistic_regression.predict(X_test_tfidf)
print(y_pred, y_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Example prediction
new_text = ["I need to get some money from the bank"]
new_text_tfidf = tfidf_vectorizer.transform(new_text)
prediction = logistic_regression.predict(new_text_tfidf)
print("Prediction:", prediction)
'''


# LOGISTIC FUNCTION TO EXTRACT DATA
'''def LogisticMLModel(new_features: str, words_df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(words_df['features'], words_df['is_cash_related'],
                                                        test_size=0.2,
                                                        random_state=42)

    # Vectorize text data using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Train Logistic Regression model
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train_tfidf, y_train)

    # Evaluate model
    y_pred = logistic_regression.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    # print("Accuracy of the model is:", accuracy)

    # Example prediction
    new_text = [new_features]
    new_text_tfidf = tfidf_vectorizer.transform(new_text)
    prediction = logistic_regression.predict(new_text_tfidf)
    # print("Prediction of the current example is:", prediction)

    return prediction[0]
'''


