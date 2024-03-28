import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder


class CashDebtClassifier:
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
