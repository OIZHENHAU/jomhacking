from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder


class LogisticSGDModel:

    def __init__(self, max_iter=1000, tol=1e-4):
        self.max_iter = max_iter
        self.tol = tol
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.model = SGDClassifier(loss='log', max_iter=max_iter, tol=tol)

    def fit(self, X, y):
        X_tfidf = self.tfidf_vectorizer.fit_transform(X)
        self.model.fit(X_tfidf, y)

    def predict(self, new_texts):
        new_text_tfidf = self.tfidf_vectorizer.transform(new_texts)
        return self.model.predict(new_text_tfidf)


'''words_to_search = {'features': ["cash and cash equivalent", "cash and bank balances", "cash at bank",
                                "cash held under housing development accounts",
                                "cash placed in conventional accounts and instruments",
                                "cash", "deposit with licensed bank", "investment", "money market instrument",
                                "other cash equivalents",
                                "deposits", "investment in cash funds", "resale agreement", "short term deposits",
                                "short term funds",
                                "short term investments", "unit trust funds", "total assets", "equity",
                                "borrowing",
                                "bank borrowings",
                                "bank overdrafts", "bankers' acceptance", "bill discounting", "bill payables",
                                "bridging loans",
                                "capital securities", "commercial papers", "commodity financing",
                                "conventional bonds", "debentures",
                                "deferred liability", "export credit refinancing", "liability"
                                                                                   "hire purchase payables",
                                "invoice financing",
                                "lease liabilities", "loan stocks",
                                "loans and borrowings"],
                   'is_cash_related': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}

X = words_to_search['features']
y = words_to_search['is_cash_related']

cash_model = LogisticSGDModel()
cash_model.fit(X, y)

new_text = ["cat"]
predictions = cash_model.predict(new_text)
# print(predictions)
'''