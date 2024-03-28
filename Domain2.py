import math

import PyPDF2
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PyPDF2 import PdfReader
import tabula
from tabula.io import read_pdf
import pandas as pd

from LogisticSGDModel import LogisticSGDModel

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

reader = PdfReader("Financial_Statements.pdf")
page = reader.pages[2]
# print(page.extract_text())


totalPages = len(reader.pages)
# print(f"Total Pages: {totalPages}")

tables = tabula.io.read_pdf("Financial_Statements.pdf", stream=True, pages="11")


# print(tables[0])


def ReadTablePages(pdf: PyPDF2.PdfReader, arr: np.ndarray):
    length_arr = len(arr)
    pages_label = str(arr[0])
    result_tables = tabula.io.read_pdf(pdf, stream=True, pages=pages_label)[0]

    for i in range(1, length_arr):
        curr_pages = str(arr[i])
        curr_tables = tabula.io.read_pdf(pdf, stream=True, pages=curr_pages)[0]
        curr_tables = curr_tables[2:]
        curr_tables.index += len(result_tables) - 2
        result_tables = pd.concat([result_tables, curr_tables], axis=0)

    return result_tables


# Get the data from the pdf file basd on pages.
df = ReadTablePages("Financial_Statements.pdf", np.array([11, 12]))


# print(df)

def ReplaceAndGetCategory(df: pd.DataFrame):
    # df.columns = [col if not col.startswith('Unnamed') else '' for col in df.columns]
    cols = list(df.columns)
    for i, col in enumerate(cols[:]):  # exclude the last column since it has no next column
        # Check if the column is 'Unnamed' and the next column is either 'Group' or 'Company'
        if "Unnamed" in col and (cols[i - 1] != "Unnamed"):
            # Rename the column to the name of the next column
            cols[i] = cols[i - 1]
    df.columns = cols
    return df


# print(ReplaceAndGetCategory(df))
df = ReplaceAndGetCategory(df)
print(df)
# first_row = df.loc[2]
# print(first_row)
print()


def ExtractIncomeData(df: pd.DataFrame):
    cols1 = df.columns
    cols2 = df.loc[0]
    cols3 = df.loc[1]
    result_df = pd.concat([cols2, cols3], axis=1)
    # print(pd.concat([cols2, cols3], axis=1).T)
    first_column = df.iloc[:, 0]
    # print(result_df)

    words_to_search = ["revenue", "profit before tax", "loss before tax", "Interest Income", "Finance Income",
                       "financial year ended", "Interest Income / Finance Income", "Profit/(Loss) Before Tax",
                       "(Loss)/Profit Before Tax"]

    print(words_to_search[7].lower())

    for i in range(2, len(first_column)):
        # print(first_column.loc[i], i)

        if not isinstance(first_column.loc[i], str):
            continue

        features = first_column.loc[i].lower().split()

        array_words = [element.lower().split() for element in words_to_search]

        for words in array_words:
            if all(word in features for word in words):
                curr_col = df.loc[i]
                result_df = pd.concat([result_df, curr_col], axis=1)
                break

    return result_df.T


income_df = ExtractIncomeData(df)
print(income_df)


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


# Extract the data from the dataset related to debt from current & non-current liabilities
def ExtractDebtData(df: pd.DataFrame):
    cols1 = df.columns
    cols2 = df.loc[0]
    cols3 = df.loc[1]
    result_df = pd.concat([cols2, cols3], axis=1)
    # print(pd.concat([cols2, cols3], axis=1).T)
    first_column = df.iloc[:, 0]
    # print(result_df)

    words_to_search = {'features': ["cash and cash equivalent", "cash and bank balances", "cash at bank",
                                    "cash held under housing development accounts",
                                    "cash placed in conventional accounts and instruments",
                                    "cash", "deposit with licensed bank", "investment", "money market instrument",
                                    "other cash equivalents",
                                    "deposits", "investment in cash funds", "resale agreement", "short term deposits",
                                    "short term funds",
                                    "short term investments", "unit trust funds", "total assets", "assets", "equity",
                                    "borrowing",
                                    "bank borrowings",
                                    "bank overdrafts", "bankers' acceptance", "bill discounting", "bill payables",
                                    "bridging loans",
                                    "capital securities", "commercial papers", "commodity financing",
                                    "conventional bonds", "debentures",
                                    "deferred liability", "export credit refinancing",
                                    "hire purchase payables",
                                    "invoice financing",
                                    "lease liabilities", "loan stocks",
                                    "loans and borrowings", "revenue", "profit before tax", "loss before tax",
                                    "Interest Income", "Finance Income",
                                    "financial year ended", "Interest Income / Finance Income",
                                    "Profit/(Loss) Before Tax",
                                    "(Loss)/Profit Before Tax"],
                       'is_cash_related': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                                           1, 1,
                                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]}

    print(len(words_to_search['features']), len(words_to_search['is_cash_related']))

    # words_df = pd.DataFrame(words_to_search)
    # print(words_df)
    X = words_to_search['features']
    y = words_to_search['is_cash_related']

    logistic_model = LogisticSGDModel()
    logistic_model.fit(X, y)

    for i in range(2, len(first_column)):
        non_current_liabilities = False
        current_liabilities = False

        if not isinstance(first_column.loc[i], str):
            continue

        features = first_column.loc[i]
        predicted_value = logistic_model.predict([features])
        # print(features, predicted_value)

        if predicted_value == 1:
            curr_col = df.loc[i]
            result_df = pd.concat([result_df, curr_col], axis=1)

        else:
            continue

    return result_df.T


debt_df = ExtractDebtData(df)
print(debt_df)
print()


# Extract data from the dataset related to cash
def ExtractCashData(df: pd.DataFrame):
    cols1 = df.columns
    cols2 = df.loc[0]
    cols3 = df.loc[1]
    result_df = pd.concat([cols2, cols3], axis=1)
    # print(pd.concat([cols2, cols3], axis=1).T)
    first_column = df.iloc[:, 0]
    # print(result_df)

    words_to_search = {'features': ["cash and cash equivalent", "cash and bank balances", "cash at bank",
                                    "cash held under housing development accounts",
                                    "cash placed in conventional accounts and instruments",
                                    "cash", "deposit with licensed bank", "investment", "money market instrument",
                                    "other cash equivalents",
                                    "deposits", "investment in cash funds", "resale agreement", "short term deposits",
                                    "short term funds",
                                    "short term investments", "unit trust funds", "total assets", "assets", "equity",
                                    "borrowing",
                                    "bank borrowings",
                                    "bank overdrafts", "bankers' acceptance", "bill discounting", "bill payables",
                                    "bridging loans",
                                    "capital securities", "commercial papers", "commodity financing",
                                    "conventional bonds", "debentures",
                                    "deferred liability", "export credit refinancing",
                                    "hire purchase payables",
                                    "invoice financing",
                                    "lease liabilities", "loan stocks",
                                    "loans and borrowings", "revenue", "profit before tax", "loss before tax",
                                    "Interest Income", "Finance Income",
                                    "financial year ended", "Interest Income / Finance Income",
                                    "Profit/(Loss) Before Tax",
                                    "(Loss)/Profit Before Tax"],
                       'is_cash_related': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2]}

    print(len(words_to_search['features']), len(words_to_search['is_cash_related']))

    X = words_to_search['features']
    y = words_to_search['is_cash_related']

    logistic_model = LogisticSGDModel()
    logistic_model.fit(X, y)

    for i in range(2, len(first_column)):

        if not isinstance(first_column.loc[i], str):
            continue

        features = first_column.loc[i]
        predicted_value = logistic_model.predict([features])

        if predicted_value == 1:
            curr_col = df.loc[i]
            result_df = pd.concat([result_df, curr_col], axis=1)

        else:
            continue

    return result_df.T


cash_df = ExtractCashData(df)
print(cash_df)
# LOL


