import math
import PyPDF2
import numpy as np
import tabula
import re
import pandas as pd
import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PyPDF2 import PdfReader
from tabula.io import read_pdf
from LogisticSGDModel import LogisticSGDModel
from NeuralNetworkModel import NeuralNetworkModel

# Set the output in the terminal with max row and column
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Extrac the text from the page of the pdf file
reader = PdfReader("Financial_Statements.pdf")
page = reader.pages[2]
text = page.extract_text()
# print(text)

# To get the total pages from the PDF file
total_Pages = len(reader.pages)
# print(f"Total Pages: {totalPages}")


# Read the specific pages from the pdf file (exp: pg 11)
tables = tabula.io.read_pdf("Financial_Statements.pdf", stream=True, pages="13")
# print(tables)


# Extract every page in the PDF file
def extract_text_from_pdf(pdf_path: PyPDF2.PdfReader):
    text = ""
    totalPages = len(pdf_path.pages)

    for i in range(totalPages):
        page = pdf_path.pages[i]

        text += page.extract_text()
        text += "\n"

    return text


# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text


# Check if the current string contains in another string
def isInPDFFile(str1: str, str2: str):
    label = 0

    if str1 in str2:
        label = 1

    return label


# Read the given pages in the PDF file
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

# Replace the unnamned column into specific label in the dataset
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
# print(df)
# print()


# Extract the data related to income from the dataset
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


# print(income_df)


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


def is_almost_match(s1: str, s2: str, threshold=3):
    s1 = s1.lower()
    s2 = s2.lower()
    distance = levenshtein_distance(s1, s2)
    # print(distance)
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
                                    "short term investments", "unit trust funds", "total assets", "assets",
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
                                    "(Loss)/Profit Before Tax", "equity", "equity investment", "stocks", "bonds",
                                    "real estate", "commodities", "collectibles", "mutual funds",
                                    "exchange-traded funds",
                                    "peer-to-peer lending", "cryptocurrencies", "hedge funds",
                                    "investments in subsidiaries", "investments in associates",
                                    "conventional banking", "conventional lending", "conventional banking and lending",
                                    "gambling", "liquor and liquor-related activities",
                                    "non-halal food", "non-halal beverage", "non-halal food and beverage",
                                    "tobacco and tobacco-related activities", "interest income",
                                    "interest income from conventional accounts", "interest income from instrument",
                                    "dividends from non-compliant investments", "Shariah non-compliant entertainment",
                                    "share trading", "stockbroking business", "rental received from non-compliant activities",
                                    "rental received from Shariah non-compliant activities"],

                       'is_cash_related': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                                           0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
                                           3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                           4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                           5, 5, 5, 5]}

    # print(len(words_to_search['features']), len(words_to_search['is_cash_related']))

    # words_df = pd.DataFrame(words_to_search)
    # print(words_df)
    X = words_to_search['features']
    y = words_to_search['is_cash_related']

    logistic_model = LogisticSGDModel()
    logistic_model.fit(X, y)

    non_current_liabilities = False
    current_liabilities = False

    for i in range(2, len(first_column)):

        # print(non_current_liabilities, current_liabilities)

        if (non_current_liabilities and current_liabilities) and not isinstance(first_column.loc[i], str):
            break

        if not isinstance(first_column.loc[i], str):
            continue

        features = first_column.loc[i]
        # print(features)

        if non_current_liabilities or current_liabilities:
            # print("case 1")

            if is_almost_match(features, "non current liabilities"):
                # print("case 1a")
                non_current_liabilities = not non_current_liabilities
                continue

            elif is_almost_match(features, "current liabilities"):
                # print("case 1b")
                current_liabilities = not current_liabilities
                continue

            else:
                # print("case 1c")
                predicted_value = logistic_model.predict([features])
                # print(features, predicted_value)

                if predicted_value == 0:
                    curr_col = df.loc[i]
                    result_df = pd.concat([result_df, curr_col], axis=1)

                else:
                    continue

        elif is_almost_match(features, "non current liabilities"):
            # print("case 2")
            non_current_liabilities = not non_current_liabilities
            continue

        elif is_almost_match(features, "current liabilities"):
            # print("case 3")
            current_liabilities = not current_liabilities
            continue

        else:
            # print("case 4")
            continue

    return result_df.T


debt_df = ExtractDebtData(df)
# print(debt_df)
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
                                    "short term investments", "unit trust funds", "total assets", "assets",
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
                                    "(Loss)/Profit Before Tax", "equity", "equity investment", "stocks", "bonds",
                                    "real estate", "commodities", "collectibles", "mutual funds",
                                    "exchange-traded funds",
                                    "peer-to-peer lending", "cryptocurrencies", "hedge funds",
                                    "investments in subsidiaries", "investments in associates",
                                    "conventional banking", "conventional lending", "conventional banking and lending",
                                    "gambling", "liquor and liquor-related activities",
                                    "non-halal food", "non-halal beverage", "non-halal food and beverage",
                                    "tobacco and tobacco-related activities", "interest income",
                                    "interest income from conventional accounts", "interest income from instrument",
                                    "dividends from non-compliant investments", "Shariah non-compliant entertainment",
                                    "share trading", "stockbroking business",
                                    "rental received from non-compliant activities",
                                    "rental received from Shariah non-compliant activities"],

                       'is_cash_related': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                                           0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
                                           3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                           4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                           5, 5, 5, 5]}

    # print(len(words_to_search['features']), len(words_to_search['is_cash_related']))

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
# print(cash_df)
# LOL
print()


# Calculate the percentage of the cash against total assets
def ComputeCashRatio(df: pd.DataFrame):
    cols0 = df.loc[0]
    cols1 = df.loc[1]
    total_assets = df.loc[0]
    total_assets_index = 0

    for index, row in df.iloc[2:].iterrows():

        if is_almost_match(row[0], "total assets"):
            total_assets_index = index

    new_df = df.drop(columns=df.columns[df.columns.str.contains('Unnamed')])
    new_df = new_df.drop(0)
    new_df = new_df.drop(1)

    new_df = new_df.fillna(0)
    new_df = new_df.replace('-', 0)
    # print(new_df)

    # Remove brackets from elements in all columns
    new_df = new_df.applymap(lambda x: str(x).replace('(', '').replace(')', '').replace(',', ''))

    total_assets = new_df.loc[total_assets_index]
    total_assets_list = np.array(total_assets.values.tolist())
    total_assets_list = total_assets_list.astype(int)

    new_df = new_df.drop(total_assets_index)

    # Convert the DataFrame into a list of NumPy arrays
    numpy_arrays = np.array(new_df.values.tolist())
    numpy_arrays = numpy_arrays.astype(int)

    sum_up_array = np.sum(numpy_arrays, axis=0)
    percentage_result = sum_up_array / total_assets_list * 100

    return percentage_result, total_assets_list


percentage_result, total_assets = ComputeCashRatio(cash_df)

print()
# print(total_assets)
# print()
# print(percentage_result)


# Calculate the percentage of the debt against total assets
def ComputeDebtRatio(df: pd.DataFrame, total_assets: np.ndarray):
    cols0 = df.loc[0]
    cols1 = df.loc[1]

    new_df = df.drop(columns=df.columns[df.columns.str.contains('Unnamed')])
    new_df = new_df.drop(0)
    new_df = new_df.drop(1)

    new_df = new_df.fillna(0)
    new_df = new_df.replace('-', 0)

    # Remove brackets from elements in all columns
    new_df = new_df.applymap(lambda x: str(x).replace('(', '').replace(')', '').replace(',', ''))

    numpy_arrays = np.array(new_df.values.tolist())
    numpy_arrays = numpy_arrays.astype(int)
    # print(numpy_arrays)

    sum_up_array = np.sum(numpy_arrays, axis=0)
    percentage_result = sum_up_array / total_assets

    return percentage_result * 100


percentage_debt = ComputeDebtRatio(debt_df, total_assets)
# print(percentage_debt)



# def Extract5BenchMark(df: pd.DataFrame):

