import math

import PyPDF2
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from PyPDF2 import PdfReader
import tabula
from tabula.io import read_pdf
import pandas as pd

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


def ExtractCashData(df: pd.DataFrame):
    cols1 = df.columns
    cols2 = df.loc[0]
    cols3 = df.loc[1]
    result_df = pd.concat([cols2, cols3], axis=1)
    # print(pd.concat([cols2, cols3], axis=1).T)
    first_column = df.iloc[:, 0]
    # print(result_df)

    words_to_search = ["cash and cash equivalent", "cash and bank balances", "cash at bank",
                       "deposits", "investment in cash funds", "resale agreement", "short term",
                       "unit trust funds", "total assets"]

    for i in range(2, len(first_column)):
        # print(first_column.loc[i], i)

        if not isinstance(first_column.loc[i], str):
            continue

        features = first_column.loc[i].split()

        array_words = [element.lower().split() for element in words_to_search]

        for words in array_words:
            if any(word in features for word in words):
                curr_col = df.loc[i]
                result_df = pd.concat([result_df, curr_col], axis=1)
                break

    return result_df.T


cash_df = ExtractCashData(df)
# print(cash_df)


def ExtractDebtData(df: pd.DataFrame):
    cols1 = df.columns
    cols2 = df.loc[0]
    cols3 = df.loc[1]
    result_df = pd.concat([cols2, cols3], axis=1)
    # print(pd.concat([cols2, cols3], axis=1).T)
    first_column = df.iloc[:, 0]
    # print(result_df)

    words_to_search = ["cash and cash equivalent", "cash and bank balances", "cash at bank",
                       "deposits", "investment in cash funds", "resale agreement", "short term",
                       "unit trust funds"]

    for i in range(2, len(first_column)):
        # print(first_column.loc[i], i)

        if not isinstance(first_column.loc[i], str):
            continue

        features = first_column.loc[i].split()

        array_words = [element.lower().split() for element in words_to_search]

        for words in array_words:
            if any(word in features for word in words):
                curr_col = df.loc[i]
                result_df = pd.concat([result_df, curr_col], axis=1)
                break

    return result_df.T


debt_df = ExtractDebtData(df)
# print(debt_df)
