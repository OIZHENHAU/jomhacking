import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
import PyPDF2
import tabula
from PyPDF2 import PdfReader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from tabula.io import read_pdf

# Function to extract text from PDF
'''def extract_text_from_pdf(pdf: PyPDF2.PdfReader):
    text = ""
    totalPages = len(pdf.pages)

    for i in range(totalPages):
        page = pdf.pages[i]

        text += page.extract_text().lower()
        text += "\n"

    return text


# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text


reader = PdfReader("Financial_Statements.pdf")

# Sample input string
input_string1 = "financial"
input_string2 = "gambling"

# Extract text from PDF
pdf_text = extract_text_from_pdf(reader)

# Preprocess text
pdf_text = preprocess_text(pdf_text)
# print(pdf_text)
arr_text = pdf_text.split()

# print(len(pdf_text), len(arr))

input_string1 = preprocess_text(input_string1)
input_string2 = preprocess_text(input_string2)


# Label data
def isInPDFFile(str1: str, str2: str):
    label = 0

    if str1 in str2:
        label = 1

    return label


label1 = isInPDFFile(input_string1, pdf_text)
label2 = isInPDFFile(input_string2, pdf_text)


# A simple tokenization and encoding function
def text_to_tensor(text, vocab=None):
    # Tokenize the text (simple split by space, consider using more sophisticated tokenization)
    tokens = text.split()

    # If no vocab is provided, create one
    if vocab is None:
        vocab = {token: i for i, token in enumerate(sorted(set(tokens)))}

    # Encode tokens using vocab
    encoded = [vocab[token] for token in tokens if token in vocab]

    # Convert to tensor
    tensor = torch.tensor(encoded, dtype=torch.float)

    return tensor, vocab


# Convert text to tensor
tensor1, vocab1 = text_to_tensor(pdf_text)

print(tensor1)
print(len(tensor1))
print()


# Define Neural Network architecture
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


# Define training data
# Assume you have X_train and y_train prepared from labeled data

# Define model parameters
input_size = len(tensor1)  # Length of PDF text
hidden_size = 128
output_size = 1

# Initialize model
model = SimpleNN(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 10
for epoch in range(epochs):
    # Convert data to tensors
    inputs = tensor1
    labels = torch.Tensor([label1])

    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(inputs)

    # Calculate loss
    loss = criterion(outputs, labels)

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()

    # print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")


def predict_next_input(model: nn.Module, tensor: torch.Tensor, input_string: str):
    # Preprocess the input string
    input_string = preprocess_text(input_string)

    # Convert the input string to a tensor
    input_tensor = tensor

    # Perform forward pass to get predictions
    with torch.no_grad():
        prediction = model(tensor)

        # Convert the prediction to a probability (0 or 1)
        print(prediction.item())
        predicted_label = 1 if prediction.item() >= 0.5 else 0

        # Output the prediction
        if predicted_label == 1:
            print("The text in the PDF matches the input string.")
        else:
            print("The text in the PDF does not match the input string.")



# Call the predict_next_input function
# predict_next_input(model, tensor1, "")
'''

# Assuming df is your DataFrame containing the data
data = {
    "col1": [1, 2, "(14324)", 4],
    "col2": ["string", 6, 7, 8],
    "col3": [9, 10, 11, "string"]
}

df = pd.DataFrame(data)

# Check if any value in each row is a string
contains_string = df.apply(lambda row: row.astype(str).str.contains('[a-zA-Z]').any(), axis=1)

print("Rows containing string values:")
print(df[contains_string])

# Assuming df is your DataFrame containing the data
data = {
    "col1": [1, 2, "(14324)", 4],
    "col2": ["(14324)", 6, 7, 8],
    "col3": [9, 10, 11, "(14324)"]
}

df = pd.DataFrame(data)

# Remove brackets from elements in all columns
df = df.applymap(lambda x: str(x).replace('(', '').replace(')', ''))

print("DataFrame after removing brackets:")
print(df)

# Example 2D NumPy array of type string
array_of_strings = np.array([["1", "2", "3"],
                             ["4", "5", "6"],
                             ["7", "8", "9"]])

# Convert the array of strings to integers
array_of_integers = array_of_strings.astype(int)

print("2D array of integers:")
print(array_of_integers)

# Example 2D numpy array with string representations of floating-point numbers
str_array_2d = np.array([['102509.6', '102448.1'], ['16592.5', '16473.5']])

# Convert the strings to floats
float_array_2d = str_array_2d.astype(float)

# Round the floats to two decimal places
rounded_array_2d = np.round(float_array_2d, 2)

print(rounded_array_2d)

# Creating a sample DataFrame
data = {
    'Name': [14, 23, 7, 19],
    'Age': [28, 34, 29, 32],
    'City': [53, 21, 16, 11]
}

df = pd.DataFrame(data)
print(df)

# Converting the DataFrame to JSON
json_str = df.to_json(orient='records')

print(json_str)
