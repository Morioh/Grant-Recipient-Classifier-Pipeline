# Pipeline Definition and Streamlit UI
# Pipeline Definition

import streamlit as st
import pandas as pd
import subprocess

file_path = "Grant-Recipient-Classifier-Pipeline/data/Applicant Data - T, T & V.csv"
# Function to load data


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Function to preprocess data


def preprocess_data():
    # Run the script as an external process
    subprocess.run(['python', 'preprocessing.py'])

# Function to train model


def train_model(X_train, y_train):
    # Run the script as an external process
    subprocess.run(['python', 'model.py'])

# Function to evaluate model


def evaluate_model(model, X_test, y_test):
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    return test_accuracy

# Streamlit UI


# Title
st.title('Pipeline Process Visualization')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Step 1: Load Data
    st.header('1. Load Data')
    data = load_data(uploaded_file)
    st.write(data.head())

    # Step 2: Preprocess Data
    st.header('2. Preprocess Data')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    st.write('Data has been split into training and testing sets.')

    # Step 3: Train Model
    st.header('3. Train Model')
    model = train_model(X_train, y_train)
    st.write('Model training completed.')

    # Step 4: Evaluate Model
    st.header('4. Evaluate Model')
    accuracy = evaluate_model(model, X_test, y_test)
    st.write(f'Model Accuracy: {accuracy:.2%}')
