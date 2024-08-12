import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from google.colab import files
from matplotlib import pyplot as plt
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, regularizers
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Load data from CSV
clean_data = pd.read_csv('/content/Cleaned Applicant Data - T, T & V.csv')

# View the first few rows of the dataset
clean_data.head()

# Summary statistics of the dataset
clean_data.describe()

# Check for missing values in the dataset
clean_data.isnull().sum()

# Count grant classes
clean_data['Grant Classifier'].value_counts()

# Visualize the grant classes
sns.countplot(x='Grant Classifier', clean_data=data)
plt.title('Grant Classifier')
plt.xlabel('Grant Classifier')
plt.ylabel('Count')
plt.show()

# General overview of data distribution
sns.pairplot(clean_data.iloc[:, :], hue="Grant Classifier")
plt.show()

# Encoded distribution of Data
qualification_map = {0: 0, 1: 1}

clean_data['Grant Classifier'] = clean_data['Grant Classifier'].map(
    qualification_map)

# Plot a correlation heatmap
corr_matrix = clean_data.corr()
fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.heatmap(corr_matrix,
                 annot=True,
                 linewidths=0.5,
                 fmt=".2f",
                 cmap="Reds")
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

plt.show()

# Features and labels of the model
X = clean_data.drop(['Grant Classifier'], axis=1)
y = clean_data['Grant Classifier']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

# Fit the scaler on the training features and transform both train and test features
X_train_standardized = scaler.fit_transform(X_train)
X_test_standardized = scaler.transform(X_test)

# Convert back to DataFrame for easier saving and handling
X_train_standardized = pd.DataFrame(
    X_train_standardized, columns=X_train.columns)
X_test_standardized = pd.DataFrame(X_test_standardized, columns=X_test.columns)

# Create the directory if it does not exist
os.makedirs('/data/train', exist_ok=True)
os.makedirs('/data/test', exist_ok=True)

# Save standardized train data
X_train_standardized.to_csv(
    '/data/train/X_train_standardized.csv', index=False)
y_train.to_csv('/data/train/y_train.csv', index=False)

# Save standardized test data
X_test_standardized.to_csv('/data/test/X_test_standardized.csv', index=False)
y_test.to_csv('/data/test/y_test.csv', index=False)

# Download the train files
files.download('/data/train/X_train_standardized.csv')

# Download the train files
files.download('/data/train/y_train.csv')

# Download the test files
files.download('/data/test/X_test_standardized.csv')

# Download the test files
files.download('/data/test/y_test.csv')

X_train.shape

X_train_standardized.shape

y_train.shape

# Define the model
model = models.Sequential()

# Input layer (you may need to adjust the input shape according to your dataset)
model.add(layers.InputLayer(input_shape=(14,)))

# First hidden layer with L2 regularization, batch normalization, and dropout
model.add(layers.Dense(64, activation='relu',
          kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))

# Second hidden layer with L2 regularization, batch normalization, and dropout
model.add(layers.Dense(64, activation='relu',
          kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))

# Third hidden layer with L2 regularization, batch normalization, and dropout
model.add(layers.Dense(64, activation='relu',
          kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))

# Output layer (adjust the units and activation based on your task, e.g., softmax for classification)
model.add(layers.Dense(1, activation='sigmoid'))  # For binary classification

model.summary()

# Compile the model
model.compile(optimizer='adam',
              # For multi-class classification; adjust as needed
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss',  # Metric to monitor (e.g., 'val_loss', 'val_accuracy')
                               patience=5,         # Number of epochs with no improvement after which training will be stopped
                               restore_best_weights=True)  # Restores model weights from the epoch with the best value of the monitored metric

# Define a learning rate scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss',  # Metric to monitor
                                 factor=0.2,          # Factor by which the learning rate will be reduced
                                 # Number of epochs with no improvement after which learning rate will be reduced
                                 patience=3,
                                 min_lr=1e-6)         # Minimum learning rate

history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping, lr_scheduler])
