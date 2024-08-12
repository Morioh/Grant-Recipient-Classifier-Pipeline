import os
import pickle
import numpy as np
import pandas as pd

# Load the data
file_path = '/content/Sample Original Applicant Data.csv'
data = pd.read_csv(file_path)

# Identify columns that won't be useful for the objectives of our analysis to drop
columns_to_drop = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                   10, 14, 16, 17, 24, 25, 27, 31, 33, 34, 35, 36]

# Drop the specified columns
data_dropped = data.drop(data.columns[columns_to_drop], axis=1)

# Create a DataFrame with column indices and names
columns_df = pd.DataFrame({
    'Index': range(len(data_dropped.columns)),
    'Column Name': data_dropped.columns
})

# Mapping of old column names to new column names
rename_mapping = {
    data_dropped.columns[0]: 'Academic Standing',
    data_dropped.columns[1]: 'Disciplinary Standing',
    data_dropped.columns[2]: 'Financial Standing',
    data_dropped.columns[3]: 'Fee balance (USD)',
    data_dropped.columns[4]: 'ALU Grant Status',
    data_dropped.columns[5]: 'ALU Grant Amount',
    data_dropped.columns[6]: 'Previous Alusive Grant Status',
    data_dropped.columns[7]: 'Grant Requested',
    data_dropped.columns[8]: 'Students in Household',
    data_dropped.columns[9]: 'Total Monthly Income',
    data_dropped.columns[10]: 'Total Monthly Income (Assets)',
    data_dropped.columns[11]: 'Household Size',
    data_dropped.columns[12]: 'Household Supporters',
    data_dropped.columns[13]: 'Household Dependants',
    data_dropped.columns[14]: 'Amount Affordable',
}

# Rename the columns
data_renamed = data_dropped.rename(columns=rename_mapping)

# Binary encode Yes/No values
# Replace 'yes' with 1 and 'no' with 0
data_renamed.replace({'Yes': 1, 'No': 0}, inplace=True)

# Encode the values
# Convert the column to strings
data_renamed['ALU Grant Amount'] = data_renamed['ALU Grant Amount'].astype(str)

# Remove 'USD. ' prefix and any spaces, then convert to integer
data_renamed['ALU Grant Amount'] = data_renamed['ALU Grant Amount'].str.replace(
    'USD. ', '', regex=False)

# If you need the value to be an integer:
data_renamed['ALU Grant Amount'] = data_renamed['ALU Grant Amount'].astype(int)

# Mapping the values
mapping = {
    "Under USD. 500": 500,
    "Under 500": 500,
    "Between 500 - 1000": 750,
    "Between USD. 500 and USD. 1500": 1000,
    "Over 1500": 2000,
}

# Replace the values in the DataFrame
data_renamed["Grant Requested"] = data_renamed["Grant Requested"].replace(
    mapping)
data_renamed["Total Monthly Income"] = data_renamed["Total Monthly Income"].replace(
    mapping)
data_renamed["Total Monthly Income (Assets)"] = data_renamed["Total Monthly Income (Assets)"].replace(
    mapping)
data_renamed["Amount Affordable"] = data_renamed["Amount Affordable"].replace(
    mapping)

# Mapping the values
mapping = {
    "Over 4": 4,
    "Under 8": 8,
    "8 and Over": 10,
}

# Replace the values in the DataFrame
data_renamed["Students in Household"] = data_renamed["Students in Household"].replace(
    mapping)
data_renamed["Household Size"] = data_renamed["Household Size"].replace(
    mapping)
data_renamed["Household Dependants"] = data_renamed["Household Dependants"].replace(
    mapping)

# Feature Engineering by Aggregation
# Create a new column that is the sum of two existing columns
data_renamed['Total Monthly Income'] = data_renamed['Total Monthly Income'] + \
    data_renamed['Total Monthly Income (Assets)']

# Drop the column and update the DataFrame
data_renamed = data_renamed.drop('Total Monthly Income (Assets)', axis=1)

# Convert the column to float
data_renamed['Fee balance (USD)'] = data_renamed['Fee balance (USD)'].astype(
    float)

processed_data = data_renamed

# Display the DataFrame
print(processed_data.head())
