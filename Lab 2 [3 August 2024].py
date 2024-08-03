#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Question 1 2 3

import numpy as np
from sklearn.linear_model import LogisticRegression

# Purchase data attributes
A = np.array([
    [20, 6, 2],
    [16, 3, 6],
    [27, 6, 2],
    [19, 1, 2],
    [24, 4, 2],
    [22, 1, 5],
    [15, 4, 2],
    [18, 4, 2],
    [21, 1, 4],
    [16, 2, 4]
])

#Purchase Payment 

C = np.array([386, 289, 393, 110, 280, 167, 271, 274, 148, 198])

#To find the dimensionality of the shape
dimensionality = A.shape[1]

#To find vectors in this vector space
num_vectors = A.shape[0]

# Rank of matrix A
rank_A = np.linalg.matrix_rank(A)

# Using pseudo-inverse to find the cost of each product
pseudo_inverse= np.linalg.pinv(A)
# Multiplication of matrix
cost = pseudo_inverse @ C

#Question 1
print(f"Dimensionality of the vector space: {dimensionality}")
print(f"Number of vectors in this vector space: {num_vectors}")
print(f"Rank of Matrix A: {rank_A}")

#Question 2
print(f"Cost of each product (Candies, Mangoes, Milk Packets): {cost}")

#Question 3
labels = np.where(C > 200, 'RICH', 'POOR')
labels_numeric = np.where(C > 200, 1, 0)

# Develop a classifier model to categorize customers
classifier = LogisticRegression()
classifier.fit(A, labels_numeric)

# Predict the categories for the customers
predictions = classifier.predict(A)
predictions_labels = np.where(predictions == 1, 'RICH', 'POOR')

print(f"Actual Labels: {labels}")
print(f"Predicted Labels: {predictions_labels}")


# In[12]:


#Question 4

import pandas as pd
import statistics
import matplotlib.pyplot as plt

data = {
    'Date': [
        'Jul 14, 2020', 'Jul 13, 2020', 'Jul 10, 2020', 'Jul 09, 2020', 'Jul 08, 2020',
        'Jul 07, 2020', 'Jul 06, 2020', 'Jul 03, 2020', 'Jul 02, 2020', 'Jul 01, 2020'
    ],
    'Month': ['Jul'] * 10,
    'Day': ['Tue', 'Mon', 'Fri', 'Thu', 'Wed', 'Tue', 'Mon', 'Fri', 'Thu', 'Wed'],
    'Close Price': [
        1362.15, 1397.35, 1400.95, 1385.05, 1390.10, 1397.40, 1400.75, 1405.10, 1412.35, 1363.05
    ],
    'Change%': [-2.52, -0.26, 1.15, -0.36, -0.52, -0.24, -0.31, -0.51, 3.62, 0.32]
}

df = pd.DataFrame(data)

#1: Calculate the mean and variance of the Close Price data
mean_close_price = statistics.mean(df['Close Price'])
variance_close_price = statistics.variance(df['Close Price'])

print(f"Mean of Close Price: {mean_close_price}")
print(f"Variance of Close Price: {variance_close_price}")
print()

#2 Select price data for all Wednesdays and calculate the sample mean
wednesday_prices = df[df['Day'] == 'Wed']['Close Price']
mean_wednesday_price = statistics.mean(wednesday_prices)

print(f"Mean of Close Price on Wednesdays: {mean_wednesday_price}")
print ()

#3 Select price data for the month of April and calculate the sample mean

april_prices = df['Close Price']
mean_april_price = statistics.mean(april_prices)

print(f"Mean of Close Price in April (simulated with provided data): {mean_april_price}")
print()

#4 Find the probability of making a loss over the stock from the Chg%

loss_probability = len(df[df['Change%'] < 0]) / len(df)
print(f"Probability of making a loss: {loss_probability}")
print()

# Task 5: Calculate the probability of making a profit on Wednesday
wednesday_data = df[df['Day'] == 'Wed']
wednesday_profit_probability = len(wednesday_data[wednesday_data['Change%'] > 0]) / len(wednesday_data)
print(f"Probability of making a profit on Wednesday: {wednesday_profit_probability}")
print()

# Task 6: Calculate the conditional probability of making profit, given that today is Wednesday
overall_profit_probability = len(df[df['Change%'] > 0]) / len(df)
conditional_profit_probability = wednesday_profit_probability / overall_profit_probability
print(f"Conditional probability of making profit given it's Wednesday: {conditional_profit_probability}")
print()

# Task 7: Make a scatter plot of Chg% data against the day of the week
plt.scatter(df['Day'], df['Change%'], color='orange')
plt.title('Change% vs Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Change%')
plt.show()


# In[17]:


#Question 5 6 7 8 9 10

import pandas as pd

# Load the data from the Excel file
file_path = r'C:\Users\tarun\Lab Codes\thyroid.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Display the first few rows of the dataframe
print(df.head())


# In[18]:


import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Study each attribute and associated values
print(df.info())
print(df.describe(include='all'))

# Identify the datatype for each attribute
data_types = df.dtypes
print(data_types)

# Check for missing values
missing_values = df.isnull().sum()
print(missing_values)

# Study data range for numeric variables
numeric_columns = df.select_dtypes(include=[np.number]).columns
numeric_range = df[numeric_columns].describe()
print(numeric_range)

# Study presence of outliers
for column in numeric_columns:
    plt.figure()
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot for {column}')
    plt.show()

# Calculate mean and variance (or standard deviation) for numeric variables
mean_variance = df[numeric_columns].agg(['mean', 'var'])
print(mean_variance)


# In[19]:


#Data Imputation

# Mean for numeric attributes without outliers
for column in numeric_columns:
    if not df[column].isnull().any():
        continue
    if df[column].isna().sum() > 0:
        if any(df[column] > df[column].mean() + 3 * df[column].std()) or any(df[column] < df[column].mean() - 3 * df[column].std()):
            # Median for numeric attributes with outliers
            df[column].fillna(df[column].median(), inplace=True)
        else:
            # Mean for numeric attributes without outliers
            df[column].fillna(df[column].mean(), inplace=True)

# Mode for categorical attributes
categorical_columns = df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    df[column].fillna(df[column].mode()[0], inplace=True)

print(df.isnull().sum())


# In[20]:


#Data Normalization / Scaling

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Example normalization for numeric columns
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

print(df.head())


# In[23]:


print(df.columns)


# In[35]:


#Data Normalization
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load data
file_path = r'C:\Users\tarun\Lab Codes\thyroid.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Identify numeric columns
numeric_columns = ['age']  # Assuming 'age' is the only numeric column

# Normalize numeric columns
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Convert categorical columns to numeric using LabelEncoder
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
categorical_columns = [col for col in categorical_columns if col not in numeric_columns]  # Exclude numeric columns

# Encoding categorical columns
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))

# Replace missing values (if there are any non-standard missing values like '?')
df.replace('?', pd.NA, inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)  # Impute missing values with the mode

print(df.head())


# In[36]:


import pandas as pd
import numpy as np

# Load the data from the Excel file
file_path = r'C:\Users\tarun\Lab Codes\thyroid.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Print column names to check for correct names
print("Column names in the DataFrame:", df.columns)

# Define the correct binary columns (adjust as needed based on your actual data)
binary_columns = ['on thyroxine', 'query on thyroxine', 'on antithyroid medication',
                   'sick', 'pregnant', 'thyroid surgery', 'I131 treatment',
                   'query hypothyroid', 'query hyperthyroid', 'lithium', 
                   'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured']

# Take the first 2 observation vectors
obs1 = df.iloc[0]
obs2 = df.iloc[1]

# Check if columns exist in DataFrame
missing_cols = [col for col in binary_columns if col not in df.columns]
if missing_cols:
    raise KeyError(f"Columns missing from DataFrame: {missing_cols}")

# Extract binary attributes for both observations from the DataFrame
obs1_binary = df.loc[0, binary_columns]
obs2_binary = df.loc[1, binary_columns]

# Convert to binary if necessary (ensure they are actually binary 0/1)
obs1_binary = obs1_binary.apply(lambda x: 1 if x.lower() == 't' else 0)
obs2_binary = obs2_binary.apply(lambda x: 1 if x.lower() == 't' else 0)

# Debugging: Print extracted binary attributes
print("obs1_binary:\n", obs1_binary)
print("obs2_binary:\n", obs2_binary)

# Calculate JC and SMC
f11 = np.sum((obs1_binary == 1) & (obs2_binary == 1))
f00 = np.sum((obs1_binary == 0) & (obs2_binary == 0))
f01 = np.sum((obs1_binary == 0) & (obs2_binary == 1))
f10 = np.sum((obs1_binary == 1) & (obs2_binary == 0))

# Debugging: Print values of f11, f00, f01, and f10
print(f"f11: {f11}, f00: {f00}, f01: {f01}, f10: {f10}")

# Compute Jaccard Coefficient (JC)
denominator_jc = (f01 + f10 + f11)
if denominator_jc != 0:
    jc = f11 / denominator_jc
else:
    jc = np.nan  # or some default value if you prefer

# Compute Simple Matching Coefficient (SMC)
denominator_smc = (f00 + f01 + f10 + f11)
if denominator_smc != 0:
    smc = (f11 + f00) / denominator_smc
else:
    smc = np.nan  # or some default value if you prefer

print(f"Jaccard Coefficient (JC): {jc}")
print(f"Simple Matching Coefficient (SMC): {smc}")


# In[28]:


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

# Load the data from the Excel file
file_path = r'C:\Users\tarun\Lab Codes\thyroid.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Print column names to check for correct names
print("Column names in the DataFrame:", df.columns)

# Define columns to use for cosine similarity
columns_to_use = df.columns  # Use all columns or specify a subset as needed

# Preprocess the data
# Convert categorical columns to numerical values using LabelEncoder
df_preprocessed = df.copy()

# Apply LabelEncoder to categorical columns
for column in df_preprocessed.columns:
    if df_preprocessed[column].dtype == 'object':
        le = LabelEncoder()
        df_preprocessed[column] = df_preprocessed[column].astype(str)  # Ensure data is in string format
        df_preprocessed[column] = le.fit_transform(df_preprocessed[column])

# Handle missing values (e.g., replacing with 0 or mean)
df_preprocessed.fillna(0, inplace=True)

# Take the complete vectors for the first two observations
obs1_complete = df_preprocessed.iloc[0].values.reshape(1, -1)
obs2_complete = df_preprocessed.iloc[1].values.reshape(1, -1)

# Calculate Cosine similarity
cosine_sim = cosine_similarity(obs1_complete, obs2_complete)[0][0]

print(f"Cosine Similarity: {cosine_sim}")


# In[40]:


#A10 Heat Map 
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the data
file_path = r'C:\Users\tarun\Lab Codes\thyroid.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Take the first 20 observations
df_20 = df.head(20)

# Convert categorical data to binary and handle missing values
df_20_binary = df_20.copy()
for column in df_20_binary.columns:
    if df_20_binary[column].dtype == 'object':
        le = LabelEncoder()
        df_20_binary[column] = df_20_binary[column].astype(str)
        df_20_binary[column] = le.fit_transform(df_20_binary[column])

df_20_binary.fillna(0, inplace=True)

# Function to calculate JC and SMC
def calculate_similarity_matrix(df_binary, similarity_type='JC'):
    n = df_binary.shape[0]
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            obs1_binary = df_binary.iloc[i].values
            obs2_binary = df_binary.iloc[j].values
            
            f11 = np.sum((obs1_binary == 1) & (obs2_binary == 1))
            f00 = np.sum((obs1_binary == 0) & (obs2_binary == 0))
            f01 = np.sum((obs1_binary == 0) & (obs2_binary == 1))
            f10 = np.sum((obs1_binary == 1) & (obs2_binary == 0))
            
            if similarity_type == 'JC':
                denominator = (f01 + f10 + f11)
                similarity = f11 / denominator if denominator != 0 else np.nan
            elif similarity_type == 'SMC':
                denominator = (f00 + f01 + f10 + f11)
                similarity = (f11 + f00) / denominator if denominator != 0 else np.nan
            else:
                raise ValueError("Unsupported similarity type")
            
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # Symmetric matrix
    
    return similarity_matrix

# Calculate JC and SMC matrices
jc_matrix = calculate_similarity_matrix(df_20_binary, similarity_type='JC')
smc_matrix = calculate_similarity_matrix(df_20_binary, similarity_type='SMC')

# Calculate Cosine Similarity matrix
def calculate_cosine_similarity_matrix(df):
    n = df.shape[0]
    cosine_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            obs1 = df.iloc[i].values.reshape(1, -1)
            obs2 = df.iloc[j].values.reshape(1, -1)
            cosine_sim = cosine_similarity(obs1, obs2)[0][0]
            cosine_matrix[i, j] = cosine_sim
            cosine_matrix[j, i] = cosine_sim  # Symmetric matrix
    
    return cosine_matrix

cosine_matrix = calculate_cosine_similarity_matrix(df_20_binary)

# Plot Jaccard Coefficient heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(jc_matrix, annot=True, cmap='viridis', cbar=True, square=True, fmt='.2f')
plt.title('Jaccard Coefficient Heatmap')
plt.show()

# Plot Simple Matching Coefficient heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(smc_matrix, annot=True, cmap='viridis', cbar=True, square=True, fmt='.2f')
plt.title('Simple Matching Coefficient Heatmap')
plt.show()

# Plot Cosine Similarity heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(cosine_matrix, annot=True, cmap='viridis', cbar=True, square=True, fmt='.2f')
plt.title('Cosine Similarity Heatmap')
plt.show()

