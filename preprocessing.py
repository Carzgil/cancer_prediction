import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# Load the dataset
df = pd.read_csv('data/Breast_Cancer_dataset.csv')

# Print column names to verify
print("Columns in the dataset:", df.columns)

# Check unique values in 'Race' before encoding
print("Unique values in 'Race' before encoding:", df['Race'].unique())

# Handle missing values
# For numerical columns
num_imputer = SimpleImputer(strategy='mean')
df[df.select_dtypes(include=[np.number]).columns] = num_imputer.fit_transform(df.select_dtypes(include=[np.number]))

# For categorical columns
cat_imputer = SimpleImputer(strategy='most_frequent')
df[df.select_dtypes(include=[object]).columns] = cat_imputer.fit_transform(df.select_dtypes(include=[object]))

# Encode the 'Status' column
status_encoder = LabelEncoder()
df['Status'] = status_encoder.fit_transform(df['Status'])

# Convert other categorical columns to numerical
label_encoders = {}
categorical_columns = ['Race', 'Marital Status', 'T Stage', 'N Stage', '6th Stage', 
                       'differentiate', 'Grade', 'A Stage', 'Estrogen Status', 'Progesterone Status']

for column in categorical_columns:
    if column in df.columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    else:
        print(f"Column '{column}' not found in the dataset.")

# Check unique values in 'Race' after encoding
print("Unique values in 'Race' after encoding:", df['Race'].unique())

# Detect and handle outliers using z-score
from scipy import stats
z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number]).drop(columns=['Status'])))
df = df[(z_scores < 3).all(axis=1)]

# Standardization (excluding 'Status')
scaler = StandardScaler()
df[df.select_dtypes(include=[np.number]).columns.difference(['Status'])] = scaler.fit_transform(
    df.select_dtypes(include=[np.number]).drop(columns=['Status']))

# Save the processed data
df.to_csv('data/processed.csv', index=False)
