import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# Load the dataset
df = pd.read_csv('data/Breast_Cancer_dataset.csv')

# Handle missing values
# For numerical columns
num_imputer = SimpleImputer(strategy='mean')
df[df.select_dtypes(include=[np.number]).columns] = num_imputer.fit_transform(df.select_dtypes(include=[np.number]))

# For categorical columns
cat_imputer = SimpleImputer(strategy='most_frequent')
df[df.select_dtypes(include=[object]).columns] = cat_imputer.fit_transform(df.select_dtypes(include=[object]))

# Convert categorical columns to numerical
label_encoders = {}
categorical_columns = ['Race', 'Marital Status', 'T Stage', 'N Stage', '6th Stage', 
                       'differentiate', 'Grade', 'A Stage', 'Estrogen Status', 'Progesterone Status', 'Status']

for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Detect and handle outliers using z-score
from scipy import stats
z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
df = df[(z_scores < 3).all(axis=1)]

# Standardization
scaler = StandardScaler()
df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(df.select_dtypes(include=[np.number]))

# Save the processed data
df.to_csv('data/processed.csv', index=False)
