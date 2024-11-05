import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
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

# Detect and handle outliers using z-score
from scipy import stats
z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
df = df[(z_scores < 3).all(axis=1)]

# Standardization
scaler = StandardScaler()
df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(df.select_dtypes(include=[np.number]))

# Save the processed data
df.to_csv('data/Breast_Cancer_dataset_processed.csv', index=False)
