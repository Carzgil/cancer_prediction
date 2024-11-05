import pandas as pd
import numpy as np
from collections import Counter
import os

# Load the processed dataset
df = pd.read_csv('data/Breast_Cancer_dataset_processed.csv')

# Split the data into features and target
X = df.drop('target', axis=1).values  # Replace 'target' with your actual target column name
y = df['target'].values

# Split the data into training and testing sets
def train_test_split(X, y, test_size=0.2):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    test_set_size = int(len(X) * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Implement KNN
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(X_train, y_train, X_test, k=3):
    predictions = []
    for test_point in X_test:
        distances = [euclidean_distance(test_point, x) for x in X_train]
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        predictions.append(most_common[0][0])
    return predictions

# Evaluate the model
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

y_pred = knn_predict(X_train, y_train, X_test)
acc = accuracy(y_test, y_pred)

# Save the model and performance
os.makedirs('Model', exist_ok=True)
os.makedirs('Performance', exist_ok=True)

# Save the model (in this case, just the training data and k value)
np.save('Model/knn_model.npy', {'X_train': X_train, 'y_train': y_train, 'k': 3})

# Save the performance
with open('Performance/knn_performance.txt', 'w') as f:
    f.write(f'Accuracy: {acc}\n')
