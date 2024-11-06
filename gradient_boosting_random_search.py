import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import os
import joblib
import numpy as np

# Load the processed dataset
df = pd.read_csv('data/processed.csv')

# Split the data into features and target
X = df.drop('Status', axis=1).values  # Assuming 'Status' is the target column
y = df['Status'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = GradientBoostingClassifier(random_state=42)

# Define the hyperparameters distribution
param_dist = {
    'n_estimators': np.arange(50, 300, 50),
    'learning_rate': np.linspace(0.01, 0.2, 10),
    'max_depth': np.arange(3, 10, 2),
    'min_samples_split': np.arange(2, 10, 2),
    'min_samples_leaf': np.arange(1, 5, 1)
}

# Perform randomized search
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=50, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

# Best parameters and performance
best_params_gb = random_search.best_params_
best_model_gb = random_search.best_estimator_
y_pred_gb = best_model_gb.predict(X_test)
acc_gb = accuracy_score(y_test, y_pred_gb)

# Save the best model and performance
os.makedirs('Model', exist_ok=True)
os.makedirs('Performance', exist_ok=True)

joblib.dump(best_model_gb, 'Model/best_gradient_boosting_model.pkl')

with open('Performance/best_gradient_boosting_performance.txt', 'w') as f:
    f.write(f'Best Parameters: {best_params_gb}\n')
    f.write(f'Accuracy: {acc_gb}\n') 