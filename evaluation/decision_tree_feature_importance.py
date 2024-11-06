import pandas as pd
import joblib
import os

# Load the Decision Tree model
model = joblib.load('Model/decision_tree_model.pkl')

# Load the processed dataset to get feature names
df = pd.read_csv('data/processed.csv')
feature_names = df.drop('Status', axis=1).columns

# Get feature importances
importances = model.feature_importances_
feature_importances = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

# Save feature importances
os.makedirs('performance', exist_ok=True)
with open('performance/decision_tree_feature_importance.txt', 'w') as f:
    for feature, importance in feature_importances:
        f.write(f'{feature}: {importance}\n') 