import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import os
import time
import joblib

# Load the processed dataset
df = pd.read_csv('data/processed.csv')

# Split the data into features and target
X = df.drop('Status', axis=1).values 
y = df['Status'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree model
model = DecisionTreeClassifier(criterion='entropy', random_state=42)  # Using 'entropy' to mimic C4.5

# Measure the time taken to train and predict
start_time = time.time()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
end_time = time.time()
execution_time = end_time - start_time

# Evaluate the model
acc = accuracy_score(y_test, y_pred)

# Save the model and performance
os.makedirs('Model', exist_ok=True)
os.makedirs('Performance', exist_ok=True)

# Save the model
joblib.dump(model, 'Model/decision_tree_model.pkl')

# Save the performance
with open('Performance/decision_tree_performance.txt', 'w') as f:
    f.write(f'Accuracy: {acc}\n')
    f.write(f'Execution Time: {execution_time} seconds\n')
