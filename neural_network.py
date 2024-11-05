import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import time
import joblib

# Load the processed dataset
df = pd.read_csv('data/processed.csv')

# Split the data into features and target
X = df.drop('Status', axis=1).values
y = df['Status'].values

# Convert data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Initialize the neural network
input_size = X_train.shape[1]
hidden_size = 10
num_classes = len(torch.unique(y_train))
model = SimpleNN(input_size, hidden_size, num_classes)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
start_time = time.time()
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Evaluate the model
with torch.no_grad():
    y_pred = model(X_test)
    _, predicted = torch.max(y_pred, 1)
    acc = accuracy_score(y_test, predicted)

end_time = time.time()
execution_time = end_time - start_time

# Save the model and performance
os.makedirs('Model', exist_ok=True)
os.makedirs('Performance', exist_ok=True)

# Save the model
torch.save(model.state_dict(), 'Model/neural_network_model.pth')

# Save the performance
with open('Performance/neural_network_performance.txt', 'w') as f:
    f.write(f'Accuracy: {acc}\n')
    f.write(f'Execution Time: {execution_time} seconds\n') 