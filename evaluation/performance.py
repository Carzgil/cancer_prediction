import matplotlib.pyplot as plt
import os

performance_dir = 'performance'

# Function to read performance metrics from a file
def read_performance(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        accuracy = None
        execution_time = None
        for line in lines:
            if 'Accuracy' in line:
                accuracy = float(line.split(': ')[1].strip())
            elif 'Execution Time' in line:
                execution_time = float(line.split(': ')[1].strip().replace(' seconds', ''))
        if accuracy is None or execution_time is None:
            raise ValueError(f"Could not find accuracy or execution time in {file_path}")
    return accuracy, execution_time

# List of models and their performance files
models = [
    ('KNN', 'knn_performance.txt'),
    ('Naive Bayes', 'naive_bayes_performance.txt'),
    ('Decision Tree', 'decision_tree_performance.txt'),
    ('Random Forest Original', 'random_forest_performance.txt'),
    ('Gradient Boosting Original', 'gradient_boosting_performance.txt'),
    ('Neural Network', 'neural_network_performance.txt')
]

# Read performance metrics
accuracies = []
execution_times = []
for model_name, file_name in models:
    file_path = os.path.join(performance_dir, file_name)
    try:
        accuracy, execution_time = read_performance(file_path)
        accuracies.append((model_name, accuracy))
        execution_times.append((model_name, execution_time))
    except ValueError as e:
        print(f"Error reading {file_name}: {e}")

# Plot execution time
plt.figure(figsize=(10, 6))
model_names, times = zip(*execution_times)
plt.bar(model_names, times, color='skyblue')
plt.title('Model Execution Time')
plt.xlabel('Model')
plt.ylabel('Time (seconds)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(performance_dir, 'model_execution_time.png'))
plt.close()

# Plot accuracy comparison
plt.figure(figsize=(10, 6))
model_names, accs = zip(*accuracies)
plt.bar(model_names, accs, color='lightgreen')
plt.title('Model Accuracy')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(performance_dir, 'model_accuracy.png'))
plt.close()

# Compare performance before and after hyperparameter tuning
tuned_models = [
    ('Random Forest Tuned', 'best_random_forest_performance.txt'),
    ('Gradient Boosting Tuned', 'best_gradient_boosting_performance.txt')
]

# Read tuned performance metrics
tuned_accuracies = []
for model_name, file_name in tuned_models:
    file_path = os.path.join(performance_dir, file_name)
    try:
        accuracy, _ = read_performance(file_path)
        tuned_accuracies.append((model_name, accuracy))
    except ValueError as e:
        print(f"Error reading {file_name}: {e}")

# Plot accuracy comparison before and after tuning
plt.figure(figsize=(10, 6))
for (model_name, original_acc), (_, tuned_acc) in zip(accuracies[3:5], tuned_accuracies):
    plt.bar(model_name, original_acc, color='orange', label='Original' if model_name == 'Random Forest Original' else "")
    plt.bar(model_name.replace('Original', 'Tuned'), tuned_acc, color='blue', label='Tuned' if model_name == 'Random Forest Original' else "")

plt.title('Accuracy Before and After Hyperparameter Tuning')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(performance_dir, 'accuracy_comparison_before_after_tuning.png'))
plt.close()