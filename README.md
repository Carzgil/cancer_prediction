# Cancer Prediction

This project involves building and evaluating various machine learning models to predict cancer outcomes based on a dataset. The models used include K-Nearest Neighbors (KNN), Naive Bayes, Decision Tree, Random Forest, Gradient Boosting, and a Neural Network.

## Preprocessing

1. **Data Loading**: The dataset is loaded and inspected for missing values and inconsistencies.
2. **Imputation**: Missing numerical values are filled with the mean, and categorical values are filled with the most frequent value.
3. **Encoding**: Categorical variables, including the target variable `Status`, which is if the patient is alive or dead, are encoded into numerical values.
4. **Outlier Detection**: Outliers are detected and removed using the z-score method.
5. **Standardization**: Numerical features are standardized to have a mean of 0 and a standard deviation of 1.

## Models

1. **K-Nearest Neighbors (KNN)**: A simple model that classifies based on the majority class of the nearest neighbors. This was implemnted by scratch using the Euclidean distance.
2. **Naive Bayes**: A probabilistic model based on Bayes' theorem with strong independence assumptions. Bayes theorem is used to calculate the probability of the target variable based on the features.
3. **Decision Tree**: A model that splits data into branches to make predictions. The Gini impurity, which measures the impurity of the node, is used to calculate the best split.
4. **Random Forest**: An ensemble of decision trees, using the 'entropy' criterion for splitting. Entropy is used to calculate the information gain of the split, to mimic the C4.5 algorithm.
5. **Gradient Boosting**: An ensemble model that builds trees sequentially to improve predictions.
6. **Neural Network**: A simple feedforward neural network implemented using PyTorch.

## Hyperparameter Tuning

- **Random Forest and Gradient Boosting**: Hyperparameters were tuned using `RandomizedSearchCV` to find the best model configurations. Although, both of the models did not see much of a performance increase after tuning. This is either due to the models already being very accurate or the search space being too large for the models to converge. 

## Performance Visualizations

### Model Execution Time

![Model Execution Time](performance/model_execution_time.png)

This bar chart shows the execution time for each model, highlighting the computational cost associated with training and predicting with each algorithm. As expected, the KNN model, which was implemented by scratch, took the longest to execute.

### Model Accuracy

![Model Accuracy](performance/model_accuracy.png)

This bar chart compares the accuracy of each model, providing insight into their predictive performance on the dataset. All of the models performed very well, with the exception of the KNN model, which was implemented by scratch.

### Accuracy Before and After Hyperparameter Tuning

![Accuracy Before and After Hyperparameter Tuning](performance/accuracy_comparison_before_after_tuning.png)

This chart compares the accuracy of the Random Forest and Gradient Boosting models before and after hyperparameter tuning, illustrating the impact of tuning on model performance.

### Conclusion

Through this project, we successfully built and evaluated multiple machine learning models to predict cancer outcomes, achieving an accuracy of up to 90%. This demonstrates the potential of these models in predicting whether a patient would die from breast cancer.

For feature importance, the most prominent features varied across models:

- **Random Forest**: Feature importance can be extracted using the `feature_importances_` attribute. Typically, features like tumor size and stage are significant.
- **Gradient Boosting**: Similar to Random Forest, feature importance can be assessed, often highlighting tumor characteristics and hormone receptor status.
- **Decision Tree**: The tree structure itself reveals which features are most important for splits, often aligning with clinical indicators.

These insights can guide further research and model refinement, emphasizing the importance of data-driven approaches in healthcare.
