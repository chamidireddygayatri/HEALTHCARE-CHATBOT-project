import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

# Assuming you have a CSV file with 'text' and 'label' columns
# Replace 'path/to/your/dataset.csv' with the actual path to your dataset
dataset_path = "/content/dataset (1).csv"
df = pd.read_csv(dataset_path)

# Assuming 'text' is the input and 'label' is the target variable
X = df['question']
y = df['intent']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to feature vectors using CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Vary the max_depth parameter and record accuracy
max_depth_values = np.arange(1, 21)
train_accuracy = []
test_accuracy = []

for depth in max_depth_values:
    # Build the Decision Tree model with varying max_depth
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train_vec, y_train)

    # Predict on training and test sets
    y_train_pred = model.predict(X_train_vec)
    y_test_pred = model.predict(X_test_vec)

    # Calculate accuracy
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    train_accuracy.append(train_acc)
    test_accuracy.append(test_acc)
# Plot the accuracy graph
plt.plot(max_depth_values, train_accuracy, label='Training Accuracy')
plt.plot(max_depth_values, test_accuracy, label='Test Accuracy')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree Accuracy vs Max Depth')
plt.legend()
plt.show()
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

