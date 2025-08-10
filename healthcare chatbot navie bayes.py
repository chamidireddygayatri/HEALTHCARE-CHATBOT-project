import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Assuming you have a CSV file with 'text' and 'label' columns
# Replace 'path/to/your/dataset.csv' with the actual path to your dataset
dataset_path = "/content/dataset (1).csv"
df = pd.read_csv(dataset_path)

# Assuming 'text' is the input and 'label' is the target variable
X = df['question']
y = df['intent']

# Label encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Extract features using Bag of Words model
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Train Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_bow, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test_bow)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot a line graph
plt.figure(figsize=(10, 8))
sns.lineplot(data=conf_matrix.T, markers=True, dashes=False)
plt.title('True vs Predicted Labels')
plt.xlabel('Labels')
plt.ylabel('Count')
+6
plt.show()
