# %%
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure plots
sns.set_style('whitegrid')
plt.style.use('ggplot')
# %%
# Load the Titanic dataset
data = pd.read_csv('train.csv')

# Display first few rows
data.head()
# %%
# Check dataset structure
print(data.info())

# Check basic statistics
print(data.describe())

# Check for missing values
print("\nMissing Values:\n")
print(data.isnull().sum())
# %%
# Survival count
sns.countplot(x='Survived', data=data)
plt.title('Survival Count (0 = No, 1 = Yes)')
plt.show()

# Survival by Sex
sns.countplot(x='Survived', hue='Sex', data=data)
plt.title('Survival Count by Gender')
plt.show()

# Survival by Pclass
sns.countplot(x='Survived', hue='Pclass', data=data)
plt.title('Survival Count by Passenger Class')
plt.show()

# Age distribution
sns.histplot(data['Age'].dropna(), bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

# Fare distribution
sns.histplot(data['Fare'], bins=40, kde=True)
plt.title('Fare Distribution')
plt.show()

# Correlation heatmap
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# %%
# %%
# Copy data to avoid modifying original
titanic = data.copy()

# Handling missing values
titanic['Age'].fillna(titanic['Age'].median(), inplace=True)
titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace=True)

# Drop irrelevant columns
titanic.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Convert categorical variables to numeric
titanic = pd.get_dummies(titanic, columns=['Sex', 'Embarked'], drop_first=True)

# Check the cleaned dataset
titanic.head()

# %%
# %%
# Import necessary libraries for modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Split data into features (X) and target (y)
X = titanic.drop('Survived', axis=1)
y = titanic['Survived']

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create the logistic regression model
model = LogisticRegression(max_iter=200)

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# %%
# %%
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import seaborn as sns

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# ROC Curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (AUC = {:.2f})'.format(roc_auc_score(y_test, y_prob)))
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# %%
