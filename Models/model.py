# Customer churn model example
# This script loads the dataset, prepares the data,
# trains a logistic regression model, and prints the accuracy.

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Load the dataset from the data folder
data = pd.read_csv("data/churn_dataset.csv")

# Separate features and target
X = data.drop(columns=["customer_id", "churn"])
y = data["churn"]

# Select categorical and numeric columns
categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
numeric_columns = X.select_dtypes(exclude=["object"]).columns.tolist()

# One-hot encode categorical columns and keep numeric columns unchanged
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ("num", "passthrough", numeric_columns),
    ]
)

# Build a simple machine learning pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)

print("Model: Logistic Regression")
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification report:")
print(classification_report(y_test, predictions))
