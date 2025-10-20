# TRAINING LOGIC

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample dataset
data = pd.DataFrame({
    "income": [30, 50, 80, 20, 45, 70, 100, 60],
    "credit_score": [600, 650, 720, 580, 640, 700, 750, 680],
    "age": [25, 30, 40, 22, 28, 35, 45, 32],
    "loan_approved": [0, 1, 1, 0, 1, 1, 1, 1]
})

X = data[["income", "credit_score", "age"]]
y = data["loan_approved"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = LogisticRegression().fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

# Save the model
import joblib
joblib.dump(model, "loan_model.pkl")
