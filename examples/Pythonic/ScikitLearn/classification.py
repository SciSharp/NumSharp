"""Breast cancer screening: train a classifier, then measure it on patients that it did not see.

This is the reference program. Classification.cs is its C# port: the same statements in the same order,
with NumSharp in the place of NumPy. Both programs print the same lines.

    python ScikitLearn/classification.py
    dotnet run -- classification
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 569 tumors with 30 numbers each. The label is 0 for a malignant tumor and 1 for a benign one.
X, y = load_breast_cancer(return_X_y=True)
print(f"tumors     {X.shape[0]} with {X.shape[1]} numbers each: {np.count_nonzero(y == 0)} malignant, {np.count_nonzero(y == 1)} benign")

# The table holds ten measurements of the cell nuclei three times: the mean, the standard error and the
# worst value. This model uses the ten means, which are the first ten columns.
means = X[:, :10]

# Keep one quarter of the patients apart. The model does not see them before the test.
X_train, X_test, y_train, y_test = train_test_split(means, y, test_size=0.25, stratify=y, random_state=0)

# An area is near 650 and a smoothness near 0.1. Bring each column to mean 0 and deviation 1.
# The scaler learns the two numbers of a column from the training patients only.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
proba = model.predict_proba(X_test_scaled)
p_benign = proba[:, 1]

print(f"accuracy   {accuracy_score(y_test, y_pred):.4f}")
print(f"roc auc    {roc_auc_score(y_test, p_benign):.4f}")
print(confusion_matrix(y_test, y_pred))

# A malignant tumor that the model calls benign is the error that costs most. The model calls a tumor
# benign when p_benign is above 0.5. Which limit finds every malignant tumor of the test, and how many
# benign tumors does that limit send to a biopsy?
malignant = y_test == 0
missed = np.count_nonzero(malignant & (y_pred == 1))
limit = p_benign[malignant].max()
referred = p_benign <= limit
print(f"missed     {missed} of {np.count_nonzero(malignant)} malignant tumors at the limit 0.5")
print(f"limit      {limit:.4f} finds all of them and refers {np.count_nonzero(referred & ~malignant)} of {np.count_nonzero(~malignant)} benign tumors")
