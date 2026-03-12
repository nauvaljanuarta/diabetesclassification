import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)

# =========================
# 1. Load Data
# =========================
df = pd.read_csv("diabetes.csv")

# =========================
# 2. Data Cleaning
# =========================
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in zero_columns:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# =========================
# 3. Feature & Target
# =========================
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# =========================
# 4. Standardisasi
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 5. Split Data
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 6. SVM
# =========================
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

grid = GridSearchCV(SVC(class_weight='balanced'), param_grid, cv=5, scoring='f1')
grid.fit(X_train, y_train)

model = grid.best_estimator_
y_pred = model.predict(X_test)

# =========================
# 7. Evaluasi
# =========================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_prob = SVC(**grid.best_params_, probability=True)
model_prob.fit(X_train, y_train)
y_prob = model_prob.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("Best Parameters:", grid.best_params_)
print("\nAccuracy  :", round(accuracy, 4))
print("Precision :", round(precision, 4))
print("Recall    :", round(recall, 4))
print("F1-Score  :", round(f1, 4))
print("AUC-ROC   :", round(auc, 4))
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
