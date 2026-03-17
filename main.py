import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ======================================================
# 1. Load Dataset
# ======================================================
df = pd.read_csv("diabetes.csv")
print("Ukuran Dataset:", df.shape)

# 1b. Histogram Distribusi Setiap Fitur
features = df.columns[:-1]  # semua kolom kecuali 'Outcome'

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
axes = axes.flatten()

for i, col in enumerate(features):
    axes[i].hist(df[col], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    axes[i].set_title(col, fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Nilai')
    axes[i].set_ylabel('Frekuensi')

fig.suptitle('Histogram Distribusi Setiap Fitur', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# 2. TANPA DATA CLEANING (replikasi jurnal)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 3. Normalisasi Z-Score (SEBELUM split)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 4. Split Data (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y
)


# 5. Model
models = {
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(C=1.0, kernel='linear', gamma='auto')
}

results = []

for model_name, model in models.items():

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    results.append([model_name, acc*100])

    print("\n====================================")
    print(f"Model: {model_name}")
    print("====================================")
    print(f"Accuracy  : {acc*100:.2f}%")
    print(f"Precision : {prec*100:.2f}%")
    print(f"Recall    : {rec*100:.2f}%")
    print(f"F1-Score  : {f1*100:.2f}%")
    print("Confusion Matrix:")
    print(cm)

    # Confusion Matrix Heatmap
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# 6. Grafik Perbandingan Akurasi
results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])

plt.figure(figsize=(6,5))
sns.barplot(data=results_df, x="Model", y="Accuracy")
plt.title("Perbandingan Akurasi Algoritma (80:20)")
plt.ylabel("Accuracy (%)")
plt.ylim(0,100)
plt.tight_layout()
plt.show()

# 7. Visualisasi Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

plt.figure(figsize=(18,8))
plot_tree(
    dt_model,
    feature_names=df.columns[:-1],
    class_names=["Non-Diabetes", "Diabetes"],
    filled=True
)
plt.title("Visualisasi Decision Tree")
plt.show()