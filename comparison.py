import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
column_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
df = pd.read_csv("data.csv", names=column_names)
X = df.drop("class", axis=1)
y = df["class"]
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), X.columns)]
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_metrics = {
    "Accuracy": accuracy_score(y_test, rf_pred),
    "Precision": precision_score(y_test, rf_pred, average='macro'),
    "Recall": recall_score(y_test, rf_pred, average='macro'),
    "F1-score": f1_score(y_test, rf_pred, average='macro')
}
print("Random Forest – Car Evaluation Dataset")
print("--------------------------------------")
for k, v in rf_metrics.items():
    print(f"{k}: {v:.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, rf_pred))
lr_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

lr_metrics = {
    "Accuracy": accuracy_score(y_test, lr_pred),
    "Precision": precision_score(y_test, lr_pred, average='macro'),
    "Recall": recall_score(y_test, lr_pred, average='macro'),
    "F1-score": f1_score(y_test, lr_pred, average='macro')
}

print("\nLogistic Regression – Car Evaluation Dataset")
print("-------------------------------------------")
for k, v in lr_metrics.items():
    print(f"{k}: {v:.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, lr_pred))
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
rf_values = [rf_metrics[m] for m in metrics]
lr_values = [lr_metrics[m] for m in metrics]
x = range(len(metrics))
width = 0.35
plt.figure(figsize=(10, 6))
plt.bar([p - width/2 for p in x], rf_values, width=width, label='Random Forest', color='green')
plt.bar([p + width/2 for p in x], lr_values, width=width, label='Logistic Regression', color='blue')
for i in x:
    plt.text(i - width/2, rf_values[i] + 0.01, f"{rf_values[i]:.4f}", ha='center', fontsize=11)
    plt.text(i + width/2, lr_values[i] + 0.01, f"{lr_values[i]:.4f}", ha='center', fontsize=11)

plt.xticks(x, metrics)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.legend()
plt.tight_layout()
plt.show()

