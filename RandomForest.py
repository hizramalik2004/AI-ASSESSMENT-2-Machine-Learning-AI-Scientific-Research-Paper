import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
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
rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, rf_pred)
precision = precision_score(y_test, rf_pred, average='macro')
recall = recall_score(y_test, rf_pred, average='macro')
f1 = f1_score(y_test, rf_pred, average='macro')
print("Random Forest – Car Evaluation Dataset")
print("--------------------------------------")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
values = [accuracy, precision, recall, f1]
plt.figure(figsize=(10, 6)) 
sns.barplot(x=metrics, y=values, palette="Greens")
for i, v in enumerate(values):
    plt.text(i, v + 0.005, f"{v:.4f}", ha='center', fontsize=12)
plt.ylim(0, 1)
plt.ylabel("Score", fontsize=12)
plt.title("Random Forest – Model Performance Metrics", fontsize=16)
plt.tight_layout()
plt.savefig("random_forest_metrics.png", dpi=300)
plt.show()
