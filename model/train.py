import joblib
import json
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X, y = iris.data, iris.target
class_names = list(iris.target_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=class_names)

print("="*50)
print(f"Model Accuracy: {accuracy *100:.2f}%")
print("="*50)
print("\nClassification Report:")
print(report)

joblib.dump(model, '../model/iris_model.pkl')

metadata = {
    "model_type": "RandomForestClassifier",
    "accuracy": round(accuracy, 4),
    "features": iris.feature_names,
    "classes": class_names,
    "n_estimators": 100,
    "train_samples": len(X_train),
    "test_samples": len(X_test)
}

with open("../model/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Model saved -> model/iris_model.pkl")
print("Metadata saved -> model/model_metadata.json")