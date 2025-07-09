import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib

df = pd.read_csv(r"E:\GUVI\Project 3\processed_data.csv")
X = df.drop(['Attrition', 'PerformanceRating'], axis=1)
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("✅ Attrition Accuracy:", accuracy_score(y_test, y_pred))
print("✅ ROC-AUC:", roc_auc_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, r"models/best_model.pkl")
print("✅ Attrition model saved to models/best_model.pkl")
