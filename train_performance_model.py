import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

df = pd.read_csv(r"E:\GUVI\Project 3\processed_data.csv")
features = ['YearsAtCompany', 'JobLevel', 'JobInvolvement', 'MonthlyIncome']
X = df[features]
y = df['PerformanceRating'].astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("✅ Performance Accuracy:", accuracy_score(y_test, y_pred))
print("📋 Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, r"models/performance_model.pkl")
joblib.dump(scaler, r"models/performance_scaler.pkl")
joblib.dump(features, r"models/performance_features.pkl")
print("✅ Performance model, scaler, and features saved.")
