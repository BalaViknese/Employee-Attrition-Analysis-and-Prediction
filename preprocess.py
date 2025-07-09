import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

os.makedirs('models', exist_ok=True)

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df.drop(['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours'], axis=1, inplace=True)
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    return df

def encode_categorical(df):
    cat_cols = df.select_dtypes(include='object').columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    joblib.dump(encoders, 'models/encoders.pkl')
    return df

def scale_and_save_features(df):
    scaler = StandardScaler()
    X = df.drop(['Attrition', 'PerformanceRating'], axis=1)
    X_scaled = scaler.fit_transform(X)
    
    joblib.dump(scaler, r"models/attrition_scaler.pkl")
    joblib.dump(list(X.columns), r"models/attrition_features.pkl")
    
    df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    df_scaled['Attrition'] = df['Attrition'].values
    df_scaled['PerformanceRating'] = df['PerformanceRating'].values
    return df_scaled

if __name__ == "__main__":
    df = load_and_clean_data(r'E:\GUVI\Project 3\Employee-Attrition - Employee-Attrition.csv')
    df = encode_categorical(df)
    df = scale_and_save_features(df)
    df.to_csv(r'E:\GUVI\Project 3\processed_data.csv', index=False)
    print("✅ Data preprocessing complete and saved.")
