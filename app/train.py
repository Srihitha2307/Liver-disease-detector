import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load Data
df = pd.read_csv('indian_liver_patient.csv')

# 2. Data Cleaning (Based on your notebook)
# Fill missing values in Albumin_and_Globulin_Ratio
df['Albumin_and_Globulin_Ratio'] = df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].median())

# 3. Encoding Gender
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# 4. Define Features (X) and Target (y)
# In this dataset, 1 = Disease, 2 = No Disease. 
# We convert it to 1 = Disease, 0 = No Disease for clearer AI logic.
X = df.drop('Dataset', axis=1)
y = df['Dataset'].map({1: 1, 2: 0}) 

# 5. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# 7. Save all assets
joblib.dump(model, 'liver_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'encoder.pkl')
joblib.dump(X.columns.tolist(), 'feature_names.pkl')

print("All files saved successfully! Now you can run the app.")