# Disease Prediction - Diabetes Detection using ML

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv',
                 names=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'])

print("🔍 First 5 rows of the dataset:")
print(df.head())

# Step 2: Check for missing values or anomalies
print("\n🧼 Dataset Info:")
print(df.info())

# Step 3: Handle missing or zero values
# Many features shouldn't be zero (like Glucose, BMI)
cols_to_clean = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_to_clean] = df[cols_to_clean].replace(0, np.nan)
df.fillna(df.median(), inplace=True)

# Step 4: Split data into features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 8: Make predictions
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
print("\n📊 Accuracy Score:", accuracy_score(y_test, y_pred))
print("\n📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# Optional: Predict for a new user
sample_input = np.array([[2, 130, 70, 0, 0, 28.0, 0.5, 40]])  # Replace with real input
sample_input_scaled = scaler.transform(sample_input)
sample_prediction = model.predict(sample_input_scaled)
print("\n🧪 Sample Prediction: Diabetes" if sample_prediction[0] == 1 else "🧪 Sample Prediction: No Diabetes")
