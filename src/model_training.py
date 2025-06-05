import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
data = pd.read_csv('weather_prediction_dataset.csv')

# Print all available columns to verify
print("Available columns:\n", data.columns.tolist())

# Create RainToday target variable (1 if precipitation > 0, else 0)
data['RainToday'] = data['BASEL_precipitation'].apply(lambda x: 1 if x > 0 else 0)

# Use REAL columns from the dataset (updated to match actual data)
selected_features = [
    'BASEL_temp_mean',       # Confirmed exists
    'BASEL_humidity',        # Confirmed exists
    'BASEL_pressure',        # Confirmed exists
    'DE_BILT_wind_speed',    # Proxy for wind (since BASEL lacks wind data)
    'HEATHROW_sunshine'      # Proxy for sunshine (since BASEL lacks sunshine)
]

# Verify features exist
missing = [f for f in selected_features if f not in data.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}. Check dataset.")

# Prepare features and target
X = data[selected_features]
y = data['RainToday']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model and scaler
with open('rain_prediction_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"Model trained and saved. Features used: {selected_features}")