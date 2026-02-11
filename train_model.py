import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

print("Loading dataset...")
# Load dataset
df = pd.read_csv("Indian_earthquake_data.csv")

# Fix first column name if corrupted
df.rename(columns={df.columns[0]: "Origin Time"}, inplace=True)

print("Dataset shape:", df.shape)
print("\nDataset info:")
print(df.info())

# Drop rows with missing values
df = df.dropna()
print("\nDataset shape after cleaning:", df.shape)

# Feature Selection
X = df.drop(["Magnitude", "Origin Time", "Location"], axis=1)   # Features
y = df["Magnitude"]               # Target

print("\nFeatures:", X.columns.tolist())
print("Target: Magnitude")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Feature Scaling
print("\nScaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest Model
print("\nTraining Random Forest model...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42
)

model.fit(X_train, y_train)
print("Model training completed!")

# Make Predictions
print("\nMaking predictions...")
y_pred = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n" + "="*50)
print("MODEL EVALUATION METRICS")
print("="*50)
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
print("="*50)

# Feature Importance
print("\nFeature Importance:")
importances = model.feature_importances_
features = X.columns if hasattr(X, 'columns') else ['Latitude', 'Longitude', 'Depth']
for feature, importance in zip(features, importances):
    print(f"  {feature}: {importance:.4f}")

# Save the Trained Model
print("\nSaving model and scaler...")
joblib.dump(model, "earthquake_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✓ Model saved as 'earthquake_model.pkl'")
print("✓ Scaler saved as 'scaler.pkl'")

# Test prediction on new data
print("\n" + "="*50)
print("TESTING PREDICTION")
print("="*50)
new_data = np.array([[34.5, 78.2, 10.5]])  # latitude, longitude, depth
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
print(f"Input: Latitude=34.5, Longitude=78.2, Depth=10.5")
print(f"Predicted Earthquake Magnitude: {prediction[0]:.2f}")
print("="*50)

print("\n✓ Model training and saving completed successfully!")
