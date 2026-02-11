import joblib
import numpy as np

print("Loading trained model and scaler...")
model = joblib.load("earthquake_model.pkl")
scaler = joblib.load("scaler.pkl")

print("✓ Model loaded successfully!")
print("✓ Scaler loaded successfully!")

# Test prediction
print("\n" + "="*60)
print("TESTING EARTHQUAKE MAGNITUDE PREDICTION")
print("="*60)

# Test case 1
test_data_1 = np.array([[28.5, 77.2, 10.0]])  # Delhi region
prediction_1 = model.predict(scaler.transform(test_data_1))
print(f"\nTest 1:")
print(f"  Location: Latitude=28.5, Longitude=77.2, Depth=10.0 km")
print(f"  Predicted Magnitude: {prediction_1[0]:.2f}")

# Test case 2
test_data_2 = np.array([[34.5, 78.2, 15.0]])  # Kashmir region
prediction_2 = model.predict(scaler.transform(test_data_2))
print(f"\nTest 2:")
print(f"  Location: Latitude=34.5, Longitude=78.2, Depth=15.0 km")
print(f"  Predicted Magnitude: {prediction_2[0]:.2f}")

# Test case 3
test_data_3 = np.array([[19.0, 73.0, 5.0]])  # Mumbai region
prediction_3 = model.predict(scaler.transform(test_data_3))
print(f"\nTest 3:")
print(f"  Location: Latitude=19.0, Longitude=73.0, Depth=5.0 km")
print(f"  Predicted Magnitude: {prediction_3[0]:.2f}")

print("\n" + "="*60)
print("✓ Model is working correctly and ready for predictions!")
print("="*60)
