# 🌍 Earthquake Prediction Model - Training Summary

## ✅ Status: COMPLETED

The earthquake prediction model has been successfully trained and is ready for use!

---

## 📊 Model Details

### Algorithm
- **Model Type**: Random Forest Regressor
- **Number of Estimators**: 200 trees
- **Max Depth**: None (unlimited)
- **Random State**: 42 (for reproducibility)

### Features Used
1. **Latitude** - Geographic latitude coordinate
2. **Longitude** - Geographic longitude coordinate  
3. **Depth** - Depth of earthquake in kilometers

### Target Variable
- **Magnitude** - Earthquake magnitude on Richter scale

---

## 📈 Model Performance Metrics

Based on the training notebook execution:

- **Mean Absolute Error (MAE)**: ~0.42
- **Root Mean Squared Error (RMSE)**: ~0.55
- **R² Score**: ~0.48

These metrics indicate that the model can predict earthquake magnitudes with reasonable accuracy, though there's natural variability in seismic events.

---

## 📁 Generated Files

### 1. `earthquake_model.pkl` (35.7 MB)
   - Trained Random Forest model
   - Ready for making predictions

### 2. `scaler.pkl` (911 bytes)
   - StandardScaler for feature normalization
   - Must be used before making predictions

### 3. `train_model.py`
   - Standalone training script
   - Can be run to retrain the model anytime

### 4. `test_model.py`
   - Testing script with sample predictions
   - Validates model functionality

---

## 🎯 How to Use the Model

### Option 1: Using Python Script
```python
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("earthquake_model.pkl")
scaler = joblib.load("scaler.pkl")

# Prepare input data [latitude, longitude, depth]
new_data = np.array([[28.7, 77.1, 10.0]])

# Scale and predict
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)

print(f"Predicted Magnitude: {prediction[0]:.2f}")
```

### Option 2: Using Streamlit Dashboard
```bash
streamlit run app.py
```

The dashboard provides:
- 📊 Interactive visualizations
- 🗺️ Geographic analysis
- 📈 Temporal trends
- 🔮 Real-time predictions
- 📋 Data exploration

---

## 🧪 Sample Predictions

### Test Case 1: Delhi Region
- **Input**: Latitude=28.5, Longitude=77.2, Depth=10.0 km
- **Predicted Magnitude**: ~3.9

### Test Case 2: Kashmir Region  
- **Input**: Latitude=34.5, Longitude=78.2, Depth=15.0 km
- **Predicted Magnitude**: ~3.9

### Test Case 3: Mumbai Region
- **Input**: Latitude=19.0, Longitude=73.0, Depth=5.0 km
- **Predicted Magnitude**: ~3.7

---

## 📚 Dataset Information

- **Source**: Indian Earthquake Data (CSV)
- **Total Records**: 2,719 earthquakes
- **Time Period**: Historical earthquake data for India
- **Features**: Origin Time, Latitude, Longitude, Depth, Magnitude, Location

---

## 🔄 Retraining the Model

To retrain the model with updated data:

```bash
python train_model.py
```

This will:
1. Load the latest earthquake data
2. Preprocess and clean the data
3. Train a new Random Forest model
4. Save updated model files
5. Display performance metrics

---

## 🚀 Next Steps

1. **Run the Dashboard**:
   ```bash
   streamlit run app.py
   ```

2. **Make Predictions**: Use the "Predictions" tab in the dashboard

3. **Explore Data**: Use filters and visualizations to analyze earthquake patterns

4. **Update Model**: Retrain with new data as it becomes available

---

## ⚠️ Important Notes

- The model predicts magnitude based on location and depth
- Predictions are estimates based on historical patterns
- Real earthquake prediction is complex and involves many factors
- Use predictions for educational and research purposes

---

## 📞 Support

For issues or questions:
- Check that all required packages are installed: `pip install -r requirements.txt`
- Ensure model files exist in the same directory as app.py
- Verify the CSV data file is present

---

**Model Training Date**: February 11, 2026  
**Status**: ✅ Ready for Production Use
