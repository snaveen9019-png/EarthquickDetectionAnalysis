# 🌍 Earthquake Prediction Dashboard

A comprehensive interactive dashboard for visualizing and predicting earthquake data in India using Streamlit and Machine Learning.

## Features

### 📊 Overview Tab
- **Key Statistics**: Total earthquakes, average magnitude, max magnitude, average depth, max depth
- **Magnitude Distribution**: Histogram showing the distribution of earthquake magnitudes
- **Depth Distribution**: Histogram showing the distribution of earthquake depths
- **Magnitude vs Depth**: Interactive scatter plot showing the relationship between magnitude and depth

### 🗺️ Geographic Analysis Tab
- **Interactive Map**: Visualize earthquake locations on an interactive map with color-coded magnitudes
- **Latitude/Longitude Distribution**: Histograms showing geographic distribution

### 📈 Trends Tab
- **Time Series Analysis**: Line chart showing earthquake magnitude over time
- **Yearly Trends**: Bar chart showing the number of earthquakes per year
- **Monthly Patterns**: Line chart showing average magnitude by month

### 🔮 Predictions Tab
- **ML Model Predictions**: Predict earthquake magnitude based on latitude, longitude, and depth
- **Risk Assessment**: Automatic risk level classification (Low/Moderate/High)
- **Model Performance Metrics**: MAE, RMSE, and R² score
- **Feature Importance**: Visualization of which features contribute most to predictions

### 📋 Data Explorer Tab
- **Search Functionality**: Search earthquakes by location
- **Interactive Data Table**: Browse and filter earthquake data
- **Download Data**: Export filtered data as CSV
- **Statistical Summary**: Descriptive statistics for all numerical columns

## Installation

1. **Install Python** (3.8 or higher recommended)

2. **Install required packages**:
```bash
pip install -r requirements.txt
```

## Running the Dashboard

1. **Make sure you have the following files in the same directory**:
   - `app.py` (the dashboard code)
   - `Indian_earthquake_data.csv` (the earthquake dataset)
   - `earthquake_model.pkl` (trained ML model)
   - `scaler.pkl` (data scaler)

2. **Run the Streamlit app**:
```bash
streamlit run app.py
```

3. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

## Training the Model

If you don't have the model files (`earthquake_model.pkl` and `scaler.pkl`), you need to train the model first using the Jupyter notebook:

1. Open `earthquick.ipynb` in Jupyter Notebook or Google Colab
2. Run all cells to train the model
3. The model files will be saved automatically

## Dashboard Controls

### Sidebar Filters
- **Date Range**: Filter earthquakes by date
- **Magnitude Range**: Filter by earthquake magnitude
- **Depth Range**: Filter by earthquake depth (in km)

### Interactive Features
- Hover over charts to see detailed information
- Click and drag on maps to zoom
- Use the search box in Data Explorer to find specific locations
- Download filtered data as CSV

## Model Information

- **Algorithm**: Random Forest Regressor
- **Features**: Latitude, Longitude, Depth
- **Target**: Earthquake Magnitude
- **Training Data**: Historical earthquake data from India

## Risk Classification

- 🟢 **Low Risk**: Magnitude < 3.0
- 🟡 **Moderate Risk**: Magnitude 3.0 - 5.0
- 🔴 **High Risk**: Magnitude > 5.0

## Technologies Used

- **Streamlit**: Web dashboard framework
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning
- **NumPy**: Numerical computing

## Troubleshooting

### Dashboard won't start
- Make sure all required packages are installed
- Check that you're in the correct directory
- Verify Python version (3.8+)

### Model predictions not working
- Ensure `earthquake_model.pkl` and `scaler.pkl` exist
- Train the model using the Jupyter notebook if files are missing

### Data not loading
- Verify `Indian_earthquake_data.csv` is in the same directory
- Check file permissions

## Future Enhancements

- [ ] Real-time earthquake data integration
- [ ] Advanced ML models (Neural Networks, Gradient Boosting)
- [ ] Earthquake early warning system
- [ ] Multi-region support
- [ ] Historical comparison tools
- [ ] Export reports as PDF

## License

This project is for educational purposes.

## Contact

For questions or suggestions, please open an issue on the repository.

---

**Note**: This dashboard is for educational and research purposes only. Always refer to official earthquake monitoring agencies for real-time alerts and safety information.
