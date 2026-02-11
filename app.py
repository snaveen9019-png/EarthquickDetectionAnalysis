import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🌍 Earthquake Prediction Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #1e2130;
        border-radius: 5px;
        padding: 10px 20px;
    }
    h1 {
        color: #00d4ff;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #00d4ff 0%, #0099cc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    h2, h3 {
        color: #00d4ff;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Indian_earthquake_data.csv')
        # Fix column name
        df.rename(columns={df.columns[0]: 'Origin Time'}, inplace=True)
        # Convert time column
        df['Origin Time'] = pd.to_datetime(
            df['Origin Time'].str.replace(' IST', '', regex=False),
            errors='coerce'
        )
        df = df.sort_values('Origin Time')
        df['Year'] = df['Origin Time'].dt.year
        df['Month'] = df['Origin Time'].dt.month
        df['Day'] = df['Origin Time'].dt.day
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('earthquake_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        return None, None

# Main app
def main():
    # Title
    st.markdown("<h1>🌍 Earthquake Prediction Dashboard - India</h1>", unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    model, scaler = load_model()
    
    if df is None:
        st.error("Failed to load data. Please check if 'Indian_earthquake_data.csv' exists.")
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/earthquake.png", width=100)
        st.title("⚙️ Dashboard Controls")
        
        # Date range filter
        st.subheader("📅 Date Range")
        min_date = df['Origin Time'].min().date()
        max_date = df['Origin Time'].max().date()
        
        date_range = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Magnitude filter
        st.subheader("📊 Magnitude Range")
        mag_range = st.slider(
            "Select Magnitude Range",
            float(df['Magnitude'].min()),
            float(df['Magnitude'].max()),
            (float(df['Magnitude'].min()), float(df['Magnitude'].max()))
        )
        
        # Depth filter
        st.subheader("🔍 Depth Range (km)")
        depth_range = st.slider(
            "Select Depth Range",
            float(df['Depth'].min()),
            float(df['Depth'].max()),
            (float(df['Depth'].min()), float(df['Depth'].max()))
        )
    
    # Filter data
    if len(date_range) == 2:
        filtered_df = df[
            (df['Origin Time'].dt.date >= date_range[0]) &
            (df['Origin Time'].dt.date <= date_range[1]) &
            (df['Magnitude'] >= mag_range[0]) &
            (df['Magnitude'] <= mag_range[1]) &
            (df['Depth'] >= depth_range[0]) &
            (df['Depth'] <= depth_range[1])
        ]
    else:
        filtered_df = df
    
    # Key Metrics
    st.markdown("## 📈 Key Statistics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Earthquakes", len(filtered_df))
    with col2:
        st.metric("Average Magnitude", f"{filtered_df['Magnitude'].mean():.2f}")
    with col3:
        st.metric("Max Magnitude", f"{filtered_df['Magnitude'].max():.2f}")
    with col4:
        st.metric("Average Depth", f"{filtered_df['Depth'].mean():.1f} km")
    with col5:
        st.metric("Max Depth", f"{filtered_df['Depth'].max():.1f} km")
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🗺️ Geographic Analysis", 
        "📉 Trends", 
        "🔮 Predictions",
        "📋 Data Explorer"
    ])
    
    with tab1:
        st.markdown("### 🌟 Earthquake Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Magnitude distribution
            fig = px.histogram(
                filtered_df, 
                x='Magnitude',
                nbins=30,
                title='Magnitude Distribution',
                color_discrete_sequence=['#00d4ff']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Depth distribution
            fig = px.histogram(
                filtered_df, 
                x='Depth',
                nbins=30,
                title='Depth Distribution',
                color_discrete_sequence=['#ff6b6b']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Magnitude vs Depth scatter
        st.markdown("### 🎯 Magnitude vs Depth Relationship")
        fig = px.scatter(
            filtered_df,
            x='Depth',
            y='Magnitude',
            color='Magnitude',
            size='Magnitude',
            hover_data=['Location', 'Origin Time'],
            title='Earthquake Magnitude vs Depth',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 🗺️ Geographic Distribution")
        
        # Interactive map
        fig = px.scatter_mapbox(
            filtered_df,
            lat='Latitude',
            lon='Longitude',
            color='Magnitude',
            size='Magnitude',
            hover_name='Location',
            hover_data={'Latitude': True, 'Longitude': True, 'Depth': True, 'Magnitude': True},
            color_continuous_scale='Reds',
            zoom=4,
            title='Earthquake Locations in India'
        )
        fig.update_layout(
            mapbox_style="carto-darkmatter",
            height=600,
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap of earthquake density
        col1, col2 = st.columns(2)
        
        with col1:
            # Latitude distribution
            fig = px.histogram(
                filtered_df,
                x='Latitude',
                nbins=50,
                title='Latitude Distribution',
                color_discrete_sequence=['#4ecdc4']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Longitude distribution
            fig = px.histogram(
                filtered_df,
                x='Longitude',
                nbins=50,
                title='Longitude Distribution',
                color_discrete_sequence=['#95e1d3']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 📈 Temporal Trends")
        
        # Time series of magnitude
        fig = px.line(
            filtered_df,
            x='Origin Time',
            y='Magnitude',
            title='Earthquake Magnitude Over Time',
            color_discrete_sequence=['#00d4ff']
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Yearly trends
        col1, col2 = st.columns(2)
        
        with col1:
            yearly_count = filtered_df.groupby('Year').size().reset_index(name='Count')
            fig = px.bar(
                yearly_count,
                x='Year',
                y='Count',
                title='Earthquakes per Year',
                color='Count',
                color_continuous_scale='Blues'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            monthly_avg = filtered_df.groupby('Month')['Magnitude'].mean().reset_index()
            fig = px.line(
                monthly_avg,
                x='Month',
                y='Magnitude',
                title='Average Magnitude by Month',
                markers=True,
                color_discrete_sequence=['#ff6b6b']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### 🔮 Earthquake Magnitude Prediction")
        
        if model is not None and scaler is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### 📍 Enter Location Details")
                
                latitude = st.number_input(
                    "Latitude",
                    min_value=0.0,
                    max_value=40.0,
                    value=28.7,
                    step=0.1,
                    help="Enter latitude (0-40°N)"
                )
                
                longitude = st.number_input(
                    "Longitude",
                    min_value=60.0,
                    max_value=100.0,
                    value=77.1,
                    step=0.1,
                    help="Enter longitude (60-100°E)"
                )
                
                depth = st.number_input(
                    "Depth (km)",
                    min_value=0.0,
                    max_value=500.0,
                    value=10.0,
                    step=1.0,
                    help="Enter depth in kilometers"
                )
                
                if st.button("🎯 Predict Magnitude", type="primary"):
                    # Make prediction
                    new_data = np.array([[latitude, longitude, depth]])
                    new_data_scaled = scaler.transform(new_data)
                    prediction = model.predict(new_data_scaled)[0]
                    
                    st.markdown(
                        f'<div class="prediction-box">Predicted Magnitude: {prediction:.2f}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Risk assessment
                    if prediction < 3.0:
                        risk = "🟢 Low Risk"
                        color = "green"
                    elif prediction < 5.0:
                        risk = "🟡 Moderate Risk"
                        color = "orange"
                    else:
                        risk = "🔴 High Risk"
                        color = "red"
                    
                    st.markdown(f"### Risk Level: :{color}[{risk}]")
            
            with col2:
                st.markdown("#### 📊 Model Performance")
                
                # Display model metrics
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler
                
                X = df[['Latitude', 'Longitude', 'Depth']].dropna()
                y = df.loc[X.index, 'Magnitude']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                scaler_temp = StandardScaler()
                X_test_scaled = scaler_temp.fit_transform(X_train)
                X_test_scaled = scaler_temp.transform(X_test)
                
                y_pred = model.predict(X_test_scaled)
                
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("MAE", f"{mae:.3f}")
                with metric_col2:
                    st.metric("RMSE", f"{rmse:.3f}")
                with metric_col3:
                    st.metric("R² Score", f"{r2:.3f}")
                
                # Feature importance
                st.markdown("#### 🎯 Feature Importance")
                importances = model.feature_importances_
                features = ['Latitude', 'Longitude', 'Depth']
                
                fig = go.Figure(go.Bar(
                    x=importances,
                    y=features,
                    orientation='h',
                    marker=dict(color=['#00d4ff', '#ff6b6b', '#4ecdc4'])
                ))
                fig.update_layout(
                    title='Feature Importance',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Model not found. Please train the model first using the Jupyter notebook.")
    
    with tab5:
        st.markdown("### 📋 Data Explorer")
        
        # Search functionality
        search_term = st.text_input("🔍 Search Location", "")
        
        if search_term:
            display_df = filtered_df[
                filtered_df['Location'].str.contains(search_term, case=False, na=False)
            ]
        else:
            display_df = filtered_df
        
        # Display dataframe
        st.dataframe(
            display_df[['Origin Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude', 'Location']],
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name=f"earthquake_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Statistics
        st.markdown("### 📊 Statistical Summary")
        st.dataframe(
            display_df[['Latitude', 'Longitude', 'Depth', 'Magnitude']].describe(),
            use_container_width=True
        )

if __name__ == "__main__":
    main()
