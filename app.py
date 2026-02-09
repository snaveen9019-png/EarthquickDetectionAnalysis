import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="🌍 Earthquake Prediction Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown("""
<style>
.main { background-color: #0e1117; }
.stMetric {
    background-color: #1e2130;
    padding: 15px;
    border-radius: 10px;
}
h1 {
    text-align: center;
    background: linear-gradient(90deg, #00d4ff, #0099cc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
h2, h3 { color: #00d4ff; }
.prediction-box {
    background: linear-gradient(135deg, #667eea, #764ba2);
    padding: 20px;
    border-radius: 10px;
    color: white;
    font-size: 24px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load data
# --------------------------------------------------
@st.cache_data
def load_data():
    if not os.path.exists("Indian_earthquake_data.csv"):
        return None

    df = pd.read_csv("Indian_earthquake_data.csv")
    df.rename(columns={df.columns[0]: "Origin Time"}, inplace=True)

    df["Origin Time"] = pd.to_datetime(
        df["Origin Time"].astype(str).str.replace(" IST", ""),
        errors="coerce"
    )

    df = df.dropna(subset=["Latitude", "Longitude", "Depth", "Magnitude"])
    df = df.sort_values("Origin Time")

    df["Year"] = df["Origin Time"].dt.year
    df["Month"] = df["Origin Time"].dt.month

    return df

# --------------------------------------------------
# Load model
# --------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists("earthquake_model.pkl") or not os.path.exists("scaler.pkl"):
        return None, None

    model = joblib.load("earthquake_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

# --------------------------------------------------
# Main App
# --------------------------------------------------
def main():

    st.markdown("<h1>🌍 Earthquake Prediction Dashboard – India</h1>", unsafe_allow_html=True)

    df = load_data()
    model, scaler = load_model()

    if df is None:
        st.error("❌ Dataset not found (Indian_earthquake_data.csv)")
        st.stop()

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.title("⚙️ Filters")

        min_date = df["Origin Time"].min().date()
        max_date = df["Origin Time"].max().date()

        date_range = st.date_input(
            "📅 Date Range",
            (min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        mag_range = st.slider(
            "📊 Magnitude",
            float(df["Magnitude"].min()),
            float(df["Magnitude"].max()),
            (float(df["Magnitude"].min()), float(df["Magnitude"].max()))
        )

        depth_range = st.slider(
            "🔍 Depth (km)",
            float(df["Depth"].min()),
            float(df["Depth"].max()),
            (float(df["Depth"].min()), float(df["Depth"].max()))
        )

    # ---------------- Filter data ----------------
    filtered_df = df[
        (df["Origin Time"].dt.date >= date_range[0]) &
        (df["Origin Time"].dt.date <= date_range[1]) &
        (df["Magnitude"].between(*mag_range)) &
        (df["Depth"].between(*depth_range))
    ]

    # ---------------- Metrics ----------------
    st.markdown("## 📈 Key Statistics")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Earthquakes", len(filtered_df))
    c2.metric("Avg Magnitude", f"{filtered_df['Magnitude'].mean():.2f}")
    c3.metric("Max Magnitude", f"{filtered_df['Magnitude'].max():.2f}")
    c4.metric("Avg Depth", f"{filtered_df['Depth'].mean():.1f} km")

    # ---------------- Tabs ----------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🗺️ Map",
        "📈 Trends",
        "🔮 Prediction",
        "📋 Data"
    ])

    # ================= OVERVIEW =================
    with tab1:
        fig = px.histogram(filtered_df, x="Magnitude", nbins=30, title="Magnitude Distribution")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(
            filtered_df,
            x="Depth",
            y="Magnitude",
            color="Magnitude",
            hover_data=["Location"],
            title="Magnitude vs Depth"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ================= MAP =================
    with tab2:
        fig = px.scatter_geo(
            filtered_df,
            lat="Latitude",
            lon="Longitude",
            color="Magnitude",
            size="Magnitude",
            projection="natural earth",
            title="Earthquake Locations"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ================= TRENDS =================
    with tab3:
        yearly = filtered_df.groupby("Year").size().reset_index(name="Count")
        fig = px.bar(yearly, x="Year", y="Count", title="Earthquakes per Year")
        st.plotly_chart(fig, use_container_width=True)

    # ================= PREDICTION =================
    with tab4:
        if model is None or scaler is None:
            st.warning("⚠️ Model files not found")
        else:
            lat = st.number_input("Latitude", 0.0, 40.0, 28.7)
            lon = st.number_input("Longitude", 60.0, 100.0, 77.1)
            depth = st.number_input("Depth (km)", 0.0, 500.0, 10.0)

            if st.button("🎯 Predict"):
                X = np.array([[lat, lon, depth]])
                X_scaled = scaler.transform(X)
                pred = model.predict(X_scaled)[0]

                st.markdown(
                    f"<div class='prediction-box'>Predicted Magnitude: {pred:.2f}</div>",
                    unsafe_allow_html=True
                )

            # Model metrics
            X_all = df[["Latitude", "Longitude", "Depth"]]
            y_all = df["Magnitude"]

            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=0.2, random_state=42
            )

            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)

            st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.3f}")
            st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
            st.metric("R²", f"{r2_score(y_test, y_pred):.3f}")

            if hasattr(model, "feature_importances_"):
                fig = go.Figure(go.Bar(
                    x=model.feature_importances_,
                    y=["Latitude", "Longitude", "Depth"],
                    orientation="h"
                ))
                st.plotly_chart(fig, use_container_width=True)

    # ================= DATA =================
    with tab5:
        st.dataframe(filtered_df, use_container_width=True)

        csv = filtered_df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            f"earthquake_data_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )


if __name__ == "__main__":
    main()
