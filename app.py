"""
AI-Based Natural Disaster Prediction Web App
Streamlit application for real-time flood prediction in KP, Pakistan

Features:
- Real-time weather data integration
- Flood risk prediction using trained ML models
- Interactive maps and visualizations
- Historical data analysis
- Alert system for high-risk predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
import os

# Try to import plotly, provide fallback
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Flood Prediction System - KP Pakistan",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths
PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = RESULTS_DIR

# Location configurations
LOCATIONS = {
    "swat": {
        "name": "Swat District, KP",
        "latitude": 34.8091,
        "longitude": 72.3617,
        "elevation": 980,
        "location_id": 0
    },
    "upper_dir": {
        "name": "Upper Dir District, KP", 
        "latitude": 35.3350,
        "longitude": 71.8760,
        "elevation": 1420,
        "location_id": 1
    }
}

# OpenWeatherMap API
try:
    OPENWEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "demo")
except:
    OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "demo")


@st.cache_resource
def load_model():
    """Load the trained flood prediction model"""
    model_paths = [
        MODELS_DIR / "best_flood_model.pkl",
        MODELS_DIR / "random_forest_model.pkl",
        MODELS_DIR / "logistic_regression_model.pkl"
    ]
    
    for path in model_paths:
        if path.exists():
            try:
                with open(path, 'rb') as f:
                    model_data = pickle.load(f)
                return model_data
            except Exception as e:
                st.warning(f"Could not load {path.name}: {e}")
    
    return None


@st.cache_data(ttl=1800)
def fetch_weather_data(lat, lon, api_key):
    """Fetch current weather from OpenWeatherMap API"""
    if api_key == "demo":
        # Return demo data for testing
        return generate_demo_weather()
    
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": api_key,
            "units": "metric"
        }
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return {
                "tavg": data["main"]["temp"],
                "tmin": data["main"]["temp_min"],
                "tmax": data["main"]["temp_max"],
                "humidity": data["main"]["humidity"],
                "pres": data["main"]["pressure"],
                "wspd": data["wind"]["speed"] * 3.6,  # m/s to km/h
                "prcp": data.get("rain", {}).get("1h", 0),
                "description": data["weather"][0]["description"],
                "icon": data["weather"][0]["icon"]
            }
    except Exception as e:
        st.warning(f"API error: {e}. Using demo data.")
    
    return generate_demo_weather()


def generate_demo_weather():
    """Generate realistic demo weather data"""
    month = datetime.now().month
    # Seasonal variations for KP Pakistan
    if month in [6, 7, 8]:  # Monsoon
        temp_base, prcp_base = 28, 25
    elif month in [12, 1, 2]:  # Winter
        temp_base, prcp_base = 5, 5
    else:
        temp_base, prcp_base = 18, 10
    
    return {
        "tavg": temp_base + np.random.uniform(-3, 3),
        "tmin": temp_base - 5 + np.random.uniform(-2, 2),
        "tmax": temp_base + 5 + np.random.uniform(-2, 2),
        "humidity": 60 + np.random.uniform(-20, 30),
        "pres": 1010 + np.random.uniform(-15, 15),
        "wspd": 10 + np.random.uniform(-5, 15),
        "prcp": prcp_base + np.random.uniform(0, 20),
        "description": "Demo mode - scattered clouds",
        "icon": "03d"
    }


def prepare_features(weather_data, location_id):
    """Prepare features for model prediction"""
    now = datetime.now()
    
    features = {
        'tavg': weather_data.get('tavg', 20),
        'tmin': weather_data.get('tmin', 15),
        'tmax': weather_data.get('tmax', 25),
        'prcp': weather_data.get('prcp', 0),
        'wspd': weather_data.get('wspd', 10),
        'wpgt': weather_data.get('wspd', 10) * 1.5,  # Estimated gust
        'pres': weather_data.get('pres', 1010),
        'humidity': weather_data.get('humidity', 60),
        'solar_radiation': 15 + np.random.uniform(-5, 10),
        'month': now.month,
        'day_of_year': now.timetuple().tm_yday,
        'quarter': (now.month - 1) // 3 + 1,
        'temp_range': weather_data.get('tmax', 25) - weather_data.get('tmin', 15),
        'high_humidity': 1 if weather_data.get('humidity', 60) > 70 else 0,
        'pressure_anomaly': weather_data.get('pres', 1010) - 1013,
        'prcp_7day_avg': weather_data.get('prcp', 0) * 0.8,
        'tavg_7day_avg': weather_data.get('tavg', 20),
        'wspd_7day_avg': weather_data.get('wspd', 10),
        'location_encoded': location_id
    }
    
    return pd.DataFrame([features])


def predict_flood_risk(model_data, features):
    """Make flood prediction using trained model"""
    if model_data is None:
        return 0.1, "No model available"
    
    try:
        model = model_data.get('model', model_data)
        threshold = model_data.get('threshold', 0.5)
        
        # Get prediction probability
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0][1]
        else:
            proba = float(model.predict(features)[0])
        
        # Determine risk level
        if proba >= 0.7:
            risk_level = "HIGH"
        elif proba >= 0.4:
            risk_level = "MODERATE"
        elif proba >= 0.2:
            risk_level = "LOW"
        else:
            risk_level = "VERY LOW"
        
        return proba, risk_level
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0.1, "Error"


def create_gauge_chart(probability, risk_level):
    """Create a gauge chart for flood risk"""
    if not PLOTLY_AVAILABLE:
        return None
    
    colors = {
        "VERY LOW": "#2ecc71",
        "LOW": "#f39c12",
        "MODERATE": "#e67e22",
        "HIGH": "#e74c3c"
    }
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Flood Risk: {risk_level}", 'font': {'size': 24}},
        delta={'reference': 30, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': colors.get(risk_level, "#3498db")},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#2ecc71'},
                {'range': [20, 40], 'color': '#f39c12'},
                {'range': [40, 70], 'color': '#e67e22'},
                {'range': [70, 100], 'color': '#e74c3c'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig


def main():
    """Main application function"""
    
    # Sidebar
    st.sidebar.title("🌊 Flood Prediction")
    st.sidebar.markdown("---")
    
    # Location selection
    selected_location = st.sidebar.selectbox(
        "Select Location",
        options=list(LOCATIONS.keys()),
        format_func=lambda x: LOCATIONS[x]["name"]
    )
    
    location = LOCATIONS[selected_location]
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Dashboard", "📊 Historical Data", "🤖 Model Info", "ℹ️ About"]
    )
    
    # Load model
    model_data = load_model()
    
    if page == "🏠 Dashboard":
        show_dashboard(location, model_data)
    elif page == "📊 Historical Data":
        show_historical_data(location)
    elif page == "🤖 Model Info":
        show_model_info(model_data)
    else:
        show_about()


def show_dashboard(location, model_data):
    """Display main dashboard"""
    st.title(f"🌊 Flood Risk Dashboard - {location['name']}")
    
    # Current weather
    st.subheader("📡 Current Weather Conditions")
    
    weather = fetch_weather_data(
        location['latitude'], 
        location['longitude'],
        OPENWEATHER_API_KEY
    )
    
    # Weather metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🌡️ Temperature", f"{weather['tavg']:.1f}°C")
    with col2:
        st.metric("💧 Humidity", f"{weather['humidity']:.0f}%")
    with col3:
        st.metric("🌧️ Precipitation", f"{weather['prcp']:.1f} mm")
    with col4:
        st.metric("💨 Wind Speed", f"{weather['wspd']:.1f} km/h")
    
    st.markdown("---")
    
    # Flood prediction
    st.subheader("🎯 Flood Risk Prediction")
    
    features = prepare_features(weather, location['location_id'])
    probability, risk_level = predict_flood_risk(model_data, features)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        gauge = create_gauge_chart(probability, risk_level)
        if gauge:
            st.plotly_chart(gauge, use_container_width=True)
        else:
            st.metric("Flood Risk", f"{probability*100:.1f}%", risk_level)
    
    with col2:
        st.markdown("### Risk Assessment")
        
        if risk_level == "HIGH":
            st.error("⚠️ **HIGH RISK** - Take immediate precautions!")
            st.markdown("""
            - Monitor official alerts
            - Prepare emergency supplies
            - Know evacuation routes
            - Stay away from rivers/streams
            """)
        elif risk_level == "MODERATE":
            st.warning("⚡ **MODERATE RISK** - Stay alert!")
            st.markdown("""
            - Monitor weather updates
            - Review emergency plans
            - Secure outdoor items
            """)
        else:
            st.success("✅ **LOW RISK** - Normal conditions")
            st.markdown("""
            - No immediate flood threat
            - Continue normal activities
            - Stay informed of weather changes
            """)
    
    # API mode indicator
    if OPENWEATHER_API_KEY == "demo":
        st.info("🔧 Running in demo mode. Set OPENWEATHER_API_KEY for live weather data.")


def show_historical_data(location):
    """Show historical flood data analysis"""
    st.title("📊 Historical Data Analysis")
    
    # Load historical dataset
    data_path = DATA_DIR / "flood_weather_dataset.csv"
    
    if data_path.exists():
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter by location
        loc_df = df[df['location_key'].str.contains(location['name'].split(',')[0].lower().replace(' ', '_'))]
        
        st.subheader(f"Data for {location['name']}")
        st.write(f"Records: {len(loc_df)}")
        
        # Time series plot
        if PLOTLY_AVAILABLE and len(loc_df) > 0:
            fig = px.line(loc_df, x='date', y='prcp', title='Precipitation Over Time')
            st.plotly_chart(fig, use_container_width=True)
            
            # Temperature
            fig2 = px.line(loc_df, x='date', y=['tmin', 'tavg', 'tmax'], 
                          title='Temperature Trends')
            st.plotly_chart(fig2, use_container_width=True)
        
        # Show data table
        with st.expander("View Raw Data"):
            st.dataframe(loc_df.head(100))
    else:
        st.warning("Historical data not found. Please run the data pipeline first.")


def show_model_info(model_data):
    """Display model information and metrics"""
    st.title("🤖 Model Information")
    
    # Load metrics
    metrics_path = RESULTS_DIR / "improved_model_metrics.csv"
    
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        
        st.subheader("📊 Model Performance Metrics")
        st.dataframe(metrics_df)
        
        # Best model info
        if model_data:
            st.subheader("🏆 Currently Loaded Model")
            if isinstance(model_data, dict):
                st.write(f"**Model:** {model_data.get('model_name', 'Unknown')}")
                st.write(f"**Threshold:** {model_data.get('threshold', 0.5):.4f}")
                
                if 'metrics' in model_data:
                    st.json(model_data['metrics'])
    else:
        st.warning("Metrics file not found. Please train the models first.")
    
    # Feature importance
    st.subheader("📈 Feature Importance")
    fi_path = RESULTS_DIR / "feature_importance.json"
    if fi_path.exists():
        with open(fi_path) as f:
            importance = json.load(f)
        st.json(importance)


def show_about():
    """Display about page"""
    st.title("ℹ️ About This Project")
    
    st.markdown("""
    ## AI-Based Natural Disaster Prediction Web App
    
    This application predicts flood risk for districts in Khyber Pakhtunkhwa, Pakistan
    using machine learning models trained on historical weather and flood data.
    
    ### 🎯 Features
    - **Real-time Weather Integration**: Live weather data from OpenWeatherMap API
    - **ML-Based Predictions**: Trained models predict flood probability
    - **Historical Analysis**: View past weather patterns and flood events
    - **Alert System**: Color-coded risk levels for quick assessment
    
    ### 📍 Covered Locations
    - Swat District
    - Upper Dir District
    
    ### 🔬 Technology Stack
    - **Frontend**: Streamlit
    - **ML Models**: Random Forest, Logistic Regression, Gradient Boosting
    - **Data Sources**: Meteostat, NASA POWER, NDMA Reports
    
    ### 👨‍💻 Developer
    CS351 - Artificial Intelligence Project
    
    ### ⚠️ Disclaimer
    This is an educational project. For actual emergency situations,
    please refer to official government sources and NDMA alerts.
    """)


if __name__ == "__main__":
    main()
