"""
AI-Based Natural Disaster Prediction Web App
Streamlit Interface for Real-time Flood Risk Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Flood Risk Predictor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .danger-card {
        background: linear-gradient(135deg, #ff9a56 0%, #ff6a88 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .header-main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS
# ============================================================================
@st.cache_resource
def load_models():
    """Load trained models from pickle files"""
    try:
        with open('results/logistic_regression_model.pkl', 'rb') as f:
            lr_model = pickle.load(f)
        with open('results/random_forest_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        return lr_model, rf_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

@st.cache_resource
def load_test_data():
    """Load test data for reference"""
    try:
        test_data = pd.read_csv('results/test_data.csv')
        return test_data
    except Exception as e:
        st.error(f"Error loading test data: {e}")
        return None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_feature_ranges(test_data):
    """Get min/max ranges for each feature"""
    feature_stats = {}
    for col in test_data.columns:
        if col != 'flood_event':
            feature_stats[col] = {
                'min': test_data[col].min(),
                'max': test_data[col].max(),
                'mean': test_data[col].mean(),
                'std': test_data[col].std()
            }
    return feature_stats

def create_sample_data(feature_dict):
    """Create a DataFrame from feature inputs"""
    return pd.DataFrame([feature_dict])

def get_prediction(model, features_df):
    """Get prediction and probability from model"""
    prediction = model.predict(features_df)[0]
    probability = model.predict_proba(features_df)[0]
    return prediction, probability

def get_risk_level(probability):
    """Convert probability to risk level"""
    flood_prob = probability[1]
    if flood_prob < 0.3:
        return "🟢 LOW RISK", flood_prob, "#00cc00"
    elif flood_prob < 0.6:
        return "🟡 MEDIUM RISK", flood_prob, "#ffcc00"
    else:
        return "🔴 HIGH RISK", flood_prob, "#ff0000"

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
    <div class="header-main">
        <h1>🌊 AI-Based Flood Risk Prediction System</h1>
        <p>Khyber Pakhtunkhwa - Real-time Weather Analysis</p>
        <p style="font-size: 14px; margin-top: 10px;">
            Predicting flood risks in Swat and Upper Dir districts using advanced machine learning
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS AND DATA
# ============================================================================
lr_model, rf_model = load_models()
test_data = load_test_data()

if lr_model is None or rf_model is None or test_data is None:
    st.error("❌ Failed to load models or test data. Please check if files exist in results/ directory.")
    st.stop()

feature_stats = get_feature_ranges(test_data)

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.title("📊 Navigation")
app_mode = st.sidebar.radio(
    "Select Mode:",
    ["🏠 Home", "🎯 Make Prediction", "📈 Model Performance", "📊 Data Analysis", "ℹ️ About"]
)

# ============================================================================
# PAGE: HOME
# ============================================================================
if app_mode == "🏠 Home":
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📌 Quick Start Guide")
        st.info("""
        **Welcome to the Flood Risk Predictor!**
        
        This application uses machine learning to predict flood risks in 
        Khyber Pakhtunkhwa province.
        
        **Features:**
        - ✅ Real-time flood risk prediction
        - ✅ Weather-based analysis
        - ✅ Model performance metrics
        - ✅ Historical data analysis
        """)
    
    with col2:
        st.markdown("### 📊 System Statistics")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric("Test Samples", f"{len(test_data):,}", "Records")
        
        with col_b:
            st.metric("Features Used", "19", "Weather Variables")
        
        with col_c:
            st.metric("Model Accuracy", "99.91%", "Test Set")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🤖 Models Available")
        st.success("""
        **1. Logistic Regression**
        - AUC-ROC: 0.8243
        - Fast predictions
        - Good baseline model
        
        **2. Random Forest** ⭐
        - AUC-ROC: 0.8643 (Better)
        - 200 decision trees
        - More accurate predictions
        """)
    
    with col2:
        st.markdown("### 🌍 Coverage Areas")
        st.info("""
        **Regions:**
        - 📍 Swat District
        - 📍 Upper Dir District
        
        **Weather Variables:**
        - Temperature (avg, min, max)
        - Precipitation
        - Wind Speed & Gust
        - Atmospheric Pressure
        - Humidity
        - Solar Radiation
        - Rolling Averages (7-day)
        """)

# ============================================================================
# PAGE: MAKE PREDICTION
# ============================================================================
elif app_mode == "🎯 Make Prediction":
    st.markdown("## 🎯 Real-time Flood Risk Prediction")
    
    # Tabs for input methods
    tab1, tab2, tab3 = st.tabs(["📝 Manual Input", "📂 Sample Data", "🎲 Random Data"])
    
    # ========== TAB 1: Manual Input ==========
    with tab1:
        st.markdown("### Enter Weather Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tavg = st.slider("Average Temperature (°C)", 
                            float(feature_stats['tavg']['min']), 
                            float(feature_stats['tavg']['max']), 
                            float(feature_stats['tavg']['mean']))
            tmin = st.slider("Minimum Temperature (°C)", 
                            float(feature_stats['tmin']['min']), 
                            float(feature_stats['tmin']['max']), 
                            float(feature_stats['tmin']['mean']))
            tmax = st.slider("Maximum Temperature (°C)", 
                            float(feature_stats['tmax']['min']), 
                            float(feature_stats['tmax']['max']), 
                            float(feature_stats['tmax']['mean']))
            prcp = st.slider("Precipitation (mm)", 
                            float(feature_stats['prcp']['min']), 
                            float(feature_stats['prcp']['max']), 
                            float(feature_stats['prcp']['mean']))
        
        with col2:
            wspd = st.slider("Wind Speed (m/s)", 
                            float(feature_stats['wspd']['min']), 
                            float(feature_stats['wspd']['max']), 
                            float(feature_stats['wspd']['mean']))
            wpgt = st.slider("Wind Gust (m/s)", 
                            float(feature_stats['wpgt']['min']), 
                            float(feature_stats['wpgt']['max']), 
                            float(feature_stats['wpgt']['mean']))
            pres = st.slider("Atmospheric Pressure (hPa)", 
                            float(feature_stats['pres']['min']), 
                            float(feature_stats['pres']['max']), 
                            float(feature_stats['pres']['mean']))
            humidity = st.slider("Humidity (%)", 
                                float(feature_stats['humidity']['min']), 
                                float(feature_stats['humidity']['max']), 
                                float(feature_stats['humidity']['mean']))
        
        with col3:
            solar_radiation = st.slider("Solar Radiation (MJ/m²)", 
                                       float(feature_stats['solar_radiation']['min']), 
                                       float(feature_stats['solar_radiation']['max']), 
                                       float(feature_stats['solar_radiation']['mean']))
            month = st.slider("Month (1-12)", 1, 12, 6)
            day_of_year = st.slider("Day of Year (1-365)", 1, 365, 180)
            quarter = st.slider("Quarter (1-4)", 1, 4, 2)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            temp_range = tmax - tmin
            st.metric("Temperature Range", f"{temp_range:.2f}°C", "Calculated")
        
        with col2:
            high_humidity = 1 if humidity > feature_stats['humidity']['mean'] else 0
            st.metric("High Humidity", "Yes" if high_humidity else "No", "Based on avg")
        
        with col3:
            pressure_anomaly = pres - feature_stats['pres']['mean']
            st.metric("Pressure Anomaly", f"{pressure_anomaly:.2f} hPa", "From mean")
        
        # Rolling averages
        st.markdown("### 7-Day Rolling Averages")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            prcp_7day_avg = st.slider("Precipitation 7-day avg (mm)", 
                                     float(feature_stats['prcp_7day_avg']['min']), 
                                     float(feature_stats['prcp_7day_avg']['max']), 
                                     float(feature_stats['prcp_7day_avg']['mean']))
        
        with col2:
            tavg_7day_avg = st.slider("Temperature 7-day avg (°C)", 
                                     float(feature_stats['tavg_7day_avg']['min']), 
                                     float(feature_stats['tavg_7day_avg']['max']), 
                                     float(feature_stats['tavg_7day_avg']['mean']))
        
        with col3:
            wspd_7day_avg = st.slider("Wind Speed 7-day avg (m/s)", 
                                     float(feature_stats['wspd_7day_avg']['min']), 
                                     float(feature_stats['wspd_7day_avg']['max']), 
                                     float(feature_stats['wspd_7day_avg']['mean']))
        
        location = st.radio("Location:", ["Swat", "Upper Dir"], horizontal=True)
        location_encoded = 1.0 if location == "Upper Dir" else -1.0
        
        # Create feature dictionary
        features = {
            'tavg': tavg,
            'tmin': tmin,
            'tmax': tmax,
            'prcp': prcp,
            'wspd': wspd,
            'wpgt': wpgt,
            'pres': pres,
            'humidity': humidity,
            'solar_radiation': solar_radiation,
            'month': month,
            'day_of_year': day_of_year,
            'quarter': quarter,
            'temp_range': temp_range,
            'high_humidity': high_humidity,
            'pressure_anomaly': pressure_anomaly,
            'prcp_7day_avg': prcp_7day_avg,
            'tavg_7day_avg': tavg_7day_avg,
            'wspd_7day_avg': wspd_7day_avg,
            'location_encoded': location_encoded
        }
        
        if st.button("🔮 Predict Flood Risk", use_container_width=True):
            features_df = create_sample_data(features)
            
            # Get predictions
            lr_pred, lr_prob = get_prediction(lr_model, features_df)
            rf_pred, rf_prob = get_prediction(rf_model, features_df)
            
            # Display results
            st.markdown("---")
            st.markdown("### 📊 Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🤖 Logistic Regression")
                lr_risk, lr_flood_prob, lr_color = get_risk_level(lr_prob)
                st.markdown(f"<div style='background: {lr_color}; padding: 20px; border-radius: 10px; color: white; text-align: center;'><h2>{lr_risk}</h2><h3>{lr_flood_prob:.2%}</h3></div>", unsafe_allow_html=True)
                
                with st.expander("📈 Detailed Probabilities"):
                    st.metric("Probability of No Flood", f"{lr_prob[0]:.4f}")
                    st.metric("Probability of Flood", f"{lr_prob[1]:.4f}")
            
            with col2:
                st.markdown("#### 🌲 Random Forest (Recommended)")
                rf_risk, rf_flood_prob, rf_color = get_risk_level(rf_prob)
                st.markdown(f"<div style='background: {rf_color}; padding: 20px; border-radius: 10px; color: white; text-align: center;'><h2>{rf_risk}</h2><h3>{rf_flood_prob:.2%}</h3></div>", unsafe_allow_html=True)
                
                with st.expander("📈 Detailed Probabilities"):
                    st.metric("Probability of No Flood", f"{rf_prob[0]:.4f}")
                    st.metric("Probability of Flood", f"{rf_prob[1]:.4f}")
            
            # Summary
            st.markdown("---")
            st.markdown("### 📋 Input Summary")
            summary_df = pd.DataFrame({
                'Feature': list(features.keys()),
                'Value': list(features.values())
            })
            st.dataframe(summary_df, use_container_width=True)
    
    # ========== TAB 2: Sample Data ==========
    with tab2:
        st.markdown("### Use Sample Data from Test Set")
        
        sample_index = st.slider("Select Sample #:", 0, len(test_data)-1, 0)
        sample_data = test_data.iloc[sample_index]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📊 Selected Sample Data")
            st.dataframe(sample_data.drop('flood_event'), use_container_width=True)
        
        with col2:
            st.markdown("#### 📝 Actual Result")
            actual_label = "🚨 FLOOD EVENT" if sample_data['flood_event'] == 1 else "✅ NO FLOOD"
            st.success(actual_label if sample_data['flood_event'] == 0 else st.error(actual_label))
        
        if st.button("🔮 Predict this Sample", use_container_width=True):
            features_df = sample_data.drop('flood_event').to_frame().T
            
            lr_pred, lr_prob = get_prediction(lr_model, features_df)
            rf_pred, rf_prob = get_prediction(rf_model, features_df)
            
            st.markdown("---")
            st.markdown("### 📊 Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🤖 Logistic Regression")
                lr_risk, lr_flood_prob, lr_color = get_risk_level(lr_prob)
                st.markdown(f"<div style='background: {lr_color}; padding: 20px; border-radius: 10px; color: white; text-align: center;'><h2>{lr_risk}</h2><h3>{lr_flood_prob:.2%}</h3></div>", unsafe_allow_html=True)
                lr_correct = "✅ Correct" if (lr_pred == sample_data['flood_event']) else "❌ Wrong"
                st.metric("Prediction Accuracy", lr_correct)
            
            with col2:
                st.markdown("#### 🌲 Random Forest (Recommended)")
                rf_risk, rf_flood_prob, rf_color = get_risk_level(rf_prob)
                st.markdown(f"<div style='background: {rf_color}; padding: 20px; border-radius: 10px; color: white; text-align: center;'><h2>{rf_risk}</h2><h3>{rf_flood_prob:.2%}</h3></div>", unsafe_allow_html=True)
                rf_correct = "✅ Correct" if (rf_pred == sample_data['flood_event']) else "❌ Wrong"
                st.metric("Prediction Accuracy", rf_correct)
    
    # ========== TAB 3: Random Data ==========
    with tab3:
        st.markdown("### Generate Random Weather Data")
        
        if st.button("🎲 Generate Random Data", use_container_width=True):
            random_features = {}
            for feature, stats in feature_stats.items():
                random_features[feature] = np.random.uniform(stats['min'], stats['max'])
            
            features_df = create_sample_data(random_features)
            
            lr_pred, lr_prob = get_prediction(lr_model, features_df)
            rf_pred, rf_prob = get_prediction(rf_model, features_df)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### 🎲 Random Weather Data")
                display_df = pd.DataFrame({
                    'Feature': list(random_features.keys()),
                    'Value': [f"{v:.2f}" for v in random_features.values()]
                })
                st.dataframe(display_df, use_container_width=True)
            
            with col2:
                st.markdown("#### 📊 Predictions")
                st.markdown("**🤖 Logistic Regression:**")
                lr_risk, lr_flood_prob, lr_color = get_risk_level(lr_prob)
                st.markdown(f"<div style='background: {lr_color}; padding: 15px; border-radius: 8px; color: white;'><p>{lr_risk}</p><p>Confidence: {lr_flood_prob:.2%}</p></div>", unsafe_allow_html=True)
                
                st.markdown("**🌲 Random Forest:**")
                rf_risk, rf_flood_prob, rf_color = get_risk_level(rf_prob)
                st.markdown(f"<div style='background: {rf_color}; padding: 15px; border-radius: 8px; color: white;'><p>{rf_risk}</p><p>Confidence: {rf_flood_prob:.2%}</p></div>", unsafe_allow_html=True)

# ============================================================================
# PAGE: MODEL PERFORMANCE
# ============================================================================
elif app_mode == "📈 Model Performance":
    st.markdown("## 📈 Model Performance Metrics")
    
    # Load metrics
    metrics_df = pd.read_csv('results/model_metrics.csv')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Performance Summary")
        st.dataframe(metrics_df, use_container_width=True)
    
    with col2:
        st.markdown("### 🏆 Model Comparison")
        comparison_data = {
            'Metric': ['Accuracy', 'AUC-ROC', 'Winner'],
            'Logistic Reg': ['99.91%', '0.8243', ''],
            'Random Forest': ['99.91%', '0.8643', '⭐']
        }
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
    
    # Display visualizations
    st.markdown("---")
    st.markdown("### 📈 Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.checkbox("Show Model Performance Comparison"):
            try:
                img = plt.imread('results/model_performance_comparison.png')
                st.image(img, caption="Model Performance Comparison", use_column_width=True)
            except:
                st.warning("Visualization not found")
    
    with col2:
        if st.checkbox("Show ROC Curves"):
            try:
                img = plt.imread('results/roc_curves.png')
                st.image(img, caption="ROC Curves", use_column_width=True)
            except:
                st.warning("Visualization not found")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.checkbox("Show Confusion Matrices"):
            try:
                img = plt.imread('results/confusion_matrices.png')
                st.image(img, caption="Confusion Matrices", use_column_width=True)
            except:
                st.warning("Visualization not found")
    
    with col2:
        if st.checkbox("Show Feature Importance Comparison"):
            try:
                img = plt.imread('results/model_performance_comparison.png')
                st.image(img, caption="Feature Analysis", use_column_width=True)
            except:
                st.warning("Visualization not found")

# ============================================================================
# PAGE: DATA ANALYSIS
# ============================================================================
elif app_mode == "📊 Data Analysis":
    st.markdown("## 📊 Data Analysis & Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Test Samples", f"{len(test_data):,}")
    with col2:
        st.metric("Number of Features", len(test_data.columns) - 1)
    with col3:
        flood_percentage = (test_data['flood_event'].sum() / len(test_data)) * 100
        st.metric("Flood Events", f"{flood_percentage:.2f}%")
    
    st.markdown("---")
    
    # Feature statistics
    st.markdown("### 📋 Feature Statistics")
    
    feature_cols = [col for col in test_data.columns if col != 'flood_event']
    selected_feature = st.selectbox("Select Feature:", feature_cols)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean", f"{test_data[selected_feature].mean():.2f}")
    with col2:
        st.metric("Median", f"{test_data[selected_feature].median():.2f}")
    with col3:
        st.metric("Min", f"{test_data[selected_feature].min():.2f}")
    with col4:
        st.metric("Max", f"{test_data[selected_feature].max():.2f}")
    
    # Distribution plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(test_data[selected_feature], bins=30, color='#667eea', edgecolor='black', alpha=0.7)
    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {selected_feature}")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Correlation analysis
    st.markdown("### 🔗 Feature Correlation")
    
    if st.checkbox("Show Correlation Matrix"):
        fig, ax = plt.subplots(figsize=(12, 10))
        correlation_matrix = test_data.corr()
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, ax=ax, square=True)
        ax.set_title("Feature Correlation Matrix")
        st.pyplot(fig)

# ============================================================================
# PAGE: ABOUT
# ============================================================================
elif app_mode == "ℹ️ About":
    st.markdown("## ℹ️ About This Application")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Project Overview
        
        This AI-Based Natural Disaster Prediction System is designed to predict 
        flood risks in Khyber Pakhtunkhwa province, specifically in Swat and Upper Dir districts.
        
        **Key Features:**
        - Real-time flood risk prediction
        - Weather-based analysis
        - Machine learning models
        - Interactive web interface
        
        **Technologies Used:**
        - Python 3.9
        - Pandas, NumPy, Scikit-learn
        - Streamlit for UI
        - Matplotlib, Seaborn for visualization
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 Models
        
        **Logistic Regression**
        - Fast and interpretable
        - AUC-ROC: 0.8243
        
        **Random Forest** ⭐
        - Ensemble method (200 trees)
        - Better performance
        - AUC-ROC: 0.8643
        
        ### 📊 Data
        
        - Total samples: 5,752
        - Test samples: 1,151
        - Features: 19 engineered features
        - Weather variables from multiple sources
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📂 Project Structure
    
    ```
    code/
    ├── preprocessing.py          # Data preprocessing
    ├── baseline_models.py        # Model training
    ├── model_evaluation.py       # Model evaluation
    └── clean_weather_pipeline.py # Weather data pipeline
    
    results/
    ├── model_metrics.csv         # Performance metrics
    ├── training_data.csv         # Preprocessed training data
    ├── test_data.csv            # Preprocessed test data
    ├── logistic_regression_model.pkl  # Trained LR model
    ├── random_forest_model.pkl        # Trained RF model
    └── *.png                    # Visualization files
    
    notebooks/
    └── ml_pipeline.ipynb        # Interactive notebook
    ```
    
    ### 📚 Documentation
    
    - `README.md` - Project overview
    - `ENVIRONMENT_SETUP.md` - Environment setup guide
    - `ML_PIPELINE_README.md` - ML pipeline documentation
    - `XGBOOST_ERROR_RESOLUTION.md` - Error resolution guide
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 👨‍💻 Developer
    
    **Developed by:** Shafiq Hussain  
    **Date:** November 2025  
    **Version:** 1.0.0
    
    ### 📞 Contact & Support
    
    For questions or issues, please refer to the documentation or contact the development team.
    
    ### 📜 License
    
    This project is provided as-is for educational and research purposes.
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 20px; color: #666;">
    <p>🌊 AI-Based Natural Disaster Prediction System | Flood Risk Predictor for Khyber Pakhtunkhwa</p>
    <p style="font-size: 12px;">© 2025 - All Rights Reserved | Version 1.0.0</p>
    </div>
    """, unsafe_allow_html=True)
