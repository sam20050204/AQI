import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# For forecasting models
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Time series forecasting libraries
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.warning("Prophet not available. Install with: pip install prophet")

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    st.warning("ARIMA not available. Install with: pip install statsmodels")

# Configuration
st.set_page_config(
    page_title="AirAware - Smart Air Quality Prediction",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}

.metric-card {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}

.alert-good {
    background-color: #d4edda;
    color: #155724;
    padding: 1rem;
    border-radius: 5px;
    border-left: 4px solid #28a745;
}

.alert-moderate {
    background-color: #fff3cd;
    color: #856404;
    padding: 1rem;
    border-radius: 5px;
    border-left: 4px solid #ffc107;
}

.alert-unhealthy {
    background-color: #f8d7da;
    color: #721c24;
    padding: 1rem;
    border-radius: 5px;
    border-left: 4px solid #dc3545;
}

.alert-hazardous {
    background-color: #d1ecf1;
    color: #0c5460;
    padding: 1rem;
    border-radius: 5px;
    border-left: 4px solid #17a2b8;
}
</style>
""", unsafe_allow_html=True)

class AirQualityPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.data = None
        
    def calculate_aqi(self, pm25, pm10, no2, o3, co, so2):
        """Calculate AQI from pollutant concentrations"""
        # Simplified AQI calculation based on PM2.5 (most critical)
        if pm25 <= 12:
            aqi = (50/12) * pm25
        elif pm25 <= 35.4:
            aqi = ((100-51)/(35.4-12.1)) * (pm25-12.1) + 51
        elif pm25 <= 55.4:
            aqi = ((150-101)/(55.4-35.5)) * (pm25-35.5) + 101
        elif pm25 <= 150.4:
            aqi = ((200-151)/(150.4-55.5)) * (pm25-55.5) + 151
        elif pm25 <= 250.4:
            aqi = ((300-201)/(250.4-150.5)) * (pm25-150.5) + 201
        else:
            aqi = ((500-301)/(500.4-250.5)) * (pm25-250.5) + 301
        
        return min(aqi, 500)
    
    def get_aqi_category(self, aqi):
        """Get AQI category and color"""
        if aqi <= 50:
            return "Good", "#28a745"
        elif aqi <= 100:
            return "Moderate", "#ffc107"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups", "#fd7e14"
        elif aqi <= 200:
            return "Unhealthy", "#dc3545"
        elif aqi <= 300:
            return "Very Unhealthy", "#6f42c1"
        else:
            return "Hazardous", "#495057"
    
    def generate_synthetic_data(self, city="Delhi", days=365):
        """Generate realistic synthetic air quality data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        
        # City-specific seed and pollution characteristics
        city_configs = {
            "Delhi": {"seed": 42, "base_pollution": 1.8, "seasonal_factor": 1.5, "traffic_factor": 1.4},
            "Mumbai": {"seed": 123, "base_pollution": 1.3, "seasonal_factor": 1.2, "traffic_factor": 1.3},
            "Bangalore": {"seed": 456, "base_pollution": 1.1, "seasonal_factor": 1.1, "traffic_factor": 1.2},
            "Chennai": {"seed": 789, "base_pollution": 1.2, "seasonal_factor": 1.0, "traffic_factor": 1.1},
            "Kolkata": {"seed": 321, "base_pollution": 1.6, "seasonal_factor": 1.4, "traffic_factor": 1.3},
            "Hyderabad": {"seed": 654, "base_pollution": 1.2, "seasonal_factor": 1.2, "traffic_factor": 1.2},
            "Pune": {"seed": 987, "base_pollution": 1.1, "seasonal_factor": 1.1, "traffic_factor": 1.1},
            "Ahmedabad": {"seed": 147, "base_pollution": 1.4, "seasonal_factor": 1.3, "traffic_factor": 1.2}
        }
        
        config = city_configs.get(city, city_configs["Delhi"])
        np.random.seed(config["seed"])
        n_points = len(date_range)
        
        # Generate seasonal patterns
        day_of_year = date_range.dayofyear
        hour_of_day = date_range.hour
        
        # City-specific seasonal effect (higher pollution in winter)
        seasonal_effect = config["seasonal_factor"] + 0.8 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # Daily cycle (higher pollution during rush hours) - city-specific traffic
        daily_cycle = config["traffic_factor"] + 0.3 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        
        # Base pollution levels with city-specific multipliers
        base_multiplier = config["base_pollution"]
        base_pm25 = (40 * base_multiplier) + 20 * seasonal_effect * daily_cycle + np.random.normal(0, 10, n_points)
        base_pm10 = base_pm25 * 1.5 + np.random.normal(0, 15, n_points)
        base_no2 = (30 * base_multiplier) + 15 * seasonal_effect + np.random.normal(0, 8, n_points)
        base_o3 = 50 + 20 * np.sin(2 * np.pi * hour_of_day / 24) + np.random.normal(0, 12, n_points)
        base_co = (1.2 * base_multiplier) + 0.8 * seasonal_effect + np.random.normal(0, 0.3, n_points)
        base_so2 = (15 * base_multiplier) + 10 * seasonal_effect + np.random.normal(0, 5, n_points)
        
        # Ensure non-negative values
        base_pm25 = np.maximum(base_pm25, 5)
        base_pm10 = np.maximum(base_pm10, 10)
        base_no2 = np.maximum(base_no2, 5)
        base_o3 = np.maximum(base_o3, 10)
        base_co = np.maximum(base_co, 0.1)
        base_so2 = np.maximum(base_so2, 1)
        
        # Calculate AQI
        aqi_values = [self.calculate_aqi(pm25, pm10, no2, o3, co, so2) 
                     for pm25, pm10, no2, o3, co, so2 in 
                     zip(base_pm25, base_pm10, base_no2, base_o3, base_co, base_so2)]
        
        # City-specific temperature and humidity patterns
        temp_base = {"Delhi": 25, "Mumbai": 28, "Bangalore": 22, "Chennai": 29, 
                    "Kolkata": 26, "Hyderabad": 26, "Pune": 24, "Ahmedabad": 27}
        humidity_base = {"Delhi": 55, "Mumbai": 75, "Bangalore": 65, "Chennai": 70,
                        "Kolkata": 70, "Hyderabad": 60, "Pune": 60, "Ahmedabad": 55}
        
        temp_variation = 10 + 5 * config["base_pollution"]  # Cities with more pollution have more extreme temps
        humidity_variation = 15 + 5 * (2 - config["base_pollution"])  # Less polluted cities more humid
        
        temperature = (temp_base.get(city, 25) + 
                      temp_variation * np.sin(2 * np.pi * day_of_year / 365) + 
                      np.random.normal(0, 3, n_points))
        
        humidity = (humidity_base.get(city, 60) + 
                   humidity_variation * np.sin(2 * np.pi * (day_of_year - 180) / 365) + 
                   np.random.normal(0, 8, n_points))
        
        # Create DataFrame
        data = pd.DataFrame({
            'datetime': date_range,
            'city': city,
            'pm25': base_pm25,
            'pm10': base_pm10,
            'no2': base_no2,
            'o3': base_o3,
            'co': base_co,
            'so2': base_so2,
            'aqi': aqi_values,
            'temperature': temperature,
            'humidity': humidity
        })
        
        # Add derived features
        data['hour'] = data['datetime'].dt.hour
        data['day_of_week'] = data['datetime'].dt.dayofweek
        data['month'] = data['datetime'].dt.month
        data['season'] = data['month'].map({12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3})
        
        return data
    
    def prepare_features(self, data, target_col, lag_periods=[1, 6, 12, 24]):
        """Prepare features for forecasting"""
        df = data.copy()
        
        # Create lag features
        for lag in lag_periods:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        # Rolling statistics
        df[f'{target_col}_roll_mean_6'] = df[target_col].rolling(window=6).mean()
        df[f'{target_col}_roll_std_6'] = df[target_col].rolling(window=6).std()
        df[f'{target_col}_roll_mean_24'] = df[target_col].rolling(window=24).mean()
        
        # Drop rows with NaN values
        df = df.dropna()
        
        return df
    
    def train_arima_model(self, data, target_col, order=(2,1,2)):
        """Train ARIMA model"""
        if not ARIMA_AVAILABLE:
            return None
        
        try:
            model = ARIMA(data[target_col], order=order)
            fitted_model = model.fit()
            return fitted_model
        except Exception as e:
            st.error(f"Error training ARIMA model: {str(e)}")
            return None
    
    def train_prophet_model(self, data, target_col):
        """Train Prophet model"""
        if not PROPHET_AVAILABLE:
            return None
        
        try:
            # Prepare data for Prophet
            prophet_data = data[['datetime', target_col]].copy()
            prophet_data.columns = ['ds', 'y']
            
            # Initialize and fit Prophet model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10,
                interval_width=0.8
            )
            model.fit(prophet_data)
            return model
        except Exception as e:
            st.error(f"Error training Prophet model: {str(e)}")
            return None
    
    def make_predictions(self, model, model_type, data, periods=24):
        """Make predictions using trained model"""
        try:
            if model_type == 'prophet' and model:
                # Create future dataframe
                future = model.make_future_dataframe(periods=periods, freq='H')
                forecast = model.predict(future)
                return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
            
            elif model_type == 'arima' and model:
                # Make ARIMA forecast
                forecast = model.forecast(steps=periods)
                conf_int = model.get_forecast(steps=periods).conf_int()
                
                future_dates = pd.date_range(
                    start=data['datetime'].iloc[-1] + timedelta(hours=1),
                    periods=periods,
                    freq='H'
                )
                
                return pd.DataFrame({
                    'ds': future_dates,
                    'yhat': forecast,
                    'yhat_lower': conf_int.iloc[:, 0],
                    'yhat_upper': conf_int.iloc[:, 1]
                })
            
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")
            return None

def main():
    st.markdown('<h1 class="main-header">🌬️ AirAware Smart Air Quality Prediction</h1>', unsafe_allow_html=True)
    
    # Initialize predictor
    if 'predictor' not in st.session_state:
        st.session_state.predictor = AirQualityPredictor()
    
    predictor = st.session_state.predictor
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    # City selection
    cities = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune", "Ahmedabad"]
    selected_city = st.sidebar.selectbox("Select City", cities)
    
    # Add city info
    city_info = {
        "Delhi": "🏛️ National Capital - High industrial pollution",
        "Mumbai": "🏙️ Financial Capital - Coastal city with moderate pollution", 
        "Bangalore": "💻 IT Hub - Relatively cleaner air quality",
        "Chennai": "🌊 Coastal Metro - Industrial and vehicular pollution",
        "Kolkata": "🏭 Industrial City - High PM2.5 levels",
        "Hyderabad": "⚡ Cyberabad - Growing urban pollution",
        "Pune": "🎓 Educational Hub - Moderate pollution levels",
        "Ahmedabad": "🏭 Industrial Center - Dust and industrial pollution"
    }
    
    st.sidebar.info(city_info.get(selected_city, "Metropolitan city"))
    
    # Force refresh button
    if st.sidebar.button("🔄 Refresh Data", help="Generate new data for selected city"):
        # Clear cached data for this city
        if f'data_{selected_city}' in st.session_state:
            del st.session_state[f'data_{selected_city}']
        st.rerun()
    
    # Date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(start_date, end_date),
        max_value=end_date
    )
    
    # Model selection
    available_models = []
    if PROPHET_AVAILABLE:
        available_models.append("Prophet")
    if ARIMA_AVAILABLE:
        available_models.append("ARIMA")
    
    if available_models:
        selected_model = st.sidebar.selectbox("Select Forecasting Model", available_models)
    else:
        st.error("No forecasting models available. Please install Prophet or statsmodels.")
        return
    
    # Forecast horizon
    forecast_hours = st.sidebar.slider("Forecast Hours", 6, 72, 24)
    
    # Generate data (cached per city)
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_city_data(city_name):
        return predictor.generate_synthetic_data(city=city_name, days=365)
    
    with st.spinner(f"Loading air quality data for {selected_city}..."):
        data = get_city_data(selected_city)
    
    # Display city-specific stats
    recent_data = data.tail(24)
    
    st.sidebar.markdown(f"**📊 Last 24h Stats for {selected_city}:**")
    st.sidebar.metric("Avg AQI", f"{recent_data['aqi'].mean():.0f}")
    st.sidebar.metric("Peak AQI", f"{recent_data['aqi'].max():.0f}")
    st.sidebar.metric("Avg Temp", f"{recent_data['temperature'].mean():.1f}°C")
    
    # Filter data based on date range
    if len(date_range) == 2:
        start_dt = pd.to_datetime(date_range[0])
        end_dt = pd.to_datetime(date_range[1]) + timedelta(days=1)
        filtered_data = data[(data['datetime'] >= start_dt) & (data['datetime'] <= end_dt)]
    else:
        filtered_data = data.tail(24*30)  # Last 30 days
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    # Current AQI with city-specific styling
    current_aqi = filtered_data['aqi'].iloc[-1]
    aqi_category, aqi_color = predictor.get_aqi_category(current_aqi)
    prev_day_aqi = filtered_data['aqi'].iloc[-25] if len(filtered_data) > 24 else current_aqi
    aqi_change = current_aqi - prev_day_aqi
    
    with col1:
        st.metric(
            "Current AQI", 
            f"{current_aqi:.0f}", 
            delta=f"{aqi_change:.0f}",
            delta_color="inverse"
        )
    
    with col2:
        current_pm25 = filtered_data['pm25'].iloc[-1]
        prev_pm25 = filtered_data['pm25'].iloc[-25] if len(filtered_data) > 24 else current_pm25
        st.metric(
            "PM2.5", 
            f"{current_pm25:.1f} µg/m³",
            delta=f"{current_pm25 - prev_pm25:.1f}",
            delta_color="inverse"
        )
    
    with col3:
        current_pm10 = filtered_data['pm10'].iloc[-1]
        prev_pm10 = filtered_data['pm10'].iloc[-25] if len(filtered_data) > 24 else current_pm10
        st.metric(
            "PM10", 
            f"{current_pm10:.1f} µg/m³",
            delta=f"{current_pm10 - prev_pm10:.1f}",
            delta_color="inverse"
        )
    
    with col4:
        current_no2 = filtered_data['no2'].iloc[-1]
        prev_no2 = filtered_data['no2'].iloc[-25] if len(filtered_data) > 24 else current_no2
        st.metric(
            "NO2", 
            f"{current_no2:.1f} µg/m³",
            delta=f"{current_no2 - prev_no2:.1f}",
            delta_color="inverse"
        )
    
    # AQI Status Alert with city name
    alert_class = f"alert-{aqi_category.lower().replace(' ', '-').replace('unhealthy-for-sensitive-groups', 'moderate')}"
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.markdown(f'''
    <div class="{alert_class}">
        <strong>🏙️ {selected_city} Air Quality Status ({current_time}):</strong> {aqi_category} (AQI: {current_aqi:.0f})
        <br><small>Last updated: {filtered_data['datetime'].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")}</small>
    </div>
    ''', unsafe_allow_html=True)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        f"📈 {selected_city} Current Trends", 
        f"🔮 {selected_city} Forecasting", 
        f"⚠️ {selected_city} Alerts & Analysis", 
        f"📊 {selected_city} Historical Data"
    ])
    
    with tab1:
        st.subheader("Real-time Air Quality Monitoring")
        
        # Recent trends
        recent_data = filtered_data.tail(168)  # Last 7 days
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('AQI Trend', 'PM2.5 & PM10', 'NO2 & O3', 'Temperature & Humidity'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": True}]]
        )
        
        # AQI trend
        fig.add_trace(
            go.Scatter(x=recent_data['datetime'], y=recent_data['aqi'], name='AQI', line=dict(color='red', width=2)),
            row=1, col=1
        )
        
        # PM2.5 & PM10
        fig.add_trace(
            go.Scatter(x=recent_data['datetime'], y=recent_data['pm25'], name='PM2.5', line=dict(color='orange')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=recent_data['datetime'], y=recent_data['pm10'], name='PM10', line=dict(color='brown')),
            row=1, col=2
        )
        
        # NO2 & O3
        fig.add_trace(
            go.Scatter(x=recent_data['datetime'], y=recent_data['no2'], name='NO2', line=dict(color='blue')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=recent_data['datetime'], y=recent_data['o3'], name='O3', line=dict(color='green')),
            row=2, col=1
        )
        
        # Temperature & Humidity
        fig.add_trace(
            go.Scatter(x=recent_data['datetime'], y=recent_data['temperature'], name='Temperature', line=dict(color='red')),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=recent_data['datetime'], y=recent_data['humidity'], name='Humidity', yaxis='y2', line=dict(color='blue')),
            row=2, col=2, secondary_y=True
        )
        
        fig.update_layout(height=600, showlegend=True, title=f"Air Quality Trends - {selected_city}")
        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="AQI", row=1, col=1)
        fig.update_yaxes(title_text="Concentration (µg/m³)", row=1, col=2)
        fig.update_yaxes(title_text="Concentration (µg/m³)", row=2, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=2, col=2)
        fig.update_yaxes(title_text="Humidity (%)", row=2, col=2, secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🔮 Air Quality Forecasting")
        
        if st.button("Generate Forecast", type="primary"):
            with st.spinner(f"Training {selected_model} model and generating forecast..."):
                
                # Train model
                if selected_model == "Prophet":
                    model = predictor.train_prophet_model(data, 'aqi')
                    model_type = 'prophet'
                elif selected_model == "ARIMA":
                    model = predictor.train_arima_model(data, 'aqi')
                    model_type = 'arima'
                
                if model:
                    # Make predictions
                    predictions = predictor.make_predictions(model, model_type, data, forecast_hours)
                    
                    if predictions is not None:
                        # Display forecast
                        st.success(f"✅ Forecast generated successfully using {selected_model}!")
                        
                        # Combine historical and forecast data for plotting
                        historical = data.tail(168)[['datetime', 'aqi']].rename(columns={'datetime': 'ds', 'aqi': 'yhat'})
                        historical['type'] = 'Historical'
                        
                        predictions_plot = predictions[['ds', 'yhat']].copy()
                        predictions_plot['type'] = 'Forecast'
                        
                        combined_data = pd.concat([historical, predictions_plot])
                        
                        # Create forecast plot
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=historical['ds'],
                            y=historical['yhat'],
                            mode='lines',
                            name='Historical AQI',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Forecast data
                        fig.add_trace(go.Scatter(
                            x=predictions['ds'],
                            y=predictions['yhat'],
                            mode='lines+markers',
                            name='Forecasted AQI',
                            line=dict(color='red', width=2, dash='dash'),
                            marker=dict(size=6)
                        ))
                        
                        # Confidence intervals (if available)
                        if 'yhat_lower' in predictions.columns and 'yhat_upper' in predictions.columns:
                            fig.add_trace(go.Scatter(
                                x=predictions['ds'],
                                y=predictions['yhat_upper'],
                                fill=None,
                                mode='lines',
                                line_color='rgba(0,0,0,0)',
                                showlegend=False
                            ))
                            fig.add_trace(go.Scatter(
                                x=predictions['ds'],
                                y=predictions['yhat_lower'],
                                fill='tonexty',
                                mode='lines',
                                line_color='rgba(0,0,0,0)',
                                name='Confidence Interval',
                                fillcolor='rgba(255,0,0,0.2)'
                            ))
                        
                        fig.update_layout(
                            title=f"AQI Forecast for {selected_city} - Next {forecast_hours} Hours",
                            xaxis_title="Time",
                            yaxis_title="AQI",
                            hovermode='x unified',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show forecast summary
                        st.subheader("📋 Forecast Summary")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            avg_forecast = predictions['yhat'].mean()
                            st.metric("Average Forecast AQI", f"{avg_forecast:.1f}")
                        
                        with col2:
                            max_forecast = predictions['yhat'].max()
                            st.metric("Peak AQI", f"{max_forecast:.1f}")
                        
                        with col3:
                            high_aqi_hours = len(predictions[predictions['yhat'] > 100])
                            st.metric("Hours Above 100 AQI", high_aqi_hours)
                        
                        # Detailed forecast table
                        st.subheader("📋 Hourly Forecast Details")
                        forecast_display = predictions.copy()
                        forecast_display['AQI Category'] = forecast_display['yhat'].apply(lambda x: predictor.get_aqi_category(x)[0])
                        forecast_display['Time'] = forecast_display['ds'].dt.strftime('%Y-%m-%d %H:%M')
                        forecast_display['AQI'] = forecast_display['yhat'].round(1)
                        
                        display_cols = ['Time', 'AQI', 'AQI Category']
                        if 'yhat_lower' in forecast_display.columns:
                            forecast_display['Lower Bound'] = forecast_display['yhat_lower'].round(1)
                            forecast_display['Upper Bound'] = forecast_display['yhat_upper'].round(1)
                            display_cols.extend(['Lower Bound', 'Upper Bound'])
                        
                        st.dataframe(forecast_display[display_cols], use_container_width=True)
        
    with tab3:
        st.subheader("⚠️ Alert System & Trend Analysis")
        
        # Alert thresholds
        st.markdown("### 🚨 Air Quality Alerts")
        
        # Check for alerts in recent data
        recent_24h = filtered_data.tail(24)
        alerts = []
        
        # High AQI alert
        high_aqi = recent_24h[recent_24h['aqi'] > 150]
        if len(high_aqi) > 0:
            alerts.append({
                'type': 'High AQI',
                'count': len(high_aqi),
                'max_value': high_aqi['aqi'].max(),
                'time': high_aqi.loc[high_aqi['aqi'].idxmax(), 'datetime']
            })
        
        # High PM2.5 alert
        high_pm25 = recent_24h[recent_24h['pm25'] > 35]
        if len(high_pm25) > 0:
            alerts.append({
                'type': 'High PM2.5',
                'count': len(high_pm25),
                'max_value': high_pm25['pm25'].max(),
                'time': high_pm25.loc[high_pm25['pm25'].idxmax(), 'datetime']
            })
        
        if alerts:
            for alert in alerts:
                st.error(f"🚨 {alert['type']} Alert: {alert['count']} readings above threshold in last 24h. Peak: {alert['max_value']:.1f} at {alert['time'].strftime('%H:%M')}")
        else:
            st.success("✅ No air quality alerts in the last 24 hours")
        
        # Trend analysis
        st.markdown("### 📈 Trend Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Weekly trend
            weekly_avg = filtered_data.groupby(filtered_data['datetime'].dt.date).agg({
                'aqi': 'mean',
                'pm25': 'mean',
                'pm10': 'mean'
            }).reset_index()
            
            fig_trend = px.line(weekly_avg, x='datetime', y=['aqi', 'pm25', 'pm10'],
                               title="Daily Average Trends")
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with col2:
            # Hourly pattern
            hourly_avg = filtered_data.groupby('hour').agg({
                'aqi': 'mean',
                'pm25': 'mean',
                'no2': 'mean'
            }).reset_index()
            
            fig_hourly = px.bar(hourly_avg, x='hour', y='aqi',
                               title="Average AQI by Hour of Day")
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        # Seasonal analysis
        st.markdown("### 🍂 Seasonal Patterns")
        monthly_avg = filtered_data.groupby('month').agg({
            'aqi': ['mean', 'std'],
            'pm25': ['mean', 'std']
        }).reset_index()
        
        monthly_avg.columns = ['month', 'aqi_mean', 'aqi_std', 'pm25_mean', 'pm25_std']
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_avg['month_name'] = monthly_avg['month'].apply(lambda x: month_names[x-1])
        
        fig_seasonal = make_subplots(rows=1, cols=2, subplot_titles=['AQI by Month', 'PM2.5 by Month'])
        
        fig_seasonal.add_trace(
            go.Bar(x=monthly_avg['month_name'], y=monthly_avg['aqi_mean'], 
                   name='AQI', error_y=dict(type='data', array=monthly_avg['aqi_std'])),
            row=1, col=1
        )
        
        fig_seasonal.add_trace(
            go.Bar(x=monthly_avg['month_name'], y=monthly_avg['pm25_mean'], 
                   name='PM2.5', error_y=dict(type='data', array=monthly_avg['pm25_std'])),
            row=1, col=2
        )
        
        fig_seasonal.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_seasonal, use_container_width=True)
    
    with tab4:
        st.subheader("📊 Historical Data Analysis")
        
        # Data summary statistics
        st.markdown("### 📋 Data Summary")
        
        summary_stats = filtered_data[['aqi', 'pm25', 'pm10', 'no2', 'o3', 'co', 'so2']].describe()
        st.dataframe(summary_stats, use_container_width=True)
        
        # Correlation analysis
        st.markdown("### 🔗 Pollutant Correlations")
        
        corr_data = filtered_data[['aqi', 'pm25', 'pm10', 'no2', 'o3', 'co', 'so2', 'temperature', 'humidity']].corr()
        
        fig_corr = px.imshow(corr_data, 
                            title="Pollutant Correlation Matrix",
                            color_continuous_scale='RdBu_r',
                            aspect="auto")
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Distribution plots
        st.markdown("### 📊 Pollutant Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_dist1 = px.histogram(filtered_data, x='aqi', nbins=30, 
                                   title="AQI Distribution")
            st.plotly_chart(fig_dist1, use_container_width=True)
        
        with col2:
            fig_dist2 = px.histogram(filtered_data, x='pm25', nbins=30,
                                   title="PM2.5 Distribution")
            st.plotly_chart(fig_dist2, use_container_width=True)
        
        # Time series decomposition visualization
        st.markdown("### 📈 Time Series Components")
        
        # Simple trend analysis
        filtered_data['rolling_mean'] = filtered_data['aqi'].rolling(window=24).mean()
        filtered_data['rolling_std'] = filtered_data['aqi'].rolling(window=24).std()
        
        fig_decomp = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Original AQI', 'Rolling Mean (24h)', 'Rolling Std (24h)'],
            vertical_spacing=0.1
        )
        
        fig_decomp.add_trace(
            go.Scatter(x=filtered_data['datetime'], y=filtered_data['aqi'], 
                      name='AQI', line=dict(color='blue')),
            row=1, col=1
        )
        
        fig_decomp.add_trace(
            go.Scatter(x=filtered_data['datetime'], y=filtered_data['rolling_mean'], 
                      name='24h Mean', line=dict(color='red')),
            row=2, col=1
        )
        
        fig_decomp.add_trace(
            go.Scatter(x=filtered_data['datetime'], y=filtered_data['rolling_std'], 
                      name='24h Std', line=dict(color='green')),
            row=3, col=1
        )
        
        fig_decomp.update_layout(height=600, showlegend=False)
        fig_decomp.update_xaxes(title_text="Time", row=3, col=1)
        st.plotly_chart(fig_decomp, use_container_width=True)
        
        # Data export option
        st.markdown("### 💾 Data Export")
        
        if st.button("Download Current Data as CSV"):
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"air_quality_data_{selected_city}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    # Footer with live data indicator
    st.markdown("---")
    
    # City comparison
    st.markdown("### 🏙️ Quick City Comparison")
    
    # Generate data for all cities for comparison
    comparison_data = []
    for city in cities[:4]:  # Show top 4 cities for performance
        city_data = get_city_data(city)
        latest = city_data.tail(1).iloc[0]
        comparison_data.append({
            'City': city,
            'Current AQI': f"{latest['aqi']:.0f}",
            'PM2.5': f"{latest['pm25']:.1f}",
            'Temperature': f"{latest['temperature']:.1f}°C",
            'Status': predictor.get_aqi_category(latest['aqi'])[0]
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    st.markdown(f"""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>🌟 AirAware Smart Air Quality Prediction System</strong></p>
        <p>🌍 Real-time air quality monitoring and forecasting for healthier cities</p>
        <p><small>
            📡 Live data simulation • 🔄 Auto-refresh every 5 minutes • 
            🏙️ Currently monitoring: <strong>{selected_city}</strong> • 
            ⏰ Last update: {datetime.now().strftime('%H:%M:%S')}
        </small></p>
        <p><small>🎯 <strong>Tip:</strong> Select different cities to see varying pollution patterns and forecasts!</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()