import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import warnings
import time
warnings.filterwarnings('ignore')

# Configuration
st.set_page_config(
    page_title="AirAware - Real Air Quality Data",
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
</style>
""", unsafe_allow_html=True)

class RealAirQualityPredictor:
    def __init__(self):
        self.city_coords = {
            "Delhi": (28.6139, 77.2090),
            "Mumbai": (19.0760, 72.8777),
            "Bangalore": (12.9716, 77.5946),
            "Chennai": (13.0827, 80.2707),
            "Kolkata": (22.5726, 88.3639),
            "Hyderabad": (17.3850, 78.4867),
            "Pune": (18.5204, 73.8567),
            "Ahmedabad": (23.0225, 72.5714)
        }
    
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
    
    def calculate_aqi_from_pm25(self, pm25):
        """Calculate AQI from PM2.5 concentration"""
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
    
    def fetch_openmeteo_data(self, city):
        """Fetch air quality data from Open-Meteo API"""
        if city not in self.city_coords:
            return None, "City not found"
        
        lat, lon = self.city_coords[city]
        
        try:
            url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone&timezone=auto"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                return None, f"API Error: {response.status_code}"
            
            data = response.json()
            current = data['current']
            
            pm25 = current.get('pm2_5', 0)
            aqi = self.calculate_aqi_from_pm25(pm25) if pm25 > 0 else 50
            
            current_data = {
                'city': city,
                'datetime': datetime.now(),
                'pm25': pm25,
                'pm10': current.get('pm10', 0),
                'no2': current.get('nitrogen_dioxide', 0),
                'o3': current.get('ozone', 0),
                'co': current.get('carbon_monoxide', 0) / 1000,
                'so2': current.get('sulphur_dioxide', 0),
                'aqi': aqi
            }
            
            return current_data, "Success"
            
        except Exception as e:
            return None, f"Error: {str(e)}"

def main():
    # Debug info
    st.sidebar.write("🔧 Debug Info")
    st.sidebar.write(f"Streamlit version: {st.__version__}")
    
    try:
        st.markdown('<h1 class="main-header">🌬️ AirAware - Real Air Quality Data</h1>', unsafe_allow_html=True)
        
        # Initialize predictor
        if 'predictor' not in st.session_state:
            st.session_state.predictor = RealAirQualityPredictor()
        
        predictor = st.session_state.predictor
        
        # Sidebar
        st.sidebar.title("🎛️ Control Panel")
        
        # City selection
        cities = list(predictor.city_coords.keys())
        selected_city = st.sidebar.selectbox("Select City", cities)
        
        # Fetch data button
        if st.sidebar.button("🔄 Fetch Live Data", type="primary"):
            with st.spinner(f"Fetching live data for {selected_city}..."):
                current_data, status = predictor.fetch_openmeteo_data(selected_city)
                
                if current_data:
                    st.session_state[f'live_data_{selected_city}'] = current_data
                    st.sidebar.success("✅ Data fetched!")
                else:
                    st.sidebar.error(f"❌ {status}")
        
        # Display data
        current_data = st.session_state.get(f'live_data_{selected_city}')
        
        if current_data:
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current AQI", f"{current_data['aqi']:.0f}")
            
            with col2:
                st.metric("PM2.5", f"{current_data['pm25']:.1f} µg/m³")
            
            with col3:
                st.metric("PM10", f"{current_data['pm10']:.1f} µg/m³")
            
            with col4:
                st.metric("NO2", f"{current_data['no2']:.1f} µg/m³")
            
            # Status
            aqi_category, aqi_color = predictor.get_aqi_category(current_data['aqi'])
            alert_class = "alert-good" if current_data['aqi'] <= 50 else "alert-moderate" if current_data['aqi'] <= 100 else "alert-unhealthy"
            
            st.markdown(f'''
            <div class="{alert_class}">
                <strong>🏙️ {selected_city} Air Quality:</strong> {aqi_category} (AQI: {current_data['aqi']:.0f})
                <br><small>Last updated: {current_data['datetime'].strftime("%H:%M:%S")}</small>
            </div>
            ''', unsafe_allow_html=True)
            
            # Charts
            st.subheader("📊 Pollutant Levels")
            
            pollutants = ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2']
            values = [
                current_data['pm25'], 
                current_data['pm10'], 
                current_data['no2'], 
                current_data['o3'], 
                current_data['co'] * 1000,  # Convert to µg/m³
                current_data['so2']
            ]
            
            fig_bar = px.bar(
                x=pollutants, 
                y=values,
                title=f"Current Pollutant Levels - {selected_city}",
                labels={'x': 'Pollutants', 'y': 'Concentration (µg/m³)'},
                color=values,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # AQI Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = current_data['aqi'],
                title = {'text': f"AQI - {selected_city}"},
                gauge = {
                    'axis': {'range': [None, 300]},
                    'bar': {'color': aqi_color},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgreen"},
                        {'range': [50, 100], 'color': "yellow"},
                        {'range': [100, 150], 'color': "orange"},
                        {'range': [150, 300], 'color': "red"}
                    ]
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Health recommendations
            st.subheader("💡 Health Recommendations")
            aqi_value = current_data['aqi']
            
            if aqi_value <= 50:
                st.success("🟢 **Good Air Quality** - Great day for outdoor activities!")
            elif aqi_value <= 100:
                st.warning("🟡 **Moderate** - Sensitive individuals should consider limiting prolonged outdoor exertion.")
            elif aqi_value <= 150:
                st.warning("🟠 **Unhealthy for Sensitive Groups** - Reduce outdoor activities if you experience symptoms.")
            else:
                st.error("🔴 **Unhealthy** - Everyone should limit outdoor activities and wear masks.")
        
        else:
            st.info(f"👆 Click 'Fetch Live Data' to get real-time air quality for {selected_city}")
            st.write("### ✨ Features:")
            st.write("- 🌍 Real-time air quality data from Open-Meteo")
            st.write("- 📊 Live AQI calculations")
            st.write("- 🏥 Health recommendations")
            st.write("- 📈 Interactive charts")
    
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.write("Please check the terminal for detailed error information.")
    
    # Footer
    st.markdown("---")
    st.markdown("🌟 **AirAware** - Real Air Quality Monitoring")

if __name__ == "__main__":
    main()