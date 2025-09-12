import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import uuid
from models.crop_recommender import CropRecommender
from models.price_predictor import PricePredictor
from utils.visualizations import create_crop_probability_chart, create_price_trend_chart, create_regional_price_comparison
from utils.recommendations import RecommendationEngine
from data.data_generator import generate_sample_data
from database.connection import get_database
from pages.farmer_profile import show_farmer_profile

# Page configuration
st.set_page_config(
    page_title="Smart Crop & Market Price Recommender",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'crop_model' not in st.session_state:
    st.session_state.crop_model = None
if 'price_model' not in st.session_state:
    st.session_state.price_model = None
if 'recommendation_engine' not in st.session_state:
    st.session_state.recommendation_engine = None

@st.cache_data
def load_and_train_models():
    """Load data and train models with caching"""
    try:
        # Generate sample data
        crop_data, price_data = generate_sample_data()
        
        # Initialize and train crop recommender
        crop_model = CropRecommender()
        crop_model.train(crop_data)
        
        # Initialize and train price predictor
        price_model = PricePredictor()
        price_model.train(price_data)
        
        # Initialize recommendation engine
        rec_engine = RecommendationEngine(crop_model, price_model)
        
        return crop_model, price_model, rec_engine, crop_data, price_data
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None

def main():
    st.title("🌾 Smart Crop & Market Price Recommender")
    st.markdown("### AI-powered farming decisions for maximum profitability")
    
    # Navigation
    page = st.sidebar.selectbox("Navigate", ["🏠 Home", "👨‍🌾 Farmer Profile"])
    
    if page == "👨‍🌾 Farmer Profile":
        show_farmer_profile()
        return
    
    # Load models
    with st.spinner("Loading AI models..."):
        crop_model, price_model, rec_engine, crop_data, price_data = load_and_train_models()
    
    if crop_model is None:
        st.error("Failed to load models. Please refresh the page.")
        return
    
    # Sidebar for input parameters
    st.sidebar.header("🌱 Farm Conditions")
    st.sidebar.markdown("Enter your soil and weather conditions:")
    
    # Soil parameters
    st.sidebar.subheader("Soil Nutrients")
    nitrogen = st.sidebar.slider("Nitrogen (N) - kg/ha", 0, 200, 90, help="Nitrogen content in soil")
    phosphorus = st.sidebar.slider("Phosphorus (P) - kg/ha", 5, 150, 42, help="Phosphorus content in soil")
    potassium = st.sidebar.slider("Potassium (K) - kg/ha", 5, 250, 43, help="Potassium content in soil")
    ph = st.sidebar.slider("Soil pH", 3.5, 10.0, 6.5, 0.1, help="Soil acidity/alkalinity level")
    
    # Weather parameters
    st.sidebar.subheader("Weather Conditions")
    temperature = st.sidebar.slider("Temperature (°C)", 8.0, 45.0, 25.0, 0.5, help="Average temperature")
    humidity = st.sidebar.slider("Humidity (%)", 14.0, 100.0, 80.0, 1.0, help="Relative humidity")
    rainfall = st.sidebar.slider("Rainfall (mm)", 20.0, 300.0, 150.0, 5.0, help="Annual rainfall")
    
    # Location selection
    st.sidebar.subheader("Location")
    states = ['Maharashtra', 'Punjab', 'Uttar Pradesh', 'Haryana', 'Gujarat', 'Rajasthan', 'West Bengal', 'Tamil Nadu']
    selected_state = st.sidebar.selectbox("Select your state:", states)
    
    # Prediction button
    if st.sidebar.button("🔍 Get Recommendations", type="primary"):
        if crop_model is None or price_model is None or rec_engine is None:
            st.error("Models not loaded properly. Please refresh the page.")
            return
            
        # Prepare input data
        input_features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        
        # Get crop recommendations
        with st.spinner("Analyzing soil and weather conditions..."):
            crop_predictions = crop_model.predict_with_probability(input_features)
            recommended_crop = crop_predictions[0]['crop']
        
        # Get price predictions and recommendations
        with st.spinner("Analyzing market data..."):
            recommendations = rec_engine.get_comprehensive_recommendation(
                input_features[0], selected_state
            )
            
            # Save recommendation to database if farmer is logged in
            if 'farmer_id' in st.session_state and st.session_state.farmer_id and recommendations:
                db = get_database()
                session_id = str(uuid.uuid4())
                
                recommendation_data = {
                    'farmer_id': st.session_state.farmer_id,
                    'session_id': session_id,
                    'recommended_crop': recommendations['recommended_crop'],
                    'suitability_score': crop_predictions[0]['suitability_score'],
                    'market_score': recommendations['crop_analysis'][0]['market_score'],
                    'combined_score': recommendations['crop_analysis'][0]['combined_score'],
                    'best_market': recommendations['market_recommendation']['best_market'],
                    'expected_price': recommendations['market_recommendation']['expected_price'],
                    'profit_margin': recommendations['market_recommendation']['profit_margin'],
                    'soil_n': nitrogen,
                    'soil_p': phosphorus,
                    'soil_k': potassium,
                    'soil_ph': ph,
                    'temperature': temperature,
                    'humidity': humidity,
                    'rainfall': rainfall,
                    'state': selected_state
                }
                
                rec_id = db.insert_recommendation(recommendation_data)
                if rec_id:
                    st.success("✅ Recommendation saved to your profile!")
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("🎯 Crop Recommendation")
            
            # Top recommendation card
            st.success(f"**Recommended Crop: {recommended_crop.title()}**")
            
            # Crop probability chart
            st.subheader("Crop Suitability Analysis")
            fig_crop = create_crop_probability_chart(crop_predictions[:5])
            st.plotly_chart(fig_crop, use_container_width=True)
            
            # Market recommendations
            st.header("💰 Market Analysis")
            
            if recommendations:
                market_rec = recommendations['market_recommendation']
                st.info(f"**Best Market: {market_rec['best_market']}**")
                st.metric("Expected Price", f"₹{market_rec['expected_price']:.2f}/kg", 
                         f"+{market_rec['profit_margin']:.1f}%")
                
                # Price trend chart
                st.subheader("Price Trend Analysis")
                if price_model is not None:
                    price_trend_data = price_model.get_price_trends(recommended_crop, selected_state)
                    fig_trend = create_price_trend_chart(price_trend_data, recommended_crop)
                    st.plotly_chart(fig_trend, use_container_width=True)
                
                # Regional price comparison
                st.subheader("Regional Price Comparison")
                if price_model is not None:
                    regional_data = price_model.get_regional_prices(recommended_crop)
                    fig_regional = create_regional_price_comparison(regional_data, recommended_crop)
                    st.plotly_chart(fig_regional, use_container_width=True)
        
        with col2:
            st.header("📊 Input Summary")
            
            # Display input parameters
            st.subheader("Soil Conditions")
            st.write(f"**Nitrogen:** {nitrogen} kg/ha")
            st.write(f"**Phosphorus:** {phosphorus} kg/ha")
            st.write(f"**Potassium:** {potassium} kg/ha")
            st.write(f"**pH Level:** {ph}")
            
            st.subheader("Weather Conditions")
            st.write(f"**Temperature:** {temperature}°C")
            st.write(f"**Humidity:** {humidity}%")
            st.write(f"**Rainfall:** {rainfall} mm")
            st.write(f"**Location:** {selected_state}")
            
            if recommendations:
                st.subheader("💡 Key Insights")
                insights = recommendations.get('insights', [])
                for insight in insights:
                    st.write(f"• {insight}")
                
                st.subheader("⚠️ Recommendations")
                if recommendations.get('recommendations'):
                    for rec in recommendations['recommendations']:
                        st.write(f"• {rec}")
    
    # Information tabs
    st.header("📚 Learn More")
    tab1, tab2, tab3 = st.tabs(["About the System", "How it Works", "Data Sources"])
    
    with tab1:
        st.markdown("""
        ### Smart Crop & Market Price Recommender
        
        This AI-powered system helps farmers make data-driven decisions by:
        
        - **Crop Recommendation**: Uses machine learning to suggest the most suitable crop based on soil nutrients and weather conditions
        - **Price Prediction**: Analyzes historical market data to predict optimal selling prices
        - **Market Analysis**: Recommends the best markets and timing for maximum profitability
        
        **Benefits:**
        - Increase crop yield through optimal crop selection
        - Maximize profits through strategic market decisions
        - Reduce farming risks through data-driven insights
        """)
    
    with tab2:
        st.markdown("""
        ### Machine Learning Models
        
        **1. Crop Recommendation Model (Random Forest Classifier)**
        - Features: N, P, K nutrients, pH, temperature, humidity, rainfall
        - Predicts: Most suitable crop for given conditions
        - Accuracy: Based on soil-crop compatibility patterns
        
        **2. Price Prediction Model (Ensemble Methods)**
        - Features: Historical prices, seasonal patterns, regional factors
        - Predicts: Expected market prices and trends
        - Analysis: Regional price variations and market opportunities
        
        **3. Recommendation Engine**
        - Combines crop suitability with market profitability
        - Provides comprehensive farming strategy
        - Includes risk assessment and timing recommendations
        """)
    
    with tab3:
        st.markdown("""
        ### Data Sources & Methodology
        
        **Agricultural Data:**
        - Soil nutrient databases
        - Weather pattern analysis
        - Crop yield historical data
        
        **Market Data:**
        - Historical mandi prices
        - Regional market variations
        - Seasonal price trends
        
        **Note:** This system uses representative agricultural data patterns. 
        For production use, integrate with live data sources like:
        - Data.gov.in agricultural datasets
        - Regional mandi price APIs
        - Weather service APIs
        """)

if __name__ == "__main__":
    main()
