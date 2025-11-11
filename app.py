import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import uuid
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
import joblib

# Page configuration
st.set_page_config(
    page_title="Smart Crop & Market Price Recommender",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Define the models directly in the main file to avoid import issues
class CropRecommender:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.scaler = StandardScaler()
        self.feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        self.crop_labels = []
        self.is_trained = False

    def generate_sample_data(self):
        """Generate sample crop data for training"""
        np.random.seed(42)
        n_samples = 1000

        crops = ['rice', 'wheat', 'cotton', 'maize', 'sugarcane', 'pulses', 'oilseeds', 'vegetables']

        data = []
        for _ in range(n_samples):
            crop = np.random.choice(crops)

            if crop == 'rice':
                N = np.random.randint(80, 120)
                P = np.random.randint(40, 60)
                K = np.random.randint(40, 60)
                temp = np.random.uniform(20, 35)
                humidity = np.random.uniform(70, 90)
                ph = np.random.uniform(5.5, 7.0)
                rainfall = np.random.uniform(150, 300)
            elif crop == 'wheat':
                N = np.random.randint(60, 100)
                P = np.random.randint(30, 50)
                K = np.random.randint(30, 50)
                temp = np.random.uniform(15, 25)
                humidity = np.random.uniform(50, 70)
                ph = np.random.uniform(6.0, 7.5)
                rainfall = np.random.uniform(50, 100)
            elif crop == 'cotton':
                N = np.random.randint(80, 150)
                P = np.random.randint(40, 80)
                K = np.random.randint(60, 100)
                temp = np.random.uniform(25, 35)
                humidity = np.random.uniform(60, 80)
                ph = np.random.uniform(6.0, 8.0)
                rainfall = np.random.uniform(75, 150)
            elif crop == 'maize':
                N = np.random.randint(100, 150)
                P = np.random.randint(50, 80)
                K = np.random.randint(50, 80)
                temp = np.random.uniform(20, 30)
                humidity = np.random.uniform(60, 80)
                ph = np.random.uniform(6.0, 7.5)
                rainfall = np.random.uniform(100, 200)
            else:
                N = np.random.randint(50, 120)
                P = np.random.randint(30, 70)
                K = np.random.randint(40, 80)
                temp = np.random.uniform(15, 35)
                humidity = np.random.uniform(50, 85)
                ph = np.random.uniform(5.5, 8.0)
                rainfall = np.random.uniform(50, 250)

            data.append([N, P, K, temp, humidity, ph, rainfall, crop])

        return pd.DataFrame(data, columns=self.feature_names + ['label'])

    def train(self, data=None):
        """Train the crop recommendation model"""
        try:
            if data is None:
                data = self.generate_sample_data()

            # Prepare features and target
            X = data[self.feature_names]
            y = data['label']

            self.crop_labels = y.unique().tolist()

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train the model
            self.model.fit(X_train_scaled, y_train)

            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)

            self.is_trained = True

            st.success(f"Model trained successfully with accuracy: {accuracy:.3f}")
            return accuracy

        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return None

    def predict(self, features):
        """Predict the best crop for given conditions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Make prediction
        prediction = self.model.predict(features_scaled)
        return prediction[0]

    def predict_with_probability(self, features):
        """Predict crop with probability scores for all crops"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Get probabilities
        probabilities = self.model.predict_proba(features_scaled)[0]

        # Create results with crop names and probabilities
        results = []
        for i, prob in enumerate(probabilities):
            results.append({
                'crop': self.model.classes_[i],
                'probability': prob,
                'suitability_score': prob * 100
            })

        # Sort by probability (descending)
        results.sort(key=lambda x: x['probability'], reverse=True)

        return results


class PricePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_trained = False

    def generate_sample_data(self):
        """Generate sample price data for training"""
        np.random.seed(42)
        n_samples = 2000

        commodities = ['rice', 'wheat', 'cotton', 'maize', 'sugarcane', 'pulses', 'oilseeds', 'vegetables']
        states = ['Maharashtra', 'Punjab', 'Uttar Pradesh', 'Haryana', 'Gujarat', 'Rajasthan', 'West Bengal',
                  'Tamil Nadu']
        markets = ['main_market', 'wholesale', 'export_hub']

        data = []
        for _ in range(n_samples):
            commodity = np.random.choice(commodities)
            state = np.random.choice(states)
            market = np.random.choice(markets)

            # Base prices for different commodities
            base_prices = {
                'rice': 45, 'wheat': 25, 'cotton': 65, 'maize': 20,
                'sugarcane': 35, 'pulses': 80, 'oilseeds': 60, 'vegetables': 40
            }

            base_price = base_prices[commodity]

            # Add variations based on state and market
            state_multipliers = {
                'Punjab': 1.1, 'Haryana': 1.05, 'Uttar Pradesh': 1.0,
                'Maharashtra': 0.95, 'Gujarat': 0.9, 'Rajasthan': 0.85,
                'West Bengal': 1.0, 'Tamil Nadu': 1.1
            }

            market_multipliers = {
                'export_hub': 1.2, 'main_market': 1.0, 'wholesale': 0.8
            }

            price = base_price * state_multipliers[state] * market_multipliers[market]
            price += np.random.normal(0, price * 0.1)  # Add some noise

            date = datetime.now() - timedelta(days=np.random.randint(0, 365))
            supply_volume = np.random.randint(500, 5000)
            demand_factor = np.random.uniform(0.8, 1.2)

            data.append({
                'commodity': commodity,
                'state': state,
                'market': market,
                'date': date,
                'price_per_kg': max(price, 10),  # Ensure minimum price
                'supply_volume': supply_volume,
                'demand_factor': demand_factor
            })

        return pd.DataFrame(data)

    def prepare_features(self, data):
        """Prepare features for price prediction"""
        df = data.copy()

        # Create time-based features
        df['month'] = pd.to_datetime(df['date']).dt.month
        df['quarter'] = pd.to_datetime(df['date']).dt.quarter
        df['is_peak_season'] = df['month'].isin([10, 11, 12, 1, 2])

        # Encode categorical variables
        categorical_cols = ['state', 'market', 'commodity']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])

        feature_cols = [
            'state_encoded', 'market_encoded', 'commodity_encoded',
            'month', 'quarter', 'is_peak_season',
            'supply_volume', 'demand_factor'
        ]

        return df[feature_cols]

    def train(self, data=None):
        """Train the price prediction model"""
        try:
            if data is None:
                data = self.generate_sample_data()

            # Prepare features
            X = self.prepare_features(data)
            y = data['price_per_kg']

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train the model
            self.model.fit(X_train_scaled, y_train)

            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            self.is_trained = True

            st.success(f"Price model trained - R²: {r2:.3f}, MAE: {mae:.3f}")
            return r2

        except Exception as e:
            st.error(f"Error training price model: {str(e)}")
            return None

    def predict_price(self, commodity, state, market):
        """Predict price for a specific commodity, state, and market"""
        if not self.is_trained:
            return 50.0  # Default fallback price

        try:
            # Create input data
            input_data = pd.DataFrame({
                'commodity': [commodity],
                'state': [state],
                'market': [market],
                'date': [datetime.now()],
                'supply_volume': [1000],
                'demand_factor': [1.0]
            })

            # Prepare features
            X = self.prepare_features(input_data)
            X_scaled = self.scaler.transform(X)

            # Make prediction
            predicted_price = self.model.predict(X_scaled)[0]
            return max(predicted_price, 10)  # Ensure positive price

        except:
            return 50.0  # Default fallback price


class RecommendationEngine:
    def __init__(self, crop_model, price_model):
        self.crop_model = crop_model
        self.price_model = price_model

    def get_comprehensive_recommendation(self, features, state):
        """Get comprehensive crop and market recommendations"""
        try:
            # Get crop predictions
            input_features = np.array([features])
            crop_predictions = self.crop_model.predict_with_probability(input_features)

            # Get market analysis for top crops
            crop_analysis = []
            for crop_pred in crop_predictions[:3]:
                crop = crop_pred['crop']
                suitability_score = crop_pred['suitability_score']

                # Get price prediction
                expected_price = self.price_model.predict_price(crop, state, f"{state}_main_market")

                # Calculate market score (normalized price)
                max_expected_price = 100  # Assuming max price
                market_score = (expected_price / max_expected_price) * 100

                # Combined score (weighted average)
                combined_score = (suitability_score * 0.6) + (market_score * 0.4)

                crop_analysis.append({
                    'crop': crop,
                    'suitability_score': suitability_score,
                    'expected_price': expected_price,
                    'market_score': market_score,
                    'combined_score': combined_score
                })

            # Sort by combined score
            crop_analysis.sort(key=lambda x: x['combined_score'], reverse=True)

            # Get market recommendation for top crop
            top_crop = crop_analysis[0]['crop']
            markets = [f"{state}_main_market", f"{state}_wholesale", f"{state}_export_hub"]

            market_prices = []
            for market in markets:
                price = self.price_model.predict_price(top_crop, state, market)
                market_prices.append({
                    'market': market,
                    'expected_price': price,
                    'market_type': market.split('_')[-1]
                })

            # Find best market
            market_prices.sort(key=lambda x: x['expected_price'], reverse=True)
            best_market = market_prices[0]

            # Calculate profit margin (simplified)
            base_cost = 20  # Assume base production cost
            profit_margin = ((best_market['expected_price'] - base_cost) / base_cost) * 100

            return {
                'recommended_crop': top_crop,
                'crop_analysis': crop_analysis,
                'market_recommendation': {
                    'best_market': best_market['market'],
                    'expected_price': best_market['expected_price'],
                    'profit_margin': profit_margin,
                    'all_markets': market_prices
                },
                'insights': [
                    f"{top_crop.title()} shows excellent compatibility with your soil conditions",
                    f"Market prices are favorable in {state} this season",
                    f"Expected profit margin: {profit_margin:.1f}%"
                ],
                'recommendations': [
                    "Consider soil testing for precise nutrient management",
                    "Monitor weather forecasts for optimal planting time",
                    "Check local government schemes for crop subsidies"
                ]
            }

        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
            return None


# Visualization functions
def create_crop_probability_chart(crop_predictions):
    """Create a bar chart showing crop probabilities"""
    crops = [pred['crop'].title() for pred in crop_predictions]
    probabilities = [pred['suitability_score'] for pred in crop_predictions]

    fig = px.bar(
        x=probabilities,
        y=crops,
        orientation='h',
        title='Crop Suitability Scores',
        labels={'x': 'Suitability Score (%)', 'y': 'Crops'},
        color=probabilities,
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        xaxis_range=[0, 100],
        showlegend=False,
        height=400
    )

    return fig


def create_price_trend_chart(trend_data, crop):
    """Create a line chart showing price trends"""
    fig = px.line(
        trend_data,
        x='date',
        y='price',
        title=f'{crop.title()} Price Trends (Last 30 Days)',
        labels={'date': 'Date', 'price': 'Price (₹/kg)'}
    )

    fig.update_layout(height=300)
    return fig


def create_regional_price_comparison(regional_data, crop):
    """Create a bar chart comparing regional prices"""
    fig = px.bar(
        regional_data,
        x='state',
        y='price',
        title=f'{crop.title()} - Regional Price Comparison',
        labels={'state': 'State', 'price': 'Price (₹/kg)'},
        color='price',
        color_continuous_scale='Blues'
    )

    fig.update_layout(height=300)
    return fig


# Initialize session state
if 'crop_model' not in st.session_state:
    st.session_state.crop_model = None
if 'price_model' not in st.session_state:
    st.session_state.price_model = None
if 'recommendation_engine' not in st.session_state:
    st.session_state.recommendation_engine = None


@st.cache_resource
def load_and_train_models():
    """Load data and train models with caching"""
    try:
        # Initialize models
        crop_model = CropRecommender()
        price_model = PricePredictor()

        # Train models
        crop_model.train()
        price_model.train()

        # Initialize recommendation engine
        rec_engine = RecommendationEngine(crop_model, price_model)

        return crop_model, price_model, rec_engine

    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None


def show_farmer_profile():
    """Simple farmer profile page"""
    st.header("👨‍🌾 Farmer Profile")

    if 'farmer_id' not in st.session_state:
        st.session_state.farmer_id = None

    if st.session_state.farmer_id is None:
        st.subheader("Login / Register")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**New Farmer?**")
            if st.button("Register"):
                st.session_state.farmer_id = str(uuid.uuid4())[:8]
                st.success("Registered successfully! Your Farmer ID: " + st.session_state.farmer_id)

        with col2:
            st.write("**Existing Farmer?**")
            farmer_id = st.text_input("Enter your Farmer ID")
            if st.button("Login"):
                if farmer_id:
                    st.session_state.farmer_id = farmer_id
                    st.success("Logged in successfully!")
                else:
                    st.error("Please enter your Farmer ID")

    if st.session_state.farmer_id:
        st.success(f"Logged in as Farmer ID: {st.session_state.farmer_id}")

        if st.button("Logout"):
            st.session_state.farmer_id = None
            st.rerun()


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
        crop_model, price_model, rec_engine = load_and_train_models()

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
                # Generate sample trend data
                dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
                base_price = market_rec['expected_price']
                trend_data = pd.DataFrame({
                    'date': dates,
                    'price': base_price + np.sin(np.arange(30) * 0.2) * (base_price * 0.1) + np.random.normal(0,
                                                                                                              base_price * 0.05,
                                                                                                              30)
                })
                fig_trend = create_price_trend_chart(trend_data, recommended_crop)
                st.plotly_chart(fig_trend, use_container_width=True)

                # Regional price comparison
                st.subheader("Regional Price Comparison")
                regional_data = pd.DataFrame([
                    {'state': state,
                     'price': price_model.predict_price(recommended_crop, state, f"{state}_main_market")}
                    for state in states
                ])
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