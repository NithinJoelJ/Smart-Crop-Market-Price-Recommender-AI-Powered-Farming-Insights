# Overview

This is a Smart Crop & Market Price Recommender application built with Streamlit that helps farmers make informed decisions about crop selection and market timing. The system combines machine learning models for crop recommendation based on soil and environmental conditions with price prediction capabilities to provide comprehensive farming guidance. It features an interactive web interface with data visualizations and generates synthetic agricultural data for training and demonstration purposes.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Streamlit Web Application**: Single-page application using Streamlit framework for interactive UI
- **Plotly Visualizations**: Interactive charts and graphs for crop probability analysis, price trends, and regional comparisons
- **Session State Management**: Caching of trained models and user data across application sessions
- **Responsive Layout**: Wide layout configuration with expandable sidebar for better user experience

## Backend Architecture
- **Model-Based Prediction System**: Separate modules for crop recommendation and price prediction
- **Machine Learning Pipeline**: 
  - Crop Recommender using Random Forest Classifier with feature scaling
  - Price Predictor using ensemble methods (Random Forest, Gradient Boosting, Linear Regression)
- **Recommendation Engine**: Combines crop suitability and market analysis for comprehensive recommendations
- **Data Generation Module**: Synthetic data generator creating realistic agricultural datasets for training

## Data Processing
- **Feature Engineering**: Automated extraction of time-based features, categorical encoding, and agricultural season indicators
- **Data Preprocessing**: StandardScaler for numerical features, LabelEncoder for categorical variables
- **Model Training Pipeline**: Train-test split with stratification, cross-validation, and performance evaluation

## Core Models
- **Crop Recommendation Model**: 
  - Input: Soil nutrients (N, P, K), environmental conditions (temperature, humidity, pH, rainfall)
  - Output: Probability scores for different crop types with suitability rankings
- **Price Prediction Model**: 
  - Input: Commodity type, location data, seasonal factors, supply-demand indicators
  - Output: Expected market prices and optimal selling strategies

# External Dependencies

## Machine Learning Libraries
- **scikit-learn**: Core machine learning algorithms (Random Forest, Gradient Boosting, preprocessing)
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing and array operations

## Visualization and UI
- **Streamlit**: Web application framework for creating interactive dashboards
- **Plotly**: Interactive plotting library for charts and visualizations (plotly.express, plotly.graph_objects)

## Model Persistence
- **joblib**: Model serialization and loading for trained machine learning models

## Data Generation
- **datetime**: Time-based feature engineering and synthetic data generation
- **random**: Synthetic data generation with realistic agricultural patterns

Note: The application currently uses synthetic data generation for demonstration purposes. In a production environment, this could be replaced with real agricultural databases, weather APIs, and market data feeds.