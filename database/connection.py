import os
import psycopg2
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, List, Any
import streamlit as st

class DatabaseConnection:
    def __init__(self):
        self.connection_url = os.getenv('DATABASE_URL')
        self.connection = None
    
    def get_connection(self):
        """Get database connection with error handling"""
        try:
            if self.connection is None or self.connection.closed:
                self.connection = psycopg2.connect(self.connection_url)
            return self.connection
        except Exception as e:
            st.error(f"Database connection failed: {str(e)}")
            return None
    
    def execute_query(self, query: str, params: Optional[tuple] = None, fetch: bool = True):
        """Execute SQL query with error handling"""
        try:
            conn = self.get_connection()
            if conn is None:
                return None
                
            with conn.cursor() as cursor:
                cursor.execute(query, params if params is not None else ())
                
                if fetch and cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    return pd.DataFrame(data=rows, columns=columns)
                else:
                    conn.commit()
                    return cursor.rowcount
        except Exception as e:
            st.error(f"Database query failed: {str(e)}")
            if self.connection:
                self.connection.rollback()
            return None
    
    def insert_farmer(self, farmer_data: Dict[str, Any]) -> Optional[int]:
        """Insert new farmer profile and return farmer ID"""
        query = """
        INSERT INTO farmers (email, name, phone, location, state, farm_size_acres, latitude, longitude)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """
        params = (
            farmer_data.get('email'),
            farmer_data.get('name'),
            farmer_data.get('phone'),
            farmer_data.get('location'),
            farmer_data.get('state'),
            farmer_data.get('farm_size_acres'),
            farmer_data.get('latitude'),
            farmer_data.get('longitude')
        )
        
        try:
            conn = self.get_connection()
            if conn is None:
                return None
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                result = cursor.fetchone()
                if result:
                    farmer_id = result[0]
                    conn.commit()
                    return farmer_id
                return None
        except Exception as e:
            st.error(f"Failed to insert farmer: {str(e)}")
            if self.connection:
                self.connection.rollback()
            return None
    
    def get_farmer_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get farmer profile by email"""
        query = "SELECT * FROM farmers WHERE email = %s"
        result = self.execute_query(query, (email,))
        
        if isinstance(result, pd.DataFrame) and not result.empty:
            return result.iloc[0].to_dict()
        return None
    
    def insert_recommendation(self, recommendation_data: Dict[str, Any]) -> Optional[int]:
        """Insert recommendation record and return recommendation ID"""
        query = """
        INSERT INTO recommendations (
            farmer_id, session_id, recommended_crop, suitability_score, market_score,
            combined_score, best_market, expected_price, profit_margin,
            soil_n, soil_p, soil_k, soil_ph, temperature, humidity, rainfall, state
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """
        params = (
            recommendation_data.get('farmer_id'),
            recommendation_data.get('session_id'),
            recommendation_data.get('recommended_crop'),
            recommendation_data.get('suitability_score'),
            recommendation_data.get('market_score'),
            recommendation_data.get('combined_score'),
            recommendation_data.get('best_market'),
            recommendation_data.get('expected_price'),
            recommendation_data.get('profit_margin'),
            recommendation_data.get('soil_n'),
            recommendation_data.get('soil_p'),
            recommendation_data.get('soil_k'),
            recommendation_data.get('soil_ph'),
            recommendation_data.get('temperature'),
            recommendation_data.get('humidity'),
            recommendation_data.get('rainfall'),
            recommendation_data.get('state')
        )
        
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                rec_id = cursor.fetchone()[0]
                conn.commit()
                return rec_id
        except Exception as e:
            st.error(f"Failed to insert recommendation: {str(e)}")
            if self.connection:
                self.connection.rollback()
            return None
    
    def get_farmer_recommendations(self, farmer_id: int, limit: int = 10) -> Optional[pd.DataFrame]:
        """Get farmer's recommendation history"""
        query = """
        SELECT r.*, f.name as farmer_name 
        FROM recommendations r 
        JOIN farmers f ON r.farmer_id = f.id 
        WHERE r.farmer_id = %s 
        ORDER BY r.created_at DESC 
        LIMIT %s
        """
        result = self.execute_query(query, (farmer_id, limit))
        return result if isinstance(result, pd.DataFrame) else None
    
    def insert_yield_data(self, yield_data: Dict[str, Any]) -> Optional[int]:
        """Insert yield tracking data"""
        query = """
        INSERT INTO yield_data (
            farmer_id, recommendation_id, crop_planted, planting_date, harvest_date,
            actual_yield_kg_per_acre, actual_price_per_kg, total_revenue, total_costs,
            net_profit, satisfaction_rating, notes
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """
        params = (
            yield_data.get('farmer_id'),
            yield_data.get('recommendation_id'),
            yield_data.get('crop_planted'),
            yield_data.get('planting_date'),
            yield_data.get('harvest_date'),
            yield_data.get('actual_yield_kg_per_acre'),
            yield_data.get('actual_price_per_kg'),
            yield_data.get('total_revenue'),
            yield_data.get('total_costs'),
            yield_data.get('net_profit'),
            yield_data.get('satisfaction_rating'),
            yield_data.get('notes')
        )
        
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                yield_id = cursor.fetchone()[0]
                conn.commit()
                return yield_id
        except Exception as e:
            st.error(f"Failed to insert yield data: {str(e)}")
            if self.connection:
                self.connection.rollback()
            return None
    
    def get_farmer_yield_history(self, farmer_id: int) -> pd.DataFrame:
        """Get farmer's yield history with performance analysis"""
        query = """
        SELECT 
            yd.*,
            r.recommended_crop,
            r.expected_price as predicted_price,
            r.profit_margin as predicted_profit_margin,
            (yd.actual_price_per_kg - r.expected_price) as price_difference,
            CASE 
                WHEN yd.actual_price_per_kg > r.expected_price THEN 'Better than predicted'
                WHEN yd.actual_price_per_kg < r.expected_price THEN 'Below prediction'
                ELSE 'As predicted'
            END as price_performance
        FROM yield_data yd
        JOIN recommendations r ON yd.recommendation_id = r.id
        WHERE yd.farmer_id = %s
        ORDER BY yd.harvest_date DESC
        """
        return self.execute_query(query, (farmer_id,))
    
    def cache_weather_data(self, weather_data: Dict[str, Any]) -> bool:
        """Cache weather data from API"""
        query = """
        INSERT INTO weather_cache (latitude, longitude, date, temperature, humidity, rainfall, wind_speed, pressure, api_source)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        """
        params = (
            weather_data.get('latitude'),
            weather_data.get('longitude'),
            weather_data.get('date'),
            weather_data.get('temperature'),
            weather_data.get('humidity'),
            weather_data.get('rainfall'),
            weather_data.get('wind_speed'),
            weather_data.get('pressure'),
            weather_data.get('api_source', 'openweather')
        )
        
        result = self.execute_query(query, params, fetch=False)
        return result is not None
    
    def get_cached_weather(self, latitude: float, longitude: float, date: str) -> Optional[Dict[str, Any]]:
        """Get cached weather data"""
        query = """
        SELECT * FROM weather_cache 
        WHERE latitude = %s AND longitude = %s AND date = %s
        ORDER BY cached_at DESC LIMIT 1
        """
        result = self.execute_query(query, (latitude, longitude, date))
        
        if isinstance(result, pd.DataFrame) and not result.empty:
            return result.iloc[0].to_dict()
        return None
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()

# Global database instance
@st.cache_resource
def get_database():
    """Get cached database connection"""
    return DatabaseConnection()