import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
class InventoryPredictor:
    def __init__(self, firebase_cred_path: str):
        """Initialize Firebase connection and ML models"""
        # Initialize Firebase
        cred = credentials.Certificate(firebase_cred_path)
        firebase_admin.initialize_app(cred)
        self.db = firestore.client()
        
        # Initialize model paths
        self.quantity_model_path = 'models/quantity_predictor.joblib'
        self.time_series_model_path = 'models/time_series_model.joblib'
        
        # Load or train models
        self.quantity_model = self.load_quantity_model()
        self.time_series_models = {}
        self.scalers = {}
        print("done itializing")
    def fetch_historical_data(self) -> pd.DataFrame:
        """Fetch historical sales and inventory data from Firebase"""
        sales_ref = self.db.collection('sales_history')
        inventory_ref = self.db.collection('inventory_changes')
        
        # Fetch sales data
        sales_data = []
        for doc in sales_ref.stream():
            sales_data.append(doc.to_dict())
        
        # Fetch inventory changes
        inventory_data = []
        for doc in inventory_ref.stream():
            inventory_data.append(doc.to_dict())
        
        # Combine and process data
        sales_df = pd.DataFrame(sales_data)
        inventory_df = pd.DataFrame(inventory_data)
        print("done fetching")
        return self.process_historical_data(sales_df, inventory_df)

    def     process_historical_data(self, sales_df: pd.DataFrame, 
                              inventory_df: pd.DataFrame) -> pd.DataFrame:
        """Process and combine historical data"""
        # Convert timestamps
        sales_df['timestamp'] = pd.to_datetime(sales_df['timestamp'])
        inventory_df['timestamp'] = pd.to_datetime(inventory_df['timestamp'])
        
        # Aggregate daily data
        daily_sales = sales_df.groupby(['item_id', 
                                      sales_df['timestamp'].dt.date]).agg({
            'quantity': 'sum',
            'price': 'mean'
        }).reset_index()
        
        # Add seasonal features
        daily_sales['day_of_week'] = pd.to_datetime(daily_sales['timestamp']).dt.dayofweek
        daily_sales['month'] = pd.to_datetime(daily_sales['timestamp']).dt.month
        daily_sales['year'] = pd.to_datetime(daily_sales['timestamp']).dt.year
        print("done processing historical shit")
        
        return daily_sales

    def prepare_features(self, data: pd.DataFrame) -> tuple:
        """Prepare features for ML model"""
        features = ['day_of_week', 'month', 'year', 'price']
        X = data[features]
        y = data['quantity']
        
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_quantity_model(self, data: pd.DataFrame):
        """Train Random Forest model for quantity prediction"""
        X_train, X_test, y_train, y_test = self.prepare_features(data)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Save model and scaler
        joblib.dump(model, self.quantity_model_path)
        joblib.dump(scaler, 'models/scaler.joblib')
        print("done training stupid quantity model")
        
        return model, scaler

    def train_time_series_model(self, data: pd.DataFrame, item_id: str):
        """Train Prophet model for time series prediction"""
        # Prepare data for Prophet
        prophet_df = data[data['item_id'] == item_id].copy()
        print( item_id)
        prophet_df = prophet_df[['timestamp', 'quantity']]
        prophet_df.columns = ['ds', 'y']
        # Train model
        model = Prophet(yearly_seasonality=True, 
                       weekly_seasonality=True,
                       daily_seasonality=False)
        model.fit(prophet_df)
        
        # Save model
        # TODO SAVING THE FUCKING TIME MODEL 
        # model.(f'models/prophet_{item_id}.json')
        print("done training time shit")
        
        return model

    def load_quantity_model(self):
        """Load or train quantity prediction model"""
        try:
            model = joblib.load(self.quantity_model_path)
            print("done loading shit")
            return model
        except:
            data = self.fetch_historical_data()
            model, _ = self.train_quantity_model(data)
            print("done taining all shit shit")
            return model

    def predict_quantity(self, item_id: str, future_date: datetime) -> dict:
        """Predict required quantity for a specific date"""
        # Get item data
        item_data = self.db.collection('items').document(item_id).get().to_dict()
        
        # Prepare features for prediction
        features = pd.DataFrame({
            'day_of_week': [future_date.weekday()],
            'month': [future_date.month],
            'year': [future_date.year],
            'price': [item_data['price']]
        })
        print(f'fucking ${features}')
        # Scale features
        scaler = joblib.load('models/scaler.joblib')
        features_scaled = scaler.transform(features)
        
        # Make prediction
        predicted_quantity = self.quantity_model.predict(features_scaled)[0]
        print(f"done predicting shit ${item_id, predicted_quantity}")
        return {
            'item_id': item_id,
            'predicted_quantity': round(predicted_quantity),
            'prediction_date': future_date.strftime('%Y-%m-%d')
        }

    def predict_optimal_purchase_time(self, item_id: str, 
                                    forecast_days: int = 30) -> dict:
        """Predict optimal purchase time based on time series analysis"""
        # Load or train time series model
        if item_id not in self.time_series_models:
            data = self.fetch_historical_data()
            self.time_series_models[item_id] = self.train_time_series_model(
                data, item_id
            )
        
        model = self.time_series_models[item_id]
        
        # Create future dates dataframe
        future_dates = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future_dates)
        
        # Get current inventory
        current_inventory = self.db.collection('items').document(item_id).get().to_dict()['quantity']
        
        # Find when inventory will hit minimum threshold
        threshold = current_inventory * 0.2  # 20% of current inventory
        forecast['inventory'] = current_inventory - forecast['yhat'].cumsum()
        purchase_date = forecast[forecast['inventory'] <= threshold]['ds'].iloc[0]
        
        return {
            'item_id': item_id,
            'suggested_purchase_date': purchase_date.strftime('%Y-%m-%d'),
            'forecasted_inventory': float(forecast[forecast['ds'] == purchase_date]['inventory'].iloc[0]),
            'confidence_interval': {
                'lower': float(forecast[forecast['ds'] == purchase_date]['yhat_lower'].iloc[0]),
                'upper': float(forecast[forecast['ds'] == purchase_date]['yhat_upper'].iloc[0])
            }
        }

    def save_predictions_to_firebase(self, predictions: dict):
        """Save predictions to Firebase"""
        predictions_ref = self.db.collection('inventory_predictions')
        predictions['timestamp'] = datetime.now()
        predictions_ref.add(predictions)

class InventoryOptimizer:
    def __init__(self, predictor: InventoryPredictor):
        self.predictor = predictor

    def generate_purchase_recommendations(self) -> list:
        """Generate purchase recommendations for all items"""
        items_ref = self.predictor.db.collection('items')
        recommendations = []
        for item in items_ref.stream():
            item_data = item.to_dict()
            item_id = item.id
            print(f"dealing with the first crap {item_id}")
            # Get quantity prediction
            next_month = datetime.now() + timedelta(days=30)
            quantity_prediction = self.predictor.predict_quantity(
                item_id, next_month
            )
            
            # Get timing prediction
            timing_prediction = self.predictor.predict_optimal_purchase_time(
                item_id
            )
            
            recommendation = {
                'item_id': item_id,
                'item_name': item_data['name'],
                'current_quantity': item_data['quantity'],
                'recommended_purchase_quantity': quantity_prediction['predicted_quantity'],
                'recommended_purchase_date': timing_prediction['suggested_purchase_date'],
                'confidence_interval': timing_prediction['confidence_interval']
            }
            
            recommendations.append(recommendation)
            
            # Save to Firebase
            self.predictor.save_predictions_to_firebase(recommendation)
        if not recommendations: 
            print('we ain\'t got no shit') 
        else: 
            print('we got some shit ')
        return recommendations

def main():
    # Initialize predictor with Firebase credentials
    predictor = InventoryPredictor('enventory-optimizer-firebase-adminsdk-v1unb-7e807e4c45.json')
    optimizer = InventoryOptimizer(predictor)
    
    # Generate and print recommendations
    recommendations = optimizer.generate_purchase_recommendations()
    
    # Print recommendations
    for rec in recommendations:
        print("done recomending shit")
        print(f"\nRecommendations for {rec['item_name']}:")
        print(f"Current Quantity: {rec['current_quantity']}")
        print(f"Recommended Purchase: {rec['recommended_purchase_quantity']} units")
        print(f"Recommended Purchase Date: {rec['recommended_purchase_date']}")
        print(f"Confidence Interval: {rec['confidence_interval']}")

if __name__ == "__main__":
    main()
