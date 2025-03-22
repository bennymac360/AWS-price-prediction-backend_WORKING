# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:33:44 2024

@author: MCNB
"""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
from datetime import datetime
import requests
import certifi
import json
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import boto3
import tempfile
from datetime import datetime, timezone  # Add timezone to the import statement






def main(coin_id):
    # --- Part 1: Fetch the latest data for the selected coin ---

    def fetch_crypto_data(coin_id='bitcoin', vs_currency='usd', days='365', interval='daily'):
        API_KEY = "CG-ZZLFUQooopRkr47Z1yqNWKyP"
        headers = {
            "accept": "application/json",
            "x-cg-demo-api-key": API_KEY
        }
        url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
        params = {
            'vs_currency': vs_currency,
            'days': days,
            'interval': interval
        }
        response = requests.get(url, params=params, headers=headers, verify=certifi.where())
        if response.status_code == 200:
            historical_data = response.json()
            # Process data into a DataFrame
            prices = historical_data['prices']
            market_caps = historical_data['market_caps']
            total_volumes = historical_data['total_volumes']
            data = []
            for i in range(len(prices) - 1):
                timestamp = prices[i][0]
                open_price = prices[i][1]
                close_price = prices[i + 1][1]
                market_cap = market_caps[i][1]
                volume = total_volumes[i][1]
                date = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).strftime('%Y-%m-%d')
                data.append({
                    'date': date,
                    'open': open_price,
                    'close': close_price,
                    'market_cap': market_cap,
                    'volume': volume
                })
            df = pd.DataFrame(data)
            # Set 'date' as index
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df
        else:
            print(f"Error fetching data for {coin_id}: {response.status_code}, {response.text}")
            return None

    # Fetch the latest data
    sequence_length = 60  # Same as used in training
    max_window_size = 100  # Largest window size used in MA and RSI calculations
    days_needed = sequence_length + max_window_size + 100  # Adjust as needed
    df = fetch_crypto_data(coin_id=coin_id, days=str(days_needed), interval='daily')
    if df is None:
        return

    # --- Part 2: Preprocess the data ---

    def preprocess_data(data):
        """Fill missing values in the data."""
        data['close'] = data['close'].ffill()
        return data

    def calculate_rsi(prices, window):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def add_features(data, MA_windows=[7, 10, 14], RSI_windows=[14]):
        """Add moving averages, RSI, and other features to the dataset."""
        # Add Moving Averages (MA)
        for window in MA_windows:
            data[f'MA_{window}'] = data['close'].rolling(window=window).mean()

        # Add Relative Strength Index (RSI)
        for window in RSI_windows:
            data[f'RSI_{window}'] = calculate_rsi(data['close'], window)

        # Add delta_close_tomorrowClose (Difference between tomorrow's close and today's close)
        data["Tomorrow"] = data["close"].shift(-1)
        data["delta_close_tomorrowClose"] = data["Tomorrow"] - data["close"]

        # Add the ratio of delta_close_tomorrowClose and today's close
        data["delta_close_ratio"] = data["delta_close_tomorrowClose"] / data["close"]

        # Handle NaN values
        data.ffill(inplace=True)
        data.bfill(inplace=True)
        data.dropna(inplace=True)

        return data

    df = preprocess_data(df)
    df = add_features(df)

    # --- Part 3: Load the saved models and feature importances ---

    def load_individual_models(directory_path):
        models = {}
        for filename in os.listdir(directory_path):
            if filename.startswith('model_') and filename.endswith('.keras'):
                feature = filename[len('model_'):-len('.keras')]
                model_path = os.path.join(directory_path, filename)
                model = load_model(model_path)
                models[feature] = model
        return models

    def load_feature_importances(filepath):
        with open(filepath, 'r') as f:
            feature_importances = json.load(f)
        return feature_importances

    # Paths to the saved models and feature importances
    models_directory = './models/individual_models'
    feature_importances_filepath = './models/feature_importances.json'

    # Load individual models
    individual_models = load_individual_models(models_directory)

    # Load feature importances
    feature_importances = load_feature_importances(feature_importances_filepath)

    # --- Part 4: Prepare the data for prediction ---

    # Define the features and target as used in training
    features = ['close', 'MA_7', 'MA_10', 'MA_14', 'delta_close_ratio']
    target = 'Tomorrow'

    # Ensure all features are present
    missing_features = [feature for feature in features if feature not in df.columns]
    if missing_features:
        print(f"Missing features in data: {missing_features}")
        return

    # Scale the features and target
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    feature_scaler.fit(df[features])
    target_scaler.fit(df[[target]])

    scaled_features = feature_scaler.transform(df[features])
    scaled_target = target_scaler.transform(df[[target]])

    # Create sequences
    X_sequences = []
    for i in range(len(scaled_features) - sequence_length):
        X_sequences.append(scaled_features[i:i + sequence_length])
    X_sequences = np.array(X_sequences)

    # --- Part 5: Make predictions ---

    weighted_predictions = np.zeros((X_sequences.shape[0], 1))

    for feature in features:
        feature_index = features.index(feature)
        X_feature_sequences = X_sequences[:, :, feature_index].reshape(-1, sequence_length, 1)
        model = individual_models.get(feature)
        if model:
            predictions = model.predict(X_feature_sequences, batch_size=16)
            weight = feature_importances.get(feature, 0)
            weighted_predictions += weight * predictions
        else:
            print(f"No model found for feature {feature}.")

    # Inverse transform the predictions
    weighted_predictions = weighted_predictions.reshape(-1, 1)
    predictions_inverse = target_scaler.inverse_transform(weighted_predictions)

    # --- Part 6: Output the predictions ---

    # Get the last predicted value
    last_prediction = predictions_inverse[-1][0]
    #print(f"Predicted {target} price for {coin_id} is: {last_prediction:.4f}")
    
    return last_prediction


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python scriptname.py <coin_id>")
    else:
        coin_id = sys.argv[1]
        predicted_price = main(coin_id)
        if predicted_price is not None:
            print(f"Predicted price for {coin_id} is: {predicted_price:.4f}")
        else:
            print("Prediction could not be made due to an error in data fetching or processing.")
