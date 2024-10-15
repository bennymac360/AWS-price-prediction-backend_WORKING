# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:33:44 2024

Author: MCNB
"""

import os
import sys  # Moved import to top

# 1. Suppress TensorFlow INFO and WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING
# If you want to suppress all logs except ERROR, set to '3'

# Existing environment variable
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from numpy import array, zeros
from pandas import DataFrame, to_datetime
from datetime import datetime, timezone
from requests import get
from certifi import where
from json import loads, dumps, JSONDecodeError
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from boto3 import client
from tempfile import NamedTemporaryFile
from botocore.exceptions import NoCredentialsError, ClientError

def main(coin_id, useModel=0, verbose=False):
    """
    Main function to predict cryptocurrency price based on selected model.
    
    Parameters:
        coin_id (str): The ID of the cryptocurrency (e.g., 'bitcoin').
        useModel (int): The model selection (0 for Baseline, 1 for Multi-Model).
        verbose (bool): If True, prints detailed logs.
    
    Returns:
        float: The predicted price.
    """
    
    def load_api_key_from_s3(bucket_name, key, verbose=verbose):
        """
        Load the API key from a JSON file stored in an S3 bucket.
        
        Parameters:
            bucket_name (str): The name of the S3 bucket.
            key (str): The key (path) of the config.json file in the S3 bucket.
            verbose (bool): If True, prints success messages.
        
        Returns:
            str: The API key if successfully loaded, None otherwise.
        """
        s3_client = client('s3')
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=key)
            content = response['Body'].read().decode('utf-8')
            config_data = loads(content)
            api_key = config_data.get('api_key')
            if verbose:
                print(f"API key successfully loaded from {key}")
            return api_key
        except NoCredentialsError:
            print("Error: AWS credentials not available.")
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                print(f"Error: The object '{key}' does not exist in bucket '{bucket_name}'.")
            else:
                print(f"An error occurred: {e}")
        except JSONDecodeError:
            print(f"Error: Failed to decode JSON from '{key}'.")
        return None
    
    
    # --- Part 1: Fetch the latest data for the selected coin ---

    def fetch_crypto_data(coin_id='bitcoin', vs_currency='usd', days='365', interval='daily'):
        # Load the API key from the S3 bucket
        bucket_name = 'pricepredictor-bucket'
        config_key = 'config.json'
        API_KEY = load_api_key_from_s3(bucket_name, config_key, verbose=verbose)
        
        if API_KEY is None:
            print("Error: API key could not be loaded.")
            return None
        
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
        response = get(url, params=params, headers=headers, verify=where())
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
            df = DataFrame(data)
            # Set 'date' as index
            df['date'] = to_datetime(df['date'])
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

        # Add Exponential Moving Averages (EMA)
        for window in MA_windows:
            data[f'EMA_{window}'] = data['close'].ewm(span=window, adjust=False).mean()

        # Add Relative Strength Index (RSI)
        for window in RSI_windows:
            data[f'RSI_{window}'] = calculate_rsi(data['close'], window)

        # Add Lagged Features
        for lag in range(1, 6):
            data[f'Close_Lag_{lag}'] = data['close'].shift(lag)

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

    # --- Part 3: Load the saved models and feature importances from S3 ---

    # Initialize the S3 client
    s3_client = client('s3')

    # Define the S3 bucket and object prefixes/keys
    bucket_name = 'pricepredictor-bucket'
    models_prefix = ''  # S3 prefix where models are stored
    feature_importances_key = 'feature_importances.json'  # S3 key for feature importances

    def load_json_from_s3(bucket, key, verbose=verbose):
        """Load JSON data from S3."""
        try:
            response = s3_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read().decode('utf-8')
            data = loads(content)
            if verbose:
                print(f"Successfully loaded '{key}' from bucket '{bucket}'.")
            return data
        except NoCredentialsError:
            print("Error: AWS credentials not available.")
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                print(f"Error: The object '{key}' does not exist in bucket '{bucket}'.")
            else:
                print(f"An error occurred: {e}")
        except JSONDecodeError:
            print(f"Error: Failed to decode JSON from '{key}'.")
        return None


    def load_baseline_model(bucket_name, models_prefix, verbose=verbose):
        """
        Load the Baseline Keras model from S3.

        Parameters:
            bucket_name (str): Name of the S3 bucket.
            models_prefix (str): S3 prefix where models are stored.
            verbose (bool): If True, prints success messages.

        Returns:
            keras.Model or None: The loaded Baseline model, or None if not found.
        """
        baseline_feature = 'baseline'
        model_filename = f'model_{baseline_feature}.keras'
        model_key = os.path.join(models_prefix, model_filename)
        
        try:
            with NamedTemporaryFile(delete=False, suffix='.keras') as tmp_file:
                s3_client.download_fileobj(bucket_name, model_key, tmp_file)
                tmp_file_path = tmp_file.name
                if verbose:
                    print(f"Downloaded '{model_key}' to temporary file '{tmp_file_path}'.")

            # Load the Keras model from the temporary file
            model = load_model(tmp_file_path)
            if verbose:
                print(f"Loaded Baseline model '{model_filename}' successfully.")
            
            # Optionally, delete the temporary file after loading
            os.remove(tmp_file_path)

            return model
        except NoCredentialsError:
            print("Error: AWS credentials not available.")
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                print(f"Error: The object '{model_key}' does not exist in bucket '{bucket_name}'.")
            else:
                print(f"An error occurred: {e}")
        except Exception as e:
            print(f"Error loading Baseline model '{model_filename}': {e}")
        
        return None


    def load_individual_models_from_s3(bucket, prefix, required_features, verbose=verbose):
        """
        Load individual Keras models from S3 corresponding to the required features.

        Parameters:
            bucket (str): Name of the S3 bucket.
            prefix (str): S3 prefix where models are stored.
            required_features (list): List of feature names whose models need to be loaded.

        Returns:
            dict: A dictionary mapping feature names to their loaded Keras models.
        """
        models = {}
        try:
            paginator = s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)

            # Keep track of which features have been loaded
            loaded_features = set()

            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        filename = os.path.basename(key)
                        if filename.endswith('.keras') and filename.startswith('model_'):
                            feature = filename[len('model_'):-len('.keras')]
                            if feature in required_features and feature not in loaded_features:
                                if verbose:
                                    print(f"Found model for feature: {feature} (Key: {key})")
                                try:
                                    with NamedTemporaryFile(delete=False, suffix='.keras') as tmp_file:
                                        s3_client.download_fileobj(bucket, key, tmp_file)
                                        tmp_file_path = tmp_file.name
                                        if verbose:
                                            print(f"Downloaded '{key}' to temporary file '{tmp_file_path}'.")

                                    # Load the Keras model from the temporary file
                                    model = load_model(tmp_file_path)
                                    if verbose:
                                        print(f"Loaded model for feature '{feature}' successfully.")
                                    models[feature] = model

                                    # Optionally, delete the temporary file after loading
                                    os.remove(tmp_file_path)

                                    # Mark the feature as loaded
                                    loaded_features.add(feature)

                                    # If all required features are loaded, exit early
                                    if len(loaded_features) == len(required_features):
                                        if verbose:
                                            print("All required models have been loaded.")
                                        return models

                                except Exception as e:
                                    print(f"Error loading model '{key}': {e}")
                else:
                    print(f"No objects found with prefix '{prefix}' in bucket '{bucket}'.")

            # Check if any required features were not loaded
            missing_features = set(required_features) - loaded_features
            if missing_features:
                print(f"Warning: Models for the following features were not found in S3: {missing_features}")
            return models
        except NoCredentialsError:
            print("Error: AWS credentials not available.")
        except ClientError as e:
            print(f"An error occurred while listing objects in S3: {e}")
        return models

    # Define the features and target as used in training
    features = [
    # Moving Averages
    'MA_7', 'MA_10', 'MA_14', 
    # Exponential Moving Averages
    'EMA_7', 'EMA_10', 'EMA_14', 
    # Close
    'close',
    # Close Lag Values
    'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_4', 'Close_Lag_5']
    target = 'Tomorrow'

    # Function to load feature importances from S3
    feature_importances = load_json_from_s3(bucket_name, feature_importances_key)
    if feature_importances is not None:
        if verbose:
            print("Feature Importances:")
            print(dumps(feature_importances, indent=4))  # Pretty-print the JSON

    # Function to load individual models from S3, modified to accept features list
    individual_models = load_individual_models_from_s3(bucket_name, models_prefix, required_features=features)
    if not individual_models:
        print("No individual models were loaded. Please check the S3 bucket and prefix.")
        return

    # --- Part 4: Prepare the data for prediction ---

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
    X_sequences = array(X_sequences)

    # --- Part 5: Make predictions ---

    weighted_predictions = zeros((X_sequences.shape[0], 1))

    if useModel == 0:
        # Baseline Model: Load and use the Baseline model
        baseline_model = load_baseline_model(bucket_name, models_prefix, verbose=verbose)
        if baseline_model:
            try:
                # Prepare input data using all features
                # Reshape X_sequences to match the Baseline model's expected input shape
                # Assuming the Baseline model expects input shape (batch_size, sequence_length, num_features)
                predictions = baseline_model.predict(X_sequences, batch_size=16, verbose=verbose)
                weight = 1.0  # Baseline model has full weight
                weighted_predictions += weight * predictions
                if verbose:
                    print(f"Made predictions using Baseline model with weight {weight}.")
            except Exception as e:
                print(f"Error making predictions with Baseline model: {e}")
        else:
            print("Baseline model could not be loaded.")
            
    elif useModel == 1:
        # Multi-Model Approach: Use all features with respective weights
        for feature in features:
            if feature in individual_models:
                try:
                    # Extract feature-specific sequences
                    feature_index = features.index(feature)
                    X_feature_sequences = X_sequences[:, :, feature_index].reshape(-1, sequence_length, 1)
                    model = individual_models[feature]
                    weight = feature_importances.get(feature, 0)  # Default weight 0 if not found
                    predictions = model.predict(X_feature_sequences, batch_size=16, verbose=verbose)
                    weighted_predictions += weight * predictions
                    if verbose:
                        print(f"Made predictions using model for feature '{feature}' with weight {weight}.")
                except Exception as e:
                    print(f"Error making predictions with model for feature '{feature}': {e}")
            else:
                print(f"No model found for feature '{feature}'. Skipping.")

    else:
        print(f"Invalid useModel value: {useModel}. Expected 0 or 1.")
        return



    # Inverse transform the predictions
    weighted_predictions = weighted_predictions.reshape(-1, 1)
    try:
        predictions_inverse = target_scaler.inverse_transform(weighted_predictions)
    except Exception as e:
        print(f"Error during inverse transformation of predictions: {e}")
        return

    # --- Part 6: Output the predictions ---

    # Get the last predicted value
    if predictions_inverse.size == 0:
        print("No predictions were made.")
        return

    last_prediction = predictions_inverse[-1][0]
    #print(f"Predicted {target} price for {coin_id} is: {last_prediction:.4f}")

    return last_prediction


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python scriptname.py <coin_id> <useModel: 0 baseline, 1 features>")
    else:
        coin_id = sys.argv[1]
        try:
            useModel = int(sys.argv[2])  # Convert to integer
        except ValueError:
            print("Error: useModel must be an integer (0 for Baseline, 1 for Multi-Model).")
            sys.exit(1)
        
        predicted_price = main(coin_id, useModel)
        if predicted_price is not None:
            print(f"{predicted_price:.4f}")
        else:
            print("Prediction could not be made due to an error in data fetching or processing.")
