# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:10:46 2024

@author: MCNB
"""


import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import json
import tempfile
from tensorflow.keras.models import load_model

# Initialize the S3 client
s3_client = boto3.client('s3')

# Define the S3 bucket and object keys
bucket_name = 'pricepredictor-bucket'
feature_importances_key = 'feature_importances.json'  # Feature importances file in S3
model_key = 'model_high.keras'  # Keras model file in S3

# Function to load JSON data from S3
def load_json_from_s3(bucket, key):
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        data = json.loads(content)
        print(f"Successfully loaded '{key}' from bucket '{bucket}'.")
        return data
    except NoCredentialsError:
        print("Error: AWS credentials not available.")
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            print(f"Error: The object '{key}' does not exist in bucket '{bucket}'.")
        else:
            print(f"An error occurred: {e}")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{key}'.")
    return None

# Function to load Keras model from S3
def load_keras_model_from_s3(bucket, key):
    try:
        # Create a temporary file to store the downloaded model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp_file:
            s3_client.download_fileobj(bucket, key, tmp_file)
            tmp_file_path = tmp_file.name
            print(f"Model '{key}' downloaded to temporary file '{tmp_file_path}'.")
        
        # Load the Keras model from the temporary file
        model = load_model(tmp_file_path)
        print("Keras model loaded successfully!")
        model.summary()
        return model
    except NoCredentialsError:
        print("Error: AWS credentials not available.")
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            print(f"Error: The model file '{key}' does not exist in bucket '{bucket}'.")
        else:
            print(f"An error occurred while downloading the model: {e}")
    except Exception as e:
        print(f"Error loading the Keras model: {e}")
    return None

# Load feature importances
feature_importances = load_json_from_s3(bucket_name, feature_importances_key)
if feature_importances is not None:
    print("Feature Importances:")
    print(json.dumps(feature_importances, indent=4))  # Pretty-print the JSON

# Load Keras model
model = load_keras_model_from_s3(bucket_name, model_key)
if model is not None:
    # You can add any additional processing with the model here
    pass
