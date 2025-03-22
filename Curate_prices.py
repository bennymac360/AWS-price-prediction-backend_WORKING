# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 10:04:23 2024

@author: MCNB
"""

import pandas as pd

pair = "XRPBTC"

# Load the data
file_path = 'D:/Users/MCNB/Downloads/merged_sorted_stock_data_'+pair+'.csv'  # Update this with your actual file path
data = pd.read_csv(file_path)

# Convert the 'date' column to datetime
data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y %H:%M')

# Set the 'date' column as the index
data.set_index('date', inplace=True)

# Define the resampling intervals
intervals = {
    '5min': '5min_data_'+pair+'.csv',
    '15min': '15min_data_'+pair+'.csv',
    '30min': '30min_data_'+pair+'.csv',
    '1h': 'hourly_data_'+pair+'.csv',
    '6h': '6hour_data_'+pair+'.csv',
    '24h': 'daily_data_'+pair+'.csv'
}

# Function to resample and aggregate data
def resample_data(data, frequency):
    return data.resample(frequency).agg({
        'open': 'first',   # Opening price of the interval
        'high': 'max',     # Maximum price during the interval
        'low': 'min',      # Minimum price during the interval
        'close': 'last',   # Closing price at the end of the interval
        'volume': 'sum',   # Total volume during the interval
        'volume_from': 'sum',
        'tradecount': 'sum'
    })

# Resample the data for each interval and save to CSV
for interval, filename in intervals.items():
    resampled_data = resample_data(data, interval)
    resampled_data.to_csv(filename)

# Display a message to indicate completion
print("Data has been resampled and saved to CSV files for each interval.")