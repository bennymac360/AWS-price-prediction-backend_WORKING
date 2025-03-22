# -*- coding: utf-8 -*-

import pandas as pd


pair = "XRPBTC"
# Load the CSV files
file_2020 = "D:/Users/MCNB/Downloads/Binance_"+pair+"_2020_minute.csv"
file_2021 = "D:/Users/MCNB/Downloads/Binance_"+pair+"_2021_minute.csv"
file_2022 = "D:/Users/MCNB/Downloads/Binance_"+pair+"_2022_minute.csv"
file_2023 = "D:/Users/MCNB/Downloads/Binance_"+pair+"_2023_minute.csv"
file_2024 = "D:/Users/MCNB/Downloads/Binance_"+pair+"_2024_minute.csv"

# Read the CSV files into DataFrames ensuring headers are correctly used
df_2020 = pd.read_csv(file_2020)
df_2021 = pd.read_csv(file_2021)
df_2022 = pd.read_csv(file_2022)
df_2023 = pd.read_csv(file_2023)
#df_2024 = pd.read_csv(file_2024)

# Check if the headers are the same in both DataFrames
if not (df_2020.columns == df_2021.columns).all():
    raise ValueError("Column headers do not match between files!")

# Merge the DataFrames
merged_df = pd.concat([df_2023,df_2022,df_2021,df_2020])

# Sort the merged DataFrame by date and time
merged_df = merged_df.sort_values(by=['date', 'unix'])

# Save the merged and sorted DataFrame to a new CSV file
output_file = 'D:/Users/MCNB/Downloads/merged_sorted_stock_data_'+pair+'.csv'
merged_df.to_csv(output_file, index=False)

# Print the path to the merged file
#print(output_file)
