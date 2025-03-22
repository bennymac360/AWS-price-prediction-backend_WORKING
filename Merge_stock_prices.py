import os
import pandas as pd

# Specify the directory containing your CSV files
directory = "./"

# The expected column names when loading and saving
load_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
save_columns = ['time', 'open', 'high', 'low', 'close', 'Volume']

# Function to extract stock name from filename
def get_stock_name(filename):
    return filename.split('_')[1].split(',')[0]

# Initialize a dictionary to store dataframes for each stock
stock_data = {}

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        # Read the CSV file with lowercase column names
        df = pd.read_csv(filepath)
        
        # Ensure the columns are lowercase and match the expected loading columns
        df.columns = df.columns.str.lower().str.strip()
        
        # Reorder the columns according to the load_columns
        df = df[load_columns]
        
        # Sort the data by time (UNIX timestamp) to ensure correct ordering
        df = df.sort_values(by='time').drop_duplicates(subset='time')
        
        # Get the stock name
        stock_name = get_stock_name(filename)
        
        if stock_name in stock_data:
            # If we already have data for this stock, append the new data
            stock_data[stock_name] = pd.concat([stock_data[stock_name], df])
        else:
            # If this is the first file for this stock, just store the dataframe
            stock_data[stock_name] = df

# Now let's remove duplicates and sort by time for each stock and save the merged file
for stock, data in stock_data.items():
    # Remove any duplicates by UNIX timestamp
    data = data.drop_duplicates(subset='time')
    # Sort the data by time to ensure chronological order
    data = data.sort_values(by='time')
    # Save the merged data to a new CSV file with correct column names
    output_filename = f"5m_{stock}.csv"
    output_filepath = os.path.join(directory, output_filename)
    # Rename columns for saving
    data.columns = save_columns
    data.to_csv(output_filepath, index=False)

print("Merging complete. Files have been saved.")
