import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from google.colab import drive
import tensorflow as tf
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping



########## FUNCTIONS ############

# Preprocess data
def preprocess_data(data):
    data['close'] = data['close'].ffill()  # Use ffill() to fill missing values
    return data

# Adjust model saving paths to handle invalid characters
def get_safe_filename(pair, interval):
    # Replace slashes or other invalid characters in the pair name with a safe character
    safe_pair = pair.replace('/', '_')
    return f'{safe_pair}_{interval}_model.keras'

# Add moving averages, RSI, and volume indicators
def add_features(data, windows=[10, 50, 100, 200]):
    for window in windows:
        data[f'MA_{window}'] = data['close'].rolling(window=window).mean()

    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    data['Volume_SMA_10'] = data['volume'].rolling(window=10).mean()

    data = data.dropna()  # Drop NaN values introduced by rolling calculations
    return data

# Detect Wyckoff accumulation phases
def detect_wyckoff_phases(data):
    phases = []

    # Initialize phase variables
    phase = None
    for i in range(len(data)):
        close = data['close'].iloc[i]
        volume = data['volume'].iloc[i]
        sma_10 = data['MA_10'].iloc[i]
        sma_50 = data['MA_50'].iloc[i]
        sma_200 = data['MA_200'].iloc[i]
        volume_sma = data['Volume_SMA_10'].iloc[i]

        # Volume increases should accompany significant price movements
        # Phase A: Preliminary support (PS) and selling climax (SC)
        if phase is None:
            if close < sma_50 and close < sma_200 and volume > volume_sma * 1.5:
                phase = 'A'
                phases.append(phase)
            else:
                phases.append('N/A')

        # Phase B: Building cause (range-bound) with low volume
        elif phase == 'A':
            if close > sma_10 and close < sma_50 and volume < volume_sma * 0.7:
                phase = 'B'
                phases.append(phase)
            else:
                phases.append('N/A')

        # Phase C: Spring or final test with higher volume
        elif phase == 'B':
            if close < sma_10 and close < sma_50 and volume > volume_sma:
                phase = 'C'
                phases.append(phase)
            else:
                phases.append('B')

        # Phase D: Breaking out of range with increasing volume
        elif phase == 'C':
            if close > sma_50 and close > sma_200 and volume > volume_sma:
                phase = 'D'
                phases.append(phase)
            else:
                phases.append('C')

        # Phase E: Markup continuation with sustained high volume
        elif phase == 'D':
            if close > sma_200 and volume > volume_sma:
                phase = 'E'
                phases.append(phase)
            else:
                phases.append('D')

    return phases

# Prepare data for training
def prepare_data(data, feature_cols, target_col, window_size):
    X, y = [], []
    data_array = data[feature_cols].values
    target_array = data[target_col].values

    for i in range(window_size, len(data)):
        X.append(data_array[i-window_size:i])
        y.append(target_array[i])

    return np.array(X), np.array(y)

# Build the LSTM model
def build_lstm_model(input_shape, lstm_units=64, dropout_rate=0.33):
    model = Sequential([
        Input(shape=input_shape),  # Define the input shape using the Input layer
        LSTM(30, activation='relu', return_sequences=True),  # Output sequences for intermediate layers
        Dropout(0.1),
        LSTM(40, activation='relu', return_sequences=True),  # Additional LSTM layer
        Dropout(0.1),
        LSTM(50, activation='relu', return_sequences=True),  # Additional LSTM layer
        Dropout(0.1),
        LSTM(60, activation='relu', return_sequences=False),  # Ensure this LSTM does not return sequences
        Dropout(0.1),
        Dense(10, activation='relu'),
        Dense(1)  # Final dense layer outputs a single value
    ])
    optimizer = Adam(learning_rate=0.0025)
    model.compile(optimizer='adam', loss=Huber())
    return model

# Plot predictions
def plot_predictions(data, y_test, predictions, interval, pair, window_size):
    # Use the original 'date' and 'close' columns for plotting
    actual_dates = pd.to_datetime(data.index[-len(y_test):].values)
    actual_prices = data['close'].iloc[-len(y_test):].values

    plt.figure(figsize=(14, 7))
    plt.plot(actual_dates, actual_prices, label='Actual Prices', color='blue')
    plt.plot(actual_dates, predictions, label='Predicted Prices', color='red', alpha=0.5)
    plt.title(f'Actual vs Predicted Prices for {pair} {interval} Interval')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=90)

    # Set x-axis to display one tick per week
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot training loss
def plot_loss(history, interval, pair):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title(f'Training Loss for {pair} {interval} Interval')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Train hierarchical models with plotting
def train_hierarchical_models_with_plots(data_dict, intervals, window_size=60, epochs=10, batch_size=32):
    previous_model = None
    for interval in intervals:
        for pair, data in data_dict[interval].items():
            print(f"Training on {pair} for interval {interval}")

            # Add Wyckoff phases to the data
            data['Wyckoff_Phase'] = detect_wyckoff_phases(data)

            # Encode phases numerically for model input
            data['Wyckoff_Phase_Encoded'] = data['Wyckoff_Phase'].astype('category').cat.codes

            # Define features
            features = ['close', 'close_btc', 'high', 'low', 'volume', 'MA_10', 'MA_50', 'MA_100', 'RSI', 'Wyckoff_Phase_Encoded']

            # Prepare data
            X, y = prepare_data(data, features, 'close', window_size)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            # Reshape for LSTM input
            X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], len(features)))
            X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], len(features)))

            # Build model
            if previous_model is None:
                model = build_lstm_model((X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
            else:
                # Initialize the new model with the weights of the previous model
                model = build_lstm_model((X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
                model.set_weights(previous_model.get_weights())

            # Train model within the device context manager
            with tf.device('/device:GPU:0' if gpus else '/cpu:0'):
                history = model.fit(
                    X_train_reshaped, 
                    y_train, 
                    epochs=epochs,  # Maximum number of epochs
                    batch_size=batch_size, 
                    verbose=1, 
                    callbacks=[early_stopping]  # Add early stopping here
                )

            # Evaluate model
            predictions_scaled = model.predict(X_test_reshaped)

            # Create an array with zeros for all columns except the one to be inverse transformed
            predictions = np.zeros((predictions_scaled.shape[0], len(columns_to_scale)))
            predictions[:, 0] = predictions_scaled.squeeze()  # Place predictions in the first column (assuming 'close' is the first feature)

            # Apply inverse transform
            predictions = scaler.inverse_transform(predictions)[:, 0]  # Only retrieve the first column which is the 'close' price

            y_test = np.squeeze(y_test)  # Ensure y_test is 1D

            # Debugging: Print shapes to verify they match
            print(f"Shape of y_test: {y_test.shape}, Shape of predictions: {predictions.shape}")

            mse = mean_squared_error(y_test, predictions)
            print(f"Mean Squared Error for {interval} interval on {pair}: {mse}")

            # Plot predictions with actual prices and dates
            plot_predictions(data, y_test, predictions, interval, pair, window_size)

            # Plot training loss
            plot_loss(history, interval, pair)

            # Save the model after training each interval and pair
            model_filename = get_safe_filename(pair, interval)
            model_save_path = f'/content/drive/My Drive/Colab Notebooks/Models/{model_filename}'
            model.save(model_save_path)
            print(f"Model saved for {pair} {interval} interval at {model_save_path}")

            # Set the current model as the previous model for the next iteration
            previous_model = model

    # Save the final model after all intervals and pairs
    final_model_save_path = '/content/drive/My Drive/Colab Notebooks/Models/final_model.keras'
    previous_model.save(final_model_save_path)
    print(f"Final model saved at {final_model_save_path}")

    return model

################## END FUNCTIONS ###########################

debug = True

# Initialize EarlyStopping
early_stopping = EarlyStopping(
    monitor='loss',  # Monitor the training loss
    patience=5,     # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restore model weights from the epoch with the best loss
)

# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')
if debug:
    if gpus:
        print("GPU is available.")
        # Log device placement for debugging
        # tf.debugging.set_log_device_placement(True)
    else:
        print("GPU is not available.")

# Mount Google Drive to access files
drive.mount('/content/drive')

# Load the CSV files for XRP/USD and XRP/BTC
def load_data(interval):
    # Adjust the file paths to the correct names
    if interval == '5m':
        usd_path = '/content/drive/My Drive/Colab Notebooks/Price_Data/XRP/USD/5min_data.csv'
        btc_path = '/content/drive/My Drive/Colab Notebooks/Price_Data/XRP/BTC/5min_data.csv'
    elif interval == '15m':
        usd_path = '/content/drive/My Drive/Colab Notebooks/Price_Data/XRP/USD/15min_data.csv'
        btc_path = '/content/drive/My Drive/Colab Notebooks/Price_Data/XRP/BTC/15min_data.csv'
    elif interval == '30m':
        usd_path = '/content/drive/My Drive/Colab Notebooks/Price_Data/XRP/USD/30min_data.csv'
        btc_path = '/content/drive/My Drive/Colab Notebooks/Price_Data/XRP/BTC/30min_data.csv'
    elif interval == '1h':
        usd_path = '/content/drive/My Drive/Colab Notebooks/Price_Data/XRP/USD/hourly_data.csv'
        btc_path = '/content/drive/My Drive/Colab Notebooks/Price_Data/XRP/BTC/hourly_data.csv'
    elif interval == '6h':
        usd_path = '/content/drive/My Drive/Colab Notebooks/Price_Data/XRP/USD/6hour_data.csv'
        btc_path = '/content/drive/My Drive/Colab Notebooks/Price_Data/XRP/BTC/6hour_data.csv'
    elif interval == '1d':
        usd_path = '/content/drive/My Drive/Colab Notebooks/Price_Data/XRP/USD/daily_data.csv'
        btc_path = '/content/drive/My Drive/Colab Notebooks/Price_Data/XRP/BTC/daily_data.csv'
    else:
        raise ValueError("Interval not recognized")

    # Load the data from CSV files
    data_xrp_usd = pd.read_csv(usd_path)
    data_xrp_btc = pd.read_csv(btc_path)
    return data_xrp_usd, data_xrp_btc

# Merge datasets
def prepare_combined_dataset(xrp_usd, xrp_btc):
    # Ensure both datasets have a consistent datetime index
    xrp_usd['date'] = pd.to_datetime(xrp_usd['date'])
    xrp_btc['date'] = pd.to_datetime(xrp_btc['date'])

    # Set the date column as the index
    xrp_usd.set_index('date', inplace=True)
    xrp_btc.set_index('date', inplace=True)

    # Merge datasets on the index, which is the date
    combined_data = xrp_usd.join(xrp_btc['close'], how='left', rsuffix='_btc')

    # Forward fill any missing values due to non-overlapping dates
    combined_data.ffill(inplace=True)

    return combined_data

# Load, combine, and preprocess data for each interval
intervals = ['15m', '30m', '1h', '6h', '1d']
interval_pairs = {}

for interval in intervals:
    data_xrp_usd, data_xrp_btc = load_data(interval)

    # Prepare combined dataset
    combined_data = prepare_combined_dataset(data_xrp_usd, data_xrp_btc)

    # Preprocess and add features to the combined dataset
    combined_data = add_features(preprocess_data(combined_data))

    # Normalize data
    scaler = MinMaxScaler()
    columns_to_scale = ['close', 'close_btc', 'high', 'low', 'volume', 'MA_10', 'MA_50', 'MA_100', 'RSI']

    # Use .loc to assign the scaled values back to the DataFrame
    combined_data.loc[:, columns_to_scale] = scaler.fit_transform(combined_data[columns_to_scale])

    # Store the scaler and the combined data for each interval
    if interval not in interval_pairs:
        interval_pairs[interval] = {}
    interval_pairs[interval]['XRP/USD'] = combined_data  # Using XRP/USD as base with additional BTC close prices

# Train models with visualization
model = train_hierarchical_models_with_plots(interval_pairs, intervals, window_size=50, epochs=1000, batch_size=256)

# Check GPU usage
!nvidia-smi
