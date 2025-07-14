
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# load the data
print("loading data...")
df = pd.read_csv('assignment 2\stock_data.csv') 

# convert date column to datetime
df['Date'] = pd.to_datetime(df['Date'])
df.reset_index(drop=True, inplace=True)

# check missing values
print("\nmissing values in each column:")
print(df.isnull().sum())

# find rows where close is missing
missing_close_idx = df[df['Close'].isnull()].index
print(f"\ntotal missing close values: {len(missing_close_idx)}")

# create features for training
feature_cols = ['Open', 'High', 'Low', 'Volume']

print("\nfilling missing values in feature columns...")

# function to fill missing values more accurately
def fill_missing_values_accurately(df):
    df_filled = df.copy()
    
    for col in feature_cols:
        missing_count = df_filled[col].isnull().sum()
        if missing_count > 0:
            print(f"filling {missing_count} missing values in {col}")
            
            # method 1: use relationship with close price if available
            if col in ['Open', 'High', 'Low']:
                # calculate typical ratios with close price
                valid_data = df_filled[(df_filled[col].notna()) & (df_filled['Close'].notna())]
                
                if len(valid_data) > 10:
                    if col == 'Open':
                        # open is usually close to previous day close
                        avg_ratio = (valid_data[col] / valid_data['Close'].shift(1)).median()
                    elif col == 'High':
                        # high is usually 1-5% above close
                        avg_ratio = (valid_data[col] / valid_data['Close']).median()
                    elif col == 'Low':
                        # low is usually 1-5% below close
                        avg_ratio = (valid_data[col] / valid_data['Close']).median()
                    
                    # fill using ratio with close price
                    for idx in df_filled[df_filled[col].isnull()].index:
                        if pd.notna(df_filled.loc[idx, 'Close']):
                            if col == 'Open' and idx > 0:
                                prev_close = df_filled.loc[idx-1, 'Close']
                                if pd.notna(prev_close):
                                    df_filled.loc[idx, col] = prev_close * avg_ratio
                            else:
                                df_filled.loc[idx, col] = df_filled.loc[idx, 'Close'] * avg_ratio
            
            # method 2: interpolation for remaining missing values
            df_filled[col] = df_filled[col].interpolate(method='linear')
            
            # method 3: forward fill then backward fill for edge cases
            df_filled[col] = df_filled[col].ffill().bfill()
            
    
    # special handling for volume
    if 'Volume' in df_filled.columns and df_filled['Volume'].isnull().any():
        print(f"filling {df_filled['Volume'].isnull().sum()} missing values in Volume")
        
        # method 1: use moving average of volume
        df_filled['Volume'] = df_filled['Volume'].fillna(
            df_filled['Volume'].rolling(window=10, min_periods=1).mean()
        )
        
        # method 2: interpolation
        df_filled['Volume'] = df_filled['Volume'].interpolate(method='linear')
        
        # method 3: forward/backward fill
        df_filled['Volume'] = df_filled['Volume'].ffill().bfill()
        
    return df_filled

# apply accurate missing value filling
df = fill_missing_values_accurately(df)

# verify no missing values remain in feature columns
print("\nafter filling - missing values in feature columns:")
for col in feature_cols:
    missing = df[col].isnull().sum()
    print(f"{col}: {missing} missing values")


# separate data with close values (for training) and without close values (for prediction)
train_data = df[df['Close'].notna()].copy()
predict_data = df[df['Close'].isna()].copy()

print(f"training data size: {len(train_data)}")
print(f"prediction data size: {len(predict_data)}")

# prepare features for training
all_features = ['Open', 'High', 'Low', 'Volume', 'Close']
train_features = train_data[all_features].values

# scale the data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_features)

# create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, :-1])  # all features except close
        y.append(data[i, -1])  # close price
    return np.array(X), np.array(y)

# sequence length - how many previous days to look at
seq_length = 20  # increased for better pattern recognition in stock data
print(f"using sequence length: {seq_length} days")

X_train, y_train = create_sequences(train_scaled, seq_length)

print(f"training sequences shape: {X_train.shape}")
print(f"training targets shape: {y_train.shape}")

# build optimized LSTM model for stock prediction
model = keras.models.Sequential()

# first LSTM layer - increased units for better pattern capture
model.add(keras.layers.LSTM(units=128, return_sequences=True, input_shape=(seq_length, 4)))
model.add(keras.layers.Dropout(0.3))  # slightly higher dropout for stock volatility

# second LSTM layer - bidirectional for better context
model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=64, return_sequences=True)))
model.add(keras.layers.Dropout(0.3))

# third LSTM layer - attention mechanism equivalent
model.add(keras.layers.LSTM(units=64, return_sequences=False))
model.add(keras.layers.Dropout(0.2))

# dense layers for better feature extraction
model.add(keras.layers.Dense(units=50, activation='relu'))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(units=25, activation='relu'))
model.add(keras.layers.Dropout(0.1))

# output layer
model.add(keras.layers.Dense(units=1, activation='linear'))

# compile model with optimized settings
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='huber',  # better for outliers in stock data
    metrics=['mae', 'mse']
)

print("\noptimized model summary:")
model.summary()

# train the model with optimized parameters
print("\ntraining optimized model...")

# callbacks for better training
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=25,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=20,
    min_lr=1e-7,
    verbose=1
)

# optimized training parameters
history = model.fit(
    X_train, y_train, 
    epochs=100,  # increased epochs with early stopping
    batch_size=16,  # smaller batch size for better gradient updates
    verbose=1, 
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    shuffle=True
)


# now predict missing close values
predictions = []

for idx in missing_close_idx:
    # get the sequence before this missing value
    start_idx = max(0, idx - seq_length)
    
    # create sequence using available data
    sequence_data = []
    for i in range(start_idx, idx):
        if i < len(df):
            row_data = []
            for col in ['Open', 'High', 'Low', 'Volume']:
                if pd.notna(df.loc[i, col]):
                    row_data.append(df.loc[i, col])
                else:
                    # use previous value if missing
                    prev_val = df.loc[max(0, i-1), col] if i > 0 else 0
                    row_data.append(prev_val)
            
            # add close value if available, otherwise use open price as estimate
            if pd.notna(df.loc[i, 'Close']):
                row_data.append(df.loc[i, 'Close'])
            else:
                row_data.append(df.loc[i, 'Open'])  # use open as close estimate
            
            sequence_data.append(row_data)
    
    # pad sequence if too short
    while len(sequence_data) < seq_length:
        if len(sequence_data) > 0:
            sequence_data.insert(0, sequence_data[0])  # repeat first row
        else:
            sequence_data.append([0, 0, 0, 0, 0])  # default values
    
    # take last seq_length rows
    sequence_data = sequence_data[-seq_length:]
    sequence_array = np.array(sequence_data)
    
    # scale the sequence
    sequence_scaled = scaler.transform(sequence_array)
    
    # prepare for prediction (remove close column)
    X_pred = sequence_scaled[:, :-1].reshape(1, seq_length, 4)
    
    # predict
    pred_scaled = model.predict(X_pred, verbose=0)
    
    # inverse transform to get actual price
    # create dummy array for inverse transform
    dummy_array = np.zeros((1, 5))
    dummy_array[0, -1] = pred_scaled[0, 0]
    pred_actual = scaler.inverse_transform(dummy_array)[0, -1]
    
    predictions.append(pred_actual)
    print(f"predicted close for index {idx} (date {df.loc[idx, 'Date']}): {pred_actual:.2f}")

# fill the missing values in dataframe
df_result = df.copy()
for i, idx in enumerate(missing_close_idx):
    df_result.loc[idx, 'Close'] = predictions[i]

# save results
print("\nsaving results...")
df_result.to_csv('stock_data_predicted.csv', index=False)

# create submission file with only missing close values
submission_data = []
for i, idx in enumerate(missing_close_idx):
    submission_data.append({
        'Date': df.loc[idx, 'Date'],
        'Close': predictions[i]
    })

submission_df = pd.DataFrame(submission_data)
submission_df.to_csv('missing_close_predictions.csv', index=False)

print("done! check these files:")
print("1. stock_data_predicted.csv - complete dataset with predictions")
print("2. missing_close_predictions.csv - only the missing close values")
