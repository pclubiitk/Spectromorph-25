import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

data = pd.read_csv("/Users/manojkumar/Desktop/HW/New/stock_data.csv")
data['Date'] = pd.to_datetime(data['Date'])

missing_dates = data[data['Close'].isna()]['Date']
train_data = data.dropna(subset=['Close']).sort_values('Date')

close_vals = train_data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(close_vals)

def create_sequences(seq, window):
    X, y = [], []
    for i in range(len(seq) - window):
        X.append(seq[i:i+window])
        y.append(seq[i+window])
    return np.array(X), np.array(y)

window_size = 10
X, y = create_sequences(scaled_close, window_size)

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(window_size, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=20, batch_size=16)

predictions = []
for date in missing_dates:
    past_data = data[data['Date'] < date].dropna(subset=['Close']).tail(window_size)
    if len(past_data) < window_size:
        predictions.append(np.nan)
        continue
    input_seq = scaler.transform(past_data['Close'].values.reshape(-1, 1))
    input_seq = np.expand_dims(input_seq, axis=0)
    pred_scaled = model.predict(input_seq)
    pred = scaler.inverse_transform(pred_scaled)[0][0]
    predictions.append(pred)

output = pd.DataFrame({
    'Date': missing_dates.dt.strftime('%Y-%m-%d'),
    'Close': predictions
})
output.dropna(inplace=True)
output.to_csv("submission.csv", index=False)