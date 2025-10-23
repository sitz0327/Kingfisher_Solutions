import datetime
import sklearn
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import KernelPCA
import numpy as np
import pandas as pd
import math
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split


DB_FILEPATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data", "DBcleaned.csv")
OUTPUT = 'kWh Normalised'


df = pd.read_csv(DB_FILEPATH)
df = df[df['id'] == 577650]
df.drop(columns = ['name','id','kWh','datetime','maximumPower','month','day','hour','dayOfYear', 'year', 'Cloud Fill Flag', 'Cloud Type', 'Fill Flag'], inplace=True)


imputer = SimpleImputer(missing_values=np.nan)  # Handling missing values
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
df = df.reset_index(drop=True)
# Applying feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df.to_numpy())
df_scaled = pd.DataFrame(df_scaled, columns=list(df.columns))
target_scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = df_scaled.astype(float)


# Single step dataset preparation
def singleStepSampler(df, window):
    xRes = []
    yRes = []
    for i in range(0, len(df) - window):
        res = []
        for j in range(0, window):
            r = []
            for col in df.columns:
                r.append(df[col][i + j])
            res.append(r)
        xRes.append(res)
        yRes.append(df[['kWh Normalised']].iloc[i + window].values)
    return np.array(xRes), np.array(yRes)


# Split data

(X, y) = singleStepSampler(df_scaled, 20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

multivariate_lstm = keras.Sequential()
multivariate_lstm.add(keras.layers.LSTM(200, input_shape=(X_train.shape[1], X_train.shape[2])))
multivariate_lstm.add(keras.layers.Dropout(0.2))
multivariate_lstm.add(keras.layers.Dense(2, activation='linear'))
multivariate_lstm.compile(loss = 'MeanSquaredError', metrics=['MAE'], optimizer='Adam')
multivariate_lstm.summary()

history = multivariate_lstm.fit(X_train, y_train, epochs=20)


# Reload the data with the date index
dataFrame = pd.read_csv(DB_FILEPATH)  # Assuming the CSV file contains a 'Date' column
dataFrame = dataFrame[dataFrame['id'] == 577650 ] 
dataFrame.set_index('datetime', inplace=True)

# Forecast Plot with Dates on X-axis
predicted_values = multivariate_lstm.predict(X_test)

d = {
    'Predicted_kWh_norm': predicted_values[:, 0],
    'Actual_kWh_norm': y_test[:, 0],
}

d = pd.DataFrame(d)
d.index = dataFrame.index[-len(y_test):]  # Assigning the correct date index

fig, ax = plt.subplots(figsize=(10, 6))
#  highlight the  forecast
highlight_start = int(len(d) * 0.9)  
highlight_end = len(d) - 1  # Adjusted to stay within bounds
# Plot the actual values
plt.plot(d[['Actual_kWh_norm']][:highlight_start], label=['Actual_kWh_norm'])

# Plot predicted values with a dashed line
plt.plot(d[['Predicted_kWh_norm']], label=['Predicted_kWh_norm'], linestyle='--')

# Highlight the forecasted portion with a different color
plt.axvspan(d.index[highlight_start], d.index[highlight_end], facecolor='lightgreen', alpha=0.5, label='Forecast')

plt.title('Multivariate Time-Series forecasting using LSTM')
plt.xlabel('Dates')
plt.ylabel('Values')
ax.legend()
plt.show()



# Model Evaluation
def eval(model):
    return {
        'MSE': sklearn.metrics.mean_squared_error(d[f'Actual_kWh_norm'].to_numpy(), d[model].to_numpy()),
        'MAE': sklearn.metrics.mean_absolute_error(d[f'Actual_kWh_norm'].to_numpy(), d[model].to_numpy()),
        'R2': sklearn.metrics.r2_score(d[f'Actual_kWh_norm'].to_numpy(), d[model].to_numpy())
    }

result = dict()

for item in ['Predicted_kWh_norm']:
    result[item] = eval(item)

result