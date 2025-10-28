import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from tabulate import tabulate

from darts import TimeSeries
from darts.models import TFTModel
from darts.models.forecasting.rnn_model import RNNModel
from darts.dataprocessing.transformers.scaler import Scaler

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping

import keras_tuner

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor



DB_FILEPATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data", "DBcleaned.csv")
OUTPUT = 'kWh Normalised'
SEED = 510444756

df = pd.read_csv(DB_FILEPATH)
print(df.columns)

X = df.select_dtypes(include=['float64', 'int64']).drop(columns=[OUTPUT, 'kWh'])
y = df[OUTPUT]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

## dimension reduction df
dr = Nystroem(kernel='rbf',gamma = 0.0002, n_components=10, random_state=510444756, n_jobs=-1)
X_nystroem = dr.fit_transform(X_scaled)

dr_X_train, dr_X_test, dr_y_train, dr_y_test = train_test_split(X_nystroem, y, test_size=0.2, random_state=42)


#All Variables:
#['name', 'kWh', 'datetime', 'Temperature', 'Alpha', 'Aerosol Optical Depth', 'Asymmetry', 'Clearsky DHI', 'Clearsky DNI','Clearsky GHI', 'Dew Point', 'DHI', 'DNI', 'GHI', 'Ozone','Relative Humidity', 'Solar Zenith Angle', 'SSA', 'Surface Albedo','Pressure', 'Precipitable Water', 'Wind Direction', 'Wind Speed','Global Horizontal UV Irradiance (280-400nm)', 'Global Horizontal UV Irradiance (295-385nm)', 'Cloud Type_0.0', 'Cloud Type_1.0', 'Cloud Type_3.0', 'Cloud Type_4.0', 'Cloud Type_5.0','Cloud Type_6.0', 'Cloud Type_7.0', 'Cloud Type_8.0', 'Cloud Type_9.0', 'Cloud Type_nan', 'day_sine', 'day_cosine', 'year_sine', 'year_cosine','kWh Normalised']
# Selected variables
X = df[['Global Horizontal UV Irradiance (280-400nm)', 'Global Horizontal UV Irradiance (295-385nm)', 'GHI', 'day_cosine', 'year_cosine', 'Relative Humidity', 'Temperature', 'Pressure', 'Clearsky GHI', 'Precipitable Water', 'Clearsky DHI', 'Solar Zenith Angle']]
y = df[['kWh Normalised']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=SEED)


## Models
models = {
    'Lin_Reg':LinearRegression,
    'TFT':TFTModel , 
    "RNN": RNNModel,
    "LGBM": LGBMRegressor,
    "XGBoost": XGBRegressor,
    "NN": Sequential
}


## Evaluation
metrics = {
    'MAE': mean_absolute_error,
    'MSE': mean_squared_error,
    'r2': r2_score
}
metrics_frame = pd.DataFrame()

def append_metrics(model_name, test, pred, note, metrics_frame):
    records = []
    for metric in metrics.keys():
        records.append({
                'Metric': [metric],
                'Model': [model_name],
                'Score': [round(metrics[metric](test,pred),3)],
                'Notes': [note]
            })
    new_rows = pd.DataFrame(records, columns = ['Metric', 'Model', 'Score', 'Notes']) 
    metrics_frame = pd.concat([metrics_frame, new_rows],ignore_index=True)
    return metrics_frame






# NON TimeSeries models:




#LGBMRegressor
model_name = 'LGBM'

model = LGBMRegressor(n_estimators=5000, learning_rate=0.05, n_jobs=-1)


model.fit(X_train, y_train)
y_pred = model.predict(X_test)
metrics_frame = append_metrics(model_name, y_test, y_pred, "Selected variables (10)", metrics_frame)


model.fit(dr_X_train, dr_y_train)
dr_y_pred = model.predict(dr_X_test)
metrics_frame = append_metrics(model_name, dr_y_test, dr_y_pred, "Dimensionally reduced (10)", metrics_frame)



# XGBoost
model_name = "XGBoost"

model = XGBRegressor(
    n_estimators=5000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=510444756
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
metrics_frame = append_metrics(model_name, y_test, y_pred, "Selected Variables (10)", metrics_frame)

model.fit(dr_X_train, dr_y_train)
dr_y_pred = model.predict(dr_X_test)
metrics_frame = append_metrics(model_name, dr_y_test, dr_y_pred, "Dimensionally reduced (10)", metrics_frame)



#TensorFlow NN
tf_models = {
    "FNNBatchNorm": Sequential([
    Dense(160, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1)]),

    "FNN": Sequential([
    Dense(160, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1)]),
}

model_name = "FNNBatchNorm"
model = tf_models[model_name]

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

y_pred = model.predict(X_test).flatten()

metrics_frame = append_metrics(model_name, y_test, y_pred, "Selected Variables (10)", metrics_frame)

# Training curve
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Training History')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.show()

# Predictions vs Actual
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual kWh_normalised')
plt.ylabel('Predicted kWh_normalised')
plt.title('Prediction Performance')
plt.show()

print(tabulate(metrics_frame, headers='keys', tablefmt='pretty', showindex=False))