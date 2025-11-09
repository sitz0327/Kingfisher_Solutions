import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from tabulate import tabulate

from darts import TimeSeries
from darts.models import TFTModel, NBEATSModel
from darts.metrics import rmse, mse, mae, r2_score ,confusion_matrix, mape
from darts.models.forecasting.rnn_model import RNNModel
from darts.utils.likelihood_models.torch import QuantileRegression
from darts.models.forecasting.tsmixer_model import TSMixerModel
from darts.dataprocessing.transformers.scaler import Scaler
from darts.dataprocessing.transformers.missing_values_filler import MissingValuesFiller

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer, accuracy_score, root_mean_squared_error, confusion_matrix
import sklearn.metrics as Metrics
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
X = df[['Global Horizontal UV Irradiance (280-400nm)', 'Global Horizontal UV Irradiance (295-385nm)', 'GHI', 'Relative Humidity', 'Temperature', 'Pressure', 'Clearsky GHI', 'Precipitable Water', 'Clearsky DHI', 'Solar Zenith Angle']] # 'day_cosine', 'year_cosine',
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
    'RMSE': rmse,
    'MAE': mae,
    'MSE': mse,
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


def temporal_robustness(model, series, past=None, future=None,
                 fh=12, stride=12, start=0.6, n_runs=5, seeds=(0,1,2,3,4)):
    scores = []
    for s in seeds[:n_runs]:
        model.random_state = s
        preds = model.historical_forecasts(
            series=series, start=start,
            forecast_horizon=fh, stride=stride,
            past_covariates=past, future_covariates=future,
            retrain=False, verbose=False)
        scores.append((mae(series, preds), rmse(series, preds), r2_score(series, preds)))
    scores = np.array(scores)
    return {
        "MAE_mean": float(np.mean(scores[:,0])),
        "MAE_med": float(np.median(scores[:,0])),
        "MAE_IQR": float(np.percentile(scores[:,0],75) - np.percentile(scores[:,0],25)),
        "RMSE_mean": float(np.mean(scores[:,1])),
        "RMSE_med": float(np.median(scores[:,1])),
        "RMSE_IQR": float(np.percentile(scores[:,1],75) - np.percentile(scores[:,1],25)),
        "r2_mean": float(np.mean(scores[:,2])),
        "r2_med": float(np.median(scores[:,2])),
        "r2_IQR": float(np.percentile(scores[:,2],75) - np.percentile(scores[:,1],25)),
    }


def drop_blocks(ts, frac=0.1, block=24, seed=0):
    idx = ts.time_index
    mask = np.ones(len(idx), bool)
    rng = np.random.default_rng(seed)
    n_blocks = max(1, int(len(idx)*frac/block))
    for _ in range(n_blocks):
        start = rng.integers(0, len(idx)-block)
        mask[start:start+block] = False
    return ts[mask]

def add_noise(ts, sigma=0.1, seed=0):
    """
    Adds Gaussian noise to one TimeSeries or a list of TimeSeries.
    Preserves shape and time alignment.
    """
    rng = np.random.default_rng(seed)

    # --- Case 1: list of TimeSeries ---
    if isinstance(ts, list):
        noisy_list = []
        for t in ts:
            noisy_vals = t.values(copy=True) + rng.normal(0, sigma, size=t.values().shape)
            noisy_list.append(
                TimeSeries.from_times_and_values(t.time_index, noisy_vals, columns=t.components)
            )
        return noisy_list

    # --- Case 2: single TimeSeries ---
    elif isinstance(ts, TimeSeries):
        noisy_vals = ts.values(copy=True) + rng.normal(0, sigma, size=ts.values().shape)
        return TimeSeries.from_times_and_values(ts.time_index, noisy_vals, columns=ts.components)

    # --- Case 3: None (no covariates) ---
    elif ts is None:
        return None

    else:
        raise TypeError("Input must be a TimeSeries or list of TimeSeries")

def brittle_test(model, train, test, past, future):

    baseline_mae = []
    baseline_rmse = []
    noisy_mae = []
    noisy_rmse = []
    for train, test, past, future in zip(train, test, past, future):
    # Baseline (clean)
        pred_base = model.predict(series=train, n=len(test), past_covariates=past, future_covariates=future, verbose = False, show_warnings=False)
        # Noisy covariates
        #noisy_past = add_noise(past_scaled, sigma=0.1, seed=510444756)
        noisy_future = add_noise(future, sigma=0.1, seed=510444756)

        pred_noise = model.predict(series=train, n=len(test), past_covariates=past, future_covariates=noisy_future, verbose = False, show_warnings=False)
        
        baseline_mae.append(mae(test, pred_base))
        baseline_rmse.append(rmse(test, pred_base))
        noisy_mae.append(mae(test, pred_noise))
        noisy_rmse.append(rmse(test, pred_base))


    


    print("\n\nBaseline  MAE", score(bases, series_scaled, target_scaler))
    print("Noise     MAE/RMSE:", score(noisy, series_scaled, target_scaler))
    # print("Gaps      MAE/RMSE:", score(pred_gaps, series_scaled))





# # NON TimeSeries models:

# #LGBMRegressor
# model_name = 'LGBM'

# model = LGBMRegressor(n_estimators=5000, learning_rate=0.05, n_jobs=-1)


# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# metrics_frame = append_metrics(model_name, y_test, y_pred, "Selected variables (12)", metrics_frame)


# model.fit(dr_X_train, dr_y_train)
# dr_y_pred = model.predict(dr_X_test)
# metrics_frame = append_metrics(model_name, dr_y_test, dr_y_pred, "Dimensionally reduced (12)", metrics_frame)



# # # XGBoost
# model_name = "XGBoost"

# model = XGBRegressor(
#     n_estimators=5000,
#     learning_rate=0.05,
#     max_depth=5,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=510444756
# )

# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# metrics_frame = append_metrics(model_name, y_test, y_pred, "Selected Variables (12)", metrics_frame)

# model.fit(dr_X_train, dr_y_train)
# dr_y_pred = model.predict(dr_X_test)
# metrics_frame = append_metrics(model_name, dr_y_test, dr_y_pred, "Dimensionally reduced (12)", metrics_frame)



# #TensorFlow NN
# tf_models = {
#     "FNNBatchNorm": Sequential([
#     Dense(160, activation='relu', input_shape=(X_train.shape[1],)),
#     Dropout(0.2),
#     BatchNormalization(),
#     Dense(32, activation='relu'),
#     Dropout(0.3),
#     Dense(1)]),

#     "FNN": Sequential([
#     Dense(160, activation='relu', input_shape=(X_train.shape[1],)),
#     Dropout(0.2),
#     Dense(32, activation='relu'),
#     Dropout(0.3),
#     Dense(1, activation='linear')]),
# }

# model_name = "FNN"
# model = tf_models[model_name]

# model.compile(
#     optimizer='adam',
#     loss='mse',
#     metrics=['mae']
# )

# early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# history = model.fit(
#     X_train, y_train,
#     validation_split=0.2,
#     epochs=100,
#     batch_size=32,
#     callbacks=[early_stop],
#     verbose=1
# )

# y_pred = model.predict(X_test).flatten()

# metrics_frame = append_metrics(model_name, y_test, y_pred, "Selected Variables (12)", metrics_frame)

# # Training curve
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Val Loss')
# plt.legend()
# plt.title('Training History')
# plt.xlabel('Epochs')
# plt.ylabel('MSE Loss')
# plt.show()

# # Predictions vs Actual
# plt.figure(figsize=(6,6))
# plt.scatter(y_test, y_pred, alpha=0.7)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
# plt.xlabel('Actual kWh_normalised')
# plt.ylabel('Predicted kWh_normalised')
# plt.title('Prediction Performance')
# plt.show()























## Sequential/ timeseries analysis using Darts
model_average_scores = {}
site_names = df['name'].unique()

target_series_list = []
past_covar_list = []
future_covar_list = []

for site in site_names:
    print(site)
    df_site = df[df['name'] == site].sort_values('date')

    #produce target and covariate time series
    target_series = TimeSeries.from_dataframe(df_site, time_col='date', value_cols='kWh Normalised', freq = 'h')
    past_covariates = TimeSeries.from_dataframe(df_site, time_col='date', freq='h', value_cols=['Global Horizontal UV Irradiance (280-400nm)', 'Global Horizontal UV Irradiance (295-385nm)', 'GHI', 'day_cosine', 'year_cosine', 'Relative Humidity', 'Temperature', 'Pressure', 'Clearsky GHI', 'Precipitable Water', 'Clearsky DHI', 'Solar Zenith Angle'])
    future_covariates = TimeSeries.from_dataframe(df_site, time_col='date', freq='h', value_cols=['GHI', 'day_cosine', 'year_cosine', 'Relative Humidity', 'Temperature', 'Pressure', 'Clearsky GHI', 'Precipitable Water', 'Clearsky DHI', 'Solar Zenith Angle'])

    #align start and end times for each timeseries
    start = max(target_series.start_time(), past_covariates.start_time(), future_covariates.start_time())
    end = min(target_series.end_time(), past_covariates.end_time(), future_covariates.end_time())
    target_series, past_covariates, future_covariates = target_series.slice(start, end), past_covariates.slice(start, end), future_covariates.slice(start, end)

    #
    target_series_list.append(target_series)
    past_covar_list.append(past_covariates)
    future_covar_list.append(future_covariates)

for series in target_series_list:
    series.plot()
    plt.show()


# Split (e.g., 80% train, 20% test)
target_scaler= Scaler()
past_covar_scaler= Scaler()
future_covar_scaler= Scaler()

series_scaled_list = target_scaler.fit_transform(target_series_list)
past_covar_scaled_list = past_covar_scaler.fit_transform(past_covar_list)
future_covar_scaled_list = future_covar_scaler.fit_transform(future_covar_list)


train_series = []
test_series = []
train_past_covar = []
test_past_covar = []
train_future_covar = []
test_future_covar = []

for s, p, f in zip(series_scaled_list, past_covar_scaled_list, future_covar_scaled_list):
    train, test = s.split_before(0.8)
    p_train, p_test = p.split_before(0.8)
    f_train, f_test = f.split_before(0.8)
 

    train_series.append(train)
    test_series.append(test)
    train_past_covar.append(p_train)
    test_past_covar.append(p)
    train_future_covar.append(f_train)
    test_future_covar.append(f)



## TFTModel
# model_name = "TFTModel"
# model = TFTModel(
#     input_chunk_length=24,   # how many past steps it looks at
#     output_chunk_length=6,  # how many future steps to predict
#     hidden_size=32,
#     lstm_layers=2,
#     n_epochs=25,
#     batch_size=32,
#     dropout=0.1,
#     random_state=SEED,
#     add_relative_index = True,
#     force_reset = True,
#     save_checkpoints=True,
#     model_name = model_name,
#     work_dir = os.path.join(os.path.dirname(DB_FILEPATH), model_name),
#     add_encoders={
#         'datetime_attribute': {
#             'past': ['hour', 'dayofyear'],
#             'future': ['hour' , 'dayofyear']
#         },
#         'cyclic': {
#             'past': ['hour', 'dayofyear'],
#             'future': ['hour' , 'dayofyear']
#         }
#     }
# )



# model.fit(series=train_series, past_covariates=train_past_covar, future_covariates = train_future_covar,  verbose=True)

# model.save("TFTmodel")
# model = TFTModel.load("TFTmodel")
# model = TFTModel.load_from_checkpoint("TFTModel", best=False, work_dir=os.path.join(os.path.dirname(DB_FILEPATH), model_name))

# forecasts = []
# for train, test, p_test, f_test in zip(train_series, test_series, test_past_covar, test_future_covar):
#     forecast = model.predict(series=train, n=len(test), past_covariates=p_test, future_covariates=f_test)
#     forecasts.append(forecast)
#     test.plot()
#     forecast.plot()
#     plt.show


# print(f"EVAL RESULTS FOR {model_name}:")
# average_scores=[0,0,0]
# for site, actual, forecast in zip(site_names, test_series, forecasts):
#     forecast = target_scaler.inverse_transform(forecast)
#     actual  = target_scaler.inverse_transform(test)
#     average_scores[0] += rmse(actual, forecast)
#     average_scores[1] += mae(actual, forecast)
#     average_scores[2] += r2_score(actual, forecast)
#     print(f"{site}: RMSE={rmse(actual, forecast):.3f}, MAE={mae(actual, forecast):.3f}, r2={r2_score(actual, forecast)}")
# average_scores = [ i / len(site_names) for i in average_scores]
# model_average_scores.update({model_name:average_scores})
# print(average_scores)

# print(temporal_robustness(model, series_scaled_list, past_covar_scaled_list, future_covar_scaled_list))
# brittle_test(model, train_series, past_covar_scaled_list, future_covar_scaled_list)

# scores = []
# for site_name, s in zip(site_names, series_scaled_list):
#         score = model.backtest(
#             series=s,
#             start=0.8,
#             forecast_horizon=6,
#             stride=6,
#             metric=[rmse,mae,r2_score],
#             verbose=False,
#             overlap_end=False,
#             retrain=False
#         )
#         scores.append((site_name, score))
#         print(f"{site_name}: RMSE={score:.3f}, mae={score[1]:.3f}, r2={score[2]:.3f}")





# Probabilistic RNN
model_name = "RNNModelSearched"
model = RNNModel(
    input_chunk_length = 24,
    model="LSTM",
    hidden_dim=25,
    n_rnn_layers=1,
    dropout=0.01,
    training_length=24,
    n_epochs = 5,
    likelihood=QuantileRegression([0.1, 0.5, 0.9]),
    random_state = SEED,
    force_reset = True,
    save_checkpoints=True,
    model_name = model_name,
    work_dir = os.path.join(os.path.dirname(DB_FILEPATH), model_name),
    add_encoders={
        'datetime_attribute': {
            'past': ['hour', 'dayofyear'],
            'future': ['hour' , 'dayofyear']
        },
        'cyclic': {
            'past': ['hour', 'dayofyear'],
            'future': ['hour' , 'dayofyear']
        }
    }
)

# models = []
# for i in range(len(train_series)):
#     chosen_model = RNNModel.gridsearch(series = train_series[i], val_series = test_series[i], future_covariates = future_covar_scaled_list[i],
#         parameters = {
#             "dropout":[0.1, 0.01, 0.001],
#             "n_epochs": [25,50],
#             "n_rnn_layers":[1,2],
#             "input_chunk_length":[24],
#         }, 
#         n_jobs = -1,
#         metric = r2_score
#     )

#     models.append(chosen_model)
#     print(chosen_model)
# print(models)

model.fit(train_series, future_covariates=train_future_covar, verbose=True)

model.save(model_name)
#model = RNNModel.load(model_name)
model = RNNModel.load_from_checkpoint(model_name, best = False, work_dir = os.path.join(os.path.dirname(DB_FILEPATH), model_name))

forecasts = []
for train, test, p_test, f_test in zip(train_series, test_series, test_past_covar, test_future_covar):
    forecast = model.predict(series=train, n=len(test), future_covariates=f_test, verbose = True)
    forecasts.append(forecast)

averages = [0,0,0]
for series, future in zip(series_scaled_list, future_covar_scaled_list):
    preds = model.historical_forecasts(
    series=series,
    future_covariates=future,
    start=0.2,                # start validation at 80% of your data
    forecast_horizon=12,      # predict 12 hours ahead each step
    stride=12,                # move 12 steps forward each iteration
    retrain=False,            # skip retraining at each step (faster)
    verbose=True)

    y_true = target_scaler.inverse_transform(series)
    y_pred = target_scaler.inverse_transform(preds)

    # Calculate metrics
    print("MAE:", mae(y_true, y_pred))
    print("RMSE:", rmse(y_true, y_pred))
    print("r2:", r2_score(y_true, y_pred))

    averages[0] += mae(y_true, y_pred)
    averages[1] += rmse(y_true, y_pred)
    averages[2] += r2_score(y_true, y_pred)

    # Visualize
    # import matplotlib.pyplot as plt
    # series.plot(label="Actual")
    # preds.plot(label="Predicted")
    # plt.title("RNNModel Validation (Historical Forecasts)")
    # plt.legend()
    # plt.show()

print(averages[0]/len(series_scaled_list))
print(averages[1]/len(series_scaled_list))
print(averages[2]/len(series_scaled_list))

    

# print(f"EVAL RESULTS FOR {model_name}:")
# average_scores=[0,0,0]
# for site, actual, forecast in zip(site_names, test_series, forecasts):
#     forecast = target_scaler.inverse_transform(forecast)
#     actual  = target_scaler.inverse_transform(test)
#     average_scores[0] += rmse(actual, forecast)
#     average_scores[1] += mae(actual, forecast)
#     average_scores[2] += r2_score(actual, forecast)
#     print(f"{site}: RMSE={rmse(actual, forecast):.3f}, MAE={mae(actual, forecast):.3f}, r2={r2_score(actual, forecast)}")

# # #     plt.figure(figsize=(10, 6))
# # #     actual.plot(label="Actual", lw=2)
# # #     forecast.plot(label="Forecast", lw=2, color='orange')
# # #     plt.title("Actual vs Predicted (RNNModel Forecast)")
# # #     plt.legend()
# # #     plt.show()

# average_scores = [ i / len(site_names) for i in average_scores]
# model_average_scores.update({model_name:average_scores})
# print(average_scores)

#print(temporal_robustness(model, series_scaled_list, None, future_covar_scaled_list))



# Run rolling forecast validation


# Inverse scale predictions


   

# scores = []
# for site_name, s,p,f in zip(site_names, series_scaled_list,past_covar_scaled_list, future_covar_scaled_list):
#         score = model.backtest(
#             series=s,
#             start=0.8,
#             forecast_horizon=6,
#             stride=6,
#             metric=[rmse,mae,r2_score],
#             verbose=False,
#             overlap_end=False,
#             retrain = False
#         )
#         scores.append((site_name, score))
#         print(f"{site_name}: RMSE={score:.3f}, mae={score[1]:.3f}, r2={score[2]:.3f}")








# TS Mixer
# model_name="TSMixerModel"
# # model = TSMixerModel(
# #     input_chunk_length = 48,
# #     output_chunk_length = 12,
# #     hidden_size=64,
# #     dropout=0.1,
# #     n_epochs=100,
# #     activation = "ReLU",
# #     random_state = SEED,
# #     force_reset = True,
# #     save_checkpoints=True,
# #     model_name = model_name,
# #     work_dir = os.path.join(os.path.dirname(DB_FILEPATH), model_name),
# #     add_encoders={
# #         'datetime_attribute': {
# #             'past': ['hour', 'dayofyear'],
# #             'future': ['hour' , 'dayofyear']
# #         },
# #         'cyclic': {
# #             'past': ['hour', 'dayofyear'],
# #             'future': ['hour' , 'dayofyear']
# #         }
# #     }
# # )

# # model.fit(train_series, past_covariates=train_past_covar, future_covariates=train_future_covar, verbose=True)

# # model.save(model_name)
# # model = TSMixerModel.load(model_name)
# model = TSMixerModel.load_from_checkpoint(model_name, best = False, work_dir = os.path.join(os.path.dirname(DB_FILEPATH), model_name))

# forecasts = []
# for train, test, p_test, f_test in zip(train_series, test_series, test_past_covar, test_future_covar):
#     forecast = model.predict(series=train, n=len(test), past_covariates=p_test, future_covariates=f_test, verbose = True)
#     forecasts.append(forecast)
#     # Invert scaling if needed
#     actual = target_scaler.inverse_transform(test)
#     forecast = target_scaler.inverse_transform(forecast)

#     # Plot both
#     # plt.figure(figsize=(10, 6))
#     # actual.plot(label="Actual", lw=2)
#     # forecast.plot(label="Forecast", lw=2, color='orange')
#     # plt.title("Actual vs Predicted (TSMixer Forecast)")
#     # plt.legend()
#     # plt.show()

# print(f"EVAL RESULTS FOR {model_name}:")
# average_scores=[0,0,0]
# for site, actual, forecast in zip(site_names, test_series, forecasts):
#     forecast = target_scaler.inverse_transform(forecast)
#     actual  = target_scaler.inverse_transform(test)
#     average_scores[0] += rmse(actual, forecast)
#     average_scores[1] += mae(actual, forecast)
#     average_scores[2] += r2_score(actual, forecast)
#     print(f"{site}: RMSE={rmse(actual, forecast):.3f}, MAE={mae(actual, forecast):.3f}, r2={r2_score(actual, forecast)}")
# average_scores = [ i / len(site_names) for i in average_scores]
# model_average_scores.update({model_name:average_scores})
# print(average_scores)



# print(temporal_robustness(model, series_scaled_list, past_covar_scaled_list, future_covar_scaled_list))

# scores = []
# for site_name, s, p, f in zip(site_names, series_scaled_list, past_covar_scaled_list, future_covar_scaled_list):
#         score = model.backtest(
#             series=s,
#             start=0.8,
#             forecast_horizon=6,
#             past_covariates = p,
#             future_covariates = f,
#             stride=6,
#             metric=[rmse,mae,r2_score],
#             verbose=False,
#             overlap_end=False,
#             retrain = False
#         )
#         scores.append((site_name, score))
        
#         print(f"{site_name}: RMSE={score[0]:.3f}, mae={score[1]:.3f}, r2={score[2]:.3f}")

print(model_average_scores)