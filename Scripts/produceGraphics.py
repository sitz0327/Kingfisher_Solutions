import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from tabulate import tabulate
import itertools

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

from pytorch_lightning.callbacks import Callback

import keras_tuner

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor

DB_FILEPATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data", "DBcleaned.csv")
OUTPUT = 'kWh Normalised'
SEED = 510444756


class LossRecorder(Callback):
    def __init__(self):
        self.train_loss_history = []
        self.val_loss_history = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_loss_history.append(trainer.callback_metrics["train_loss"].item())
        self.val_loss_history.append(trainer.callback_metrics["val_loss"].item())


df = pd.read_csv(DB_FILEPATH)

X = df.select_dtypes(include=['float64', 'int64']).drop(columns=[OUTPUT, 'kWh'])
y = df[OUTPUT]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

## dimension reduction df
dr = Nystroem(kernel='rbf',gamma = 0.0002, n_components=10, random_state=510444756, n_jobs=-1)
X_nystroem = dr.fit_transform(X_scaled)

dr_X_train, dr_X_test, dr_y_train, dr_y_test = train_test_split(X_nystroem, y, test_size=0.2, random_state=42)


# #All Variables:
# #['name', 'kWh', 'datetime', 'Temperature', 'Alpha', 'Aerosol Optical Depth', 'Asymmetry', 'Clearsky DHI', 'Clearsky DNI','Clearsky GHI', 'Dew Point', 'DHI', 'DNI', 'GHI', 'Ozone','Relative Humidity', 'Solar Zenith Angle', 'SSA', 'Surface Albedo','Pressure', 'Precipitable Water', 'Wind Direction', 'Wind Speed','Global Horizontal UV Irradiance (280-400nm)', 'Global Horizontal UV Irradiance (295-385nm)', 'Cloud Type_0.0', 'Cloud Type_1.0', 'Cloud Type_3.0', 'Cloud Type_4.0', 'Cloud Type_5.0','Cloud Type_6.0', 'Cloud Type_7.0', 'Cloud Type_8.0', 'Cloud Type_9.0', 'Cloud Type_nan', 'day_sine', 'day_cosine', 'year_sine', 'year_cosine','kWh Normalised']
# Selected variables
X = df[['Global Horizontal UV Irradiance (280-400nm)', 'Global Horizontal UV Irradiance (295-385nm)', 'GHI', 'Relative Humidity', 'Temperature', 'Pressure', 'Clearsky GHI', 'Precipitable Water', 'Clearsky DHI', 'Solar Zenith Angle']] # 'day_cosine', 'year_cosine',
y = df[['kWh Normalised']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

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
val_series = []
val_past_covar = []
val_future_covar = []

for s, p, f in zip(series_scaled_list, past_covar_scaled_list, future_covar_scaled_list):
    train, test = s.split_before(0.7)
    p_train, p_test = p.split_before(0.7)
    f_train, f_test = f.split_before(0.7)
    val, test = test.split_before(0.1)
    p_val, p_test = p_test.split_before(0.1)
    f_val, f_test = f_test.split_before(0.1)
 

    train_series.append(train)
    test_series.append(test)
    train_past_covar.append(p_train)
    test_past_covar.append(p_test)
    train_future_covar.append(f_train)
    test_future_covar.append(f_test)
    val_series.append(val)
    val_past_covar.append(p_val)
    val_future_covar.append(f_val)



def train_and_plot(model_kwargs, train_series,train_future_covar,test_series, test_future_covar,run_name):
    loss_recorder = LossRecorder()

    model = RNNModel(
    input_chunk_length = model_kwargs["inl"],
    model="LSTM",
    hidden_dim=model_kwargs["hidden_dim"],
    n_rnn_layers=1,
    dropout=model_kwargs["dropout"],
    training_length=24,
    n_epochs = model_kwargs["epochs"],
    optimizer_kwargs = {'lr': 1e-4},
    likelihood=QuantileRegression([0.1, 0.5, 0.9]),
    random_state = SEED,
    force_reset = True,
    save_checkpoints=True,
    model_name = run_name,
    work_dir = os.path.join(os.path.dirname(DB_FILEPATH), run_name),
    add_encoders={
        'datetime_attribute': {
            'past': ['hour', 'dayofyear'],
            'future': ['hour' , 'dayofyear']
        },
        'cyclic': {
            'past': ['hour', 'dayofyear'],
            'future': ['hour' , 'dayofyear']
        }
    },
    pl_trainer_kwargs = {"callbacks": [loss_recorder]}
)
    
    model.fit(train_series, future_covariates=train_future_covar,val_series = test_series, val_future_covariates = test_future_covar, verbose=True)

    ## train loss curve
    loss = loss_recorder.train_loss_history
    train_loss = loss_recorder.val_loss_history
    plt.figure(figsize=(8, 5))
    plt.plot(loss, label='Training Loss', linewidth=2)
    plt.plot(train_loss,label='Validation Loss', linewidth=2 )
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join((os.path.join(os.path.dirname(DB_FILEPATH), "plots")), f"{run_name}.png")
    plt.savefig(save_path)
    plt.close() 

    return loss_recorder


# model_name = "RNNModel"

# model = RNNModel(
#     input_chunk_length = 24,
#     model="LSTM",
#     hidden_dim=25,
#     n_rnn_layers=1,
#     dropout=0.01,
#     training_length=24,
#     n_epochs = 5,
#     likelihood=QuantileRegression([0.1, 0.5, 0.9]),
#     random_state = SEED,
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
#     },
#     pl_trainer_kwargs = {"callbacks": [loss_recorder]}
# )

# model.fit(train_series, future_covariates=train_future_covar,val_series = test_series, val_future_covariates = test_future_covar, verbose=True)

# model = RNNModel.load_from_checkpoint(model_name, best = False, work_dir = os.path.join(os.path.dirname(DB_FILEPATH), model_name))

## generate search
# input_lengths = [24, 48, 12]
# hiddens = [25, 50, 100]
# dropouts = [0.1, 0.01, 0.001]
input_lengths = [24,12]
hiddens = [25,50]
dropouts = [0.1,0.01]
epochs = [25]
param_grid = [
    {"inl": inl, "hidden_dim": h, "dropout": d, "epochs":e}
    for inl, h, d,e  in itertools.product(input_lengths, hiddens, dropouts, epochs)
]

results = []

for i, params in enumerate(param_grid):
    run_name = f"run_{i+1}_inl{params['inl']}_hd{params['hidden_dim']}_do{params['dropout']}_ep{params['epochs']}"
    print(f"=== {run_name} ===")
    
    history = train_and_plot(params, train_series,train_future_covar,val_series, val_future_covar, run_name)
    
    final_train_loss = history.train_loss_history[-1]
    final_val_loss = history.val_loss_history[-1]
    
    results.append({
        "params": params,
        "train_loss": final_train_loss,
        "val_loss": final_val_loss
    })

df = pd.DataFrame(results)
df.sort_values("val_loss", inplace=True)
df.to_csv(os.path.join((os.path.join(os.path.dirname(DB_FILEPATH), "plots")), "results.csv"))



