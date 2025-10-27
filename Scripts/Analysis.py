
import pandas as pd
import matplotlib.pyplot as plt
import os
import itertools
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from sklearn.linear_model import SGDRegressor

from sktime.forecasting.compose import make_reduction
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.ets import AutoETS

from darts import TimeSeries
from darts.models import TFTModel


DB_FILEPATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data", "DBcleaned.csv")
OUTPUT = 'kWh Normalised'

## Parameters:

regression_models = {
    'Lin_Reg': LinearRegression,
    'LGBM': LGBMRegressor
}

statistical_models = {
    'AutoARIMA': AutoARIMA,
    'AutoETS': AutoETS
}

metrics = {
    'R2': r2_score,
    'MAE': mean_absolute_error,
    'MSE': mean_squared_error
}

def get_evaluation_results(predictions):
    models = [model_name for model_name in hyperparameter_result.columns if model_name != 'y_true']

    metrics_frame = []
    for metric_name in metrics.keys():
        for model in models:
            metrics_frame.append(pd.DataFrame({
                'Metric': [metric_name],
                'Model': [model],
                'Score': [metrics[metric_name](hyperparameter_result['y_true'], hyperparameter_result[model])]
            }))

    metrics_frame = pd.concat(metrics_frame)
    
    return metrics_frame



#load Data
df = pd.read_csv(DB_FILEPATH)

#split data
X = df.drop(columns=['kWh','kWh Normalised', 'datetime', 'name'])
y = df['kWh Normalised']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



for site, site_df in df.groupby("name"):
    site_X = site_df.drop(columns=['kWh', 'kWh Normalised'])
    site_y = site_df['kWh Normalised']
    X_train, X_test, y_train, y_test = temporal_train_test_split(site_y, site_X, test_size=0.2)


# #SGD regressor
# sgd_model = SGDRegressor(
#         loss="squared_error",              # Robust to outliers
#         penalty="elasticnet",      # Mix of L1/L2 regularization
#         alpha=0.0001,              # Regularization strength
#         max_iter=1000000000,
#         tol=1e-3,
#         random_state=43,
#         shuffle=True,
#         learning_rate="optimal"
#     )

# # Train the model
# sgd_model.fit(X_train, y_train)
# print(sgd_model.feature_names_in_)

# # Predictions
# y_pred = sgd_model.predict(X_test)

# # Evaluate
# r2 =  r2_score(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# print("R² Score:",r2)
# print("MAE:", mae)

# #visualise
# plt.figure(figsize=(7,7))
# sns.scatterplot(x=y_test, y=y_pred, alpha=0.3)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
# plt.title(f"SGD Regressor: Actual vs Predicted\nR²={r2:.3f}, MAE={mae:.3f}")
# plt.xlabel("Actual Normalised kWh")
# plt.ylabel("Predicted Normalised kWh")
# plt.grid(True)
# plt.show()


# residuals = y_test - y_pred
# plt.figure(figsize=(7,5))
# sns.histplot(residuals, bins=50, kde=True, color='skyblue')
# plt.title("Residual Distribution (y_true - y_pred)")
# plt.xlabel("Residual")
# plt.ylabel("Frequency")
# plt.grid(True)
# plt.show()



# # SVR
# svr = SVR(kernel='linear', C=100, gamma=0.1, epsilon=0.01)
# svr.fit(X_train_scaled, y_train)

# y_pred = svr.predict(X_test_scaled)

# print("R²:", r2_score(y_test, y_pred))
# print("MAE:", mean_absolute_error(y_test, y_pred))


# darts timeseries

from darts import TimeSeries
from darts.models import TFTModel
from darts.utils.timeseries_generation import datetime_attribute_timeseries

# Suppose df is your data
series_list = []
for site, site_df in df.groupby("name"):
    site_ts = TimeSeries.from_dataframe(
        site_df,
        freq = 'h',
        time_col="datetime",
        value_cols=["kWh Normalised", 'Temperature', 'Alpha', 'Aerosol Optical Depth',  'Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI',  'DHI', 'DNI', 'GHI','Solar Zenith Angle', 'Wind Speed', 'Global Horizontal UV Irradiance (280-400nm)', 'Cloud Type_1.0', 'Cloud Type_3.0', 'Cloud Type_4.0', 'Cloud Type_5.0', 'Cloud Type_6.0', 'Cloud Type_7.0' ,'Cloud Type_8.0', 'Cloud Type_9.0', 'Cloud Type_nan']
    )
    series_list.append(site_ts)

# Use a global model (e.g., TFT, RNN, or NBEATS)
model = TFTModel(
    input_chunk_length=24,
    output_chunk_length=6,
    n_epochs=50,
    random_state=42,
    add_relative_index = True,
    # pl_trainer_kwargs={
    #     "accelerator":'gpu'
    # }
)

# Train across all sites at once (global model)
model.fit(series_list, verbose=True)

# Predict for one site
forecast = model.predict(n=12, series=series_list[0])
forecast.plot()
