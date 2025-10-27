import tensorflow as tf
import keras_tuner as kt
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DB_FILEPATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data", "DBcleaned.csv")
OUTPUT = 'kWh Normalised'
SEED = 510444756

df = pd.read_csv(DB_FILEPATH)
print(df.columns)

X = df[['GHI', 'day_cosine', 'Dew Point', 'year_cosine', 'Relative Humidity', 'Temperature', 'Pressure', 'Clearsky GHI', 'Precipitable Water', 'Clearsky DHI']]
y = df[['kWh Normalised']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=SEED)

def model_builder(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(10,)))  # 11 features

    # Tune the number of layers and units
    for i in range(hp.Int('num_layers', 2, 4)):
        units = hp.Int(f'units_{i}', min_value=32, max_value=256, step=32)
        model.add(tf.keras.layers.Dense(units, activation='relu'))
        model.add(tf.keras.layers.Dropout(hp.Float(f'dropout_{i}', 0.1, 0.5, step=0.1)))

    model.add(tf.keras.layers.Dense(1))  # Output layer

    # Tune the learning rate
    lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='mse',
                  metrics=['mae'])
    return model

tuner = kt.RandomSearch(
    model_builder,
    objective='val_mae',
    max_trials=20,
    executions_per_trial=2,
    directory='kt_dir',
    project_name='kwh_prediction'
)

tuner.search(X_train, y_train,
             validation_split=0.2,
             epochs=100,
             batch_size=32,
             callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)])


best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best learning rate: {best_hps.get('learning_rate')}")
print(f"Best number of layers: {best_hps.get('num_layers')}")

model = tuner.hypermodel.build(best_hps)
model.fit(X_train, y_train, validation_split=0.2, epochs=100)
