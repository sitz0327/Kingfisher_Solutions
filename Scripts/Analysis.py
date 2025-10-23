
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

DB_FILEPATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data", "DBcleaned.csv")
OUTPUT = 'kWh Normalised'


# Load data
df = pd.read_csv(DB_FILEPATH)

df = df.drop(columns=['name','id','kWh','datetime','maximumPower','month','day','hour','dayOfYear', 'year', 'Cloud Fill Flag', 'Cloud Type', 'Fill Flag'])

# Select numeric features only
X = df.select_dtypes(include=['float64', 'int64']).drop(columns=[OUTPUT])
y = df[OUTPUT]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit regression
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("R²:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))


pca = PCA(n_components=0.95)
pca.fit(X)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA: keep 95% of variance
pca = PCA(n_components=0.95)

X_pca = pca.fit_transform(X_scaled)


loadings = pd.DataFrame(pca.components_.T, 
                        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
                        index=X.columns)
print(loadings.head())

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42
)

strategy = tf.distribute.get_strategy()

# Define the model
with strategy.scope():

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_pca.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(1, activation='linear')  # regression output
    ])

# with strategy.scope():
#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.Dense(10, activation='softmax')
#     ])

#     model.compile(optimizer='adam',
#                     loss='sparse_categorical_crossentropy',
#                     metrics=['accuracy'])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
model.summary()


history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    verbose=1
)


import matplotlib.pyplot as plt

# Plot training/validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training Performance')
plt.show()

# Evaluate
y_pred = model.predict(X_test).flatten()
print("R²:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))