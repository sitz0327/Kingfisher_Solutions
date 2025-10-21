import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(script_dir)
DB = os.path.join(parent, "Data", "DB.csv")

df = pd.read_csv(DB, low_memory=False)

# Drop irrelevant or repeated columns
df.drop(columns=['address', 'public_url', 'installationDate', 'uid', 'date'], inplace=True)



# fix types
df['kWh'] = (
    df['kWh'].replace(',','',regex=True)
    .astype('float64')
)

#onehot encode the cloud type:
encoder = OneHotEncoder(sparse_output=False, max_categories=15)
encoded = encoder.fit_transform(df[['Cloud Type']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['Cloud Type']))
print(encoded_df)



df = pd.concat([df, encoded_df], axis = 1, ignore_index=False)

encoded_df.to_csv(os.path.join(parent, "Data", "DBdropped.csv"), index=True)


# get sine/cosine encodings of time of day and day of year
df['datetime'] = pd.to_datetime(df['datetime'])
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
df['dayOfYear'] = df['datetime'].dt.dayofyear
df['year'] = df['datetime'].dt.year

df['day_sine'] = np.sin(2 * np.pi * df['hour'] / 24)
df['day_cosine'] = np.cos(2 * np.pi * df['hour'] / 24)
df['year_sine'] = np.sin(2 * np.pi * df['dayOfYear'] / 365)
df['year_cosine'] = np.cos(2 * np.pi * df['dayOfYear'] / 365)

# dates after 24/10/23 are measured in watt hours not kWh -> divide by 1000 to fix
df.loc[df['datetime'] > pd.to_datetime('24/10/2023 00:00', format='%d/%m/%Y %H:%M'), 'kWh'] *= 0.001

# normalise output by maximum power of site
df['kWh Normalised'] = df['kWh']/df['maximumPower']


# drop dates that have no weather data:
df.dropna(inplace=True)

df.to_csv(os.path.join(parent, "Data", "DBCleaned.csv"), index=False)

print(df.describe())
