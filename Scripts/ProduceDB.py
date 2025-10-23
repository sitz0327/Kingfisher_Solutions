import pandas as pd
from pathlib import Path
from typing import Optional
from sklearn.preprocessing import OneHotEncoder
import numpy as np

ROOT = script_dir = Path(__file__).resolve().parent.parent


def split_solar_data_by_site():
    """
    Splits solar data from a single CSV into multiple CSVs,
    one for each unique site found in the 'name' column.
    """
    input_file_path = ROOT/"Data"/"Solar_Energy_Production_20251008.csv"
    output_directory = ROOT/"Data"/"SplitSites"

    print("Splitting Solar data by site")
    print(f"--- Starting data processing for file: {input_file_path}")

    # --- 2. Read CSV and Error Handling ---
    df = pd.read_csv(input_file_path)
    # Clean column names and the 'name' field
    df.columns = df.columns.str.strip()

    df["name"] = df["name"].astype(str).str.strip()

    processed_count = 0
    total_groups = len(df["name"].unique())
    print(f"Found {total_groups} unique sites to process.\n")

    # --- 3. Split and Save Data ---
    for site_name, site_data in df.groupby("name"):
        filepath = output_directory / f"{site_name}.csv"
        site_data.to_csv(filepath, index=False)
        processed_count += 1
        print(f"Created ({processed_count}/{total_groups}): {filepath.name}")

    print(f"\n--- Done! Processed {processed_count} site files in '{output_directory}'.")


def merge_solar_sutput_and_weather():
    weather_path = ROOT / "Data" /"NSRDB"
    solar_path = ROOT / "Data" / "SplitSites"
    installation_path = ROOT / "Data" / "Solar_Photovoltaic_Sites_20250925.csv"
    output_path = ROOT / "Data" / "MergedSiteData"

    all_sites = list(Path(weather_path).glob("*"))

    for site in all_sites:
        name = site.name
        print(f"--- Starting data processing for: {name}")

        site_solar_path = solar_path / f"{name}.csv"
        site_weather_path = weather_path / name / f"{name}_combined.csv"
        site_output_path = output_path / f"{name}.csv"

        # === LOAD SOLAR DATA ===
        solar = pd.read_csv(site_solar_path)
        solar.columns = solar.columns.str.strip()
        solar['datetime'] = pd.to_datetime(solar['date'], format='%d/%m/%Y %H:%M', errors='coerce')
        solar = solar.dropna(subset=['datetime'])

        # === LOAD WEATHER DATA ===
        weather = pd.read_csv(site_weather_path)
        weather.columns = weather.columns.str.strip()

        # === Load Installation data ===
        installations = pd.read_csv(installation_path)
        installations = installations[['id', 'maximumPower']]

        # Combine date columns into a single datetime
        weather['datetime'] = pd.to_datetime(weather[['Year', 'Month', 'Day', 'Hour', 'Minute']], errors='coerce')
        weather = weather.dropna(subset=['datetime'])

        # === ROUND WEATHER TIMES UP TO NEAREST HOUR ===
        weather['datetime'] = weather['datetime'].dt.ceil('h')

        # === MERGE DATASETS ===
        # We'll use "merge_asof" to attach the nearest *next* weather reading to each solar measurement
        merged = pd.merge(
            solar,
            weather,
            on='datetime',
            how = 'left'
        )

        merged = pd.merge(
            merged,
            installations,
            on='id',
            how = 'left'
        )

        # === SAVE OUTPUT ===
        merged.to_csv(site_output_path, index=True)
        print(f"Merged data for: {name}")

    print(f"✅ Merged data saved to:\n{output_path}")
    print(f"Total merged rows: {len(merged)}")


def merge_db():
    input_dir = ROOT / "Data" / "MergedSiteData"
    output_path = ROOT / "Data" / "DB.csv"

    file_list = list(Path(input_dir).glob("*.csv"))

    df = pd.read_csv(file_list[0], low_memory=False)

    for f in file_list[1:]:
        print(f"Adding {f.name}")
        temp = pd.read_csv(f, low_memory=False)
        df = pd.concat([df,temp], ignore_index=True)

    df.to_csv(output_path, index = False)
    return df


def clean_db(df):
    # Drop irrelevant or repeated columns
    df.drop(columns=['address', 'public_url', 'installationDate', 'uid', 'date'], inplace=True)

    # fix types
    df['kWh'] = (
        df['kWh'].replace(',','',regex=True)
        .astype('float64')
    )
    df['datetime'] = pd.to_datetime(df['datetime'])

    #onehot encode the cloud type:
    encoder = OneHotEncoder(sparse_output=False, max_categories=15)
    encoded = encoder.fit_transform(df[['Cloud Type']])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['Cloud Type']))
    print(encoded_df)

    df = pd.concat([df, encoded_df], axis = 1, ignore_index=False)

    
    df['dayOfYear'] = df['datetime'].dt.dayofyear

    df['day_sine'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['day_cosine'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['year_sine'] = np.sin(2 * np.pi * df['dayOfYear'] / 365)
    df['year_cosine'] = np.cos(2 * np.pi * df['dayOfYear'] / 365)

    # dates after 24/10/23 are measured in watt hours not kWh -> divide by 1000 to fix
    df.loc[df['datetime'] > pd.to_datetime('24/10/2023 00:00', format='%d/%m/%Y %H:%M'), 'kWh'] *= 0.001

    # normalise output by maximum power of site
    df['kWh Normalised'] = df['kWh']/df['maximumPower']

    # drop dates that have no weather data:
    df.dropna(inplace=True)

    df.drop(columns = ['name','id','maximumPower','Month','Day','Hour','dayOfYear', 'Year', 'Cloud Fill Flag', 'Cloud Type', 'Fill Flag'], inplace=True)

    df.to_csv(ROOT / "Data" / "DBCleaned.csv", index=False)

    print(df.describe())

 
if __name__ == "__main__":
    split_solar_data_by_site()
    merge_solar_sutput_and_weather()
    df = merge_db()
    clean_db(df)