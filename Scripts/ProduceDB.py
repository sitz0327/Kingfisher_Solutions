import pandas as pd
from pathlib import Path
from typing import Optional
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os
import glob

ROOT = script_dir = Path(__file__).resolve().parent.parent


def concat_files():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(script_dir)
    parent_dir = os.path.join(parent, "Data", "NSRDB")
    print(parent_dir)


    for dir in os.listdir(parent_dir):
        input_path = os.path.join(parent_dir, dir) 
        output_path = os.path.join(input_path, f"{dir}_combined.csv")
        
        print(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)
        files = glob.glob(os.path.join(input_path, "*.csv"))

        df = pd.read_csv(files[0], skiprows=2)
        
        i = 0
        for f in files[1:]:
            temp = pd.read_csv(f,skiprows=2)
            df = pd.concat([df,temp], ignore_index=True)
            df.to_csv(os.path.join(output_path), index=False)

        df.to_csv(output_path, index=False)
    return

def split_solar_output_data_by_site():
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


def merge_solar_output_and_weather():
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
        solar['date'] = pd.to_datetime(solar['date'], format='%d/%m/%Y %H:%M')
        solar = solar[solar['date'].dt.minute == 0]
        solar.sort_values('date', inplace=True)
        solar.set_index('date', inplace=True)
        full_range = pd.date_range(solar.index.min(), solar.index.max(), freq='H')
        solar = solar.reindex(full_range)
        solar['kWh'] = solar['kWh'].fillna(0)

        for col in ['name', 'id', 'address', 'public_url', 'installationDate', 'uid']:
            solar[col] = solar[col].fillna(method='ffill')
        #solar = solar.dropna(subset=['datetime'])
        solar = solar.reset_index().rename(columns={'index': 'date'})

        # === LOAD WEATHER DATA ===
        weather = pd.read_csv(site_weather_path)
        weather.columns = weather.columns.str.strip()

        # === Load Installation data ===
        installations = pd.read_csv(installation_path)
        installations = installations[['id', 'maximumPower']]

        # Combine date columns into a single datetime
        weather['date'] = pd.to_datetime(weather[['Year', 'Month', 'Day', 'Hour', 'Minute']], format='%d/%m/%Y %H:%M', errors='coerce')
        weather = weather[weather['date'].dt.minute== 0]
        #weather = weather.dropna(subset=['datetime'])

        # === ROUND WEATHER TIMES UP TO NEAREST HOUR ===
        #weather.set_index('date', inplace=True)
        

        # === MERGE DATASETS ===
        # We'll use "merge_asof" to attach the nearest *next* weather reading to each solar measurement


        merged = pd.merge(
            solar,
            weather,
            on='date',
            how = 'outer'
        )

        
        merged = pd.merge(
            merged,
            installations,
            on='id',
            how = 'left'
        )

        merged.dropna(inplace=True)
        print(merged.columns)

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
    df['datetime'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    # Drop irrelevant or repeated columns
    df.drop(columns=['address', 'public_url', 'installationDate', 'uid'], inplace=True)

    # fix types
    df['kWh'] = (
        df['kWh'].replace(',','',regex=True)
        .astype('float64')
    )

    #onehot encode the cloud type:
    encoder = OneHotEncoder(sparse_output=False, max_categories=15)
    encoded = encoder.fit_transform(df[['Cloud Type']])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['Cloud Type']))

    df = pd.concat([df, encoded_df], axis = 1, ignore_index=False)

    
    
    df['dayOfYear'] = df['datetime'].dt.dayofyear
    print(df.columns)
    print(df.head())


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

    df.drop(columns = ['id','maximumPower','Month','Day','Hour','Minute','dayOfYear', 'Year', 'Cloud Fill Flag', 'Cloud Type', 'Fill Flag', 'Unnamed: 0'], inplace=True)

    df.to_csv(ROOT / "Data" / "DBCleaned.csv", index=False)

    print(df.describe())

 
if __name__ == "__main__":
    concat_files()
    split_solar_output_data_by_site()
    merge_solar_output_and_weather()
    df = merge_db()
    clean_db(df)