import pandas as pd
from pathlib import Path
import os
import glob

# === FILE PATHS ===
script_dir = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(script_dir)
weather_path = os.path.join(parent, "Data", "NSRDB")
solar_path = os.path.join(parent, "Data", "SplitSites")
installation_path = os.path.join(parent, "Data", "Solar_Photovoltaic_Sites_20250925.csv")
output_path = os.path.join(parent, "Data", "MergedSiteData")

#solar_path = Path(r"\Kingfisher_Solutions\Data\SplitSites")
#weather_path = Path(r"\Kingfisher_Solutions\Data\NSRDB")
#output_path = Path(r"\Kingfisher_Solutions\Data\MergedSiteData")

all_sites = glob.glob(os.path.join(weather_path, "*"))
print(all_sites)

# loop through all sites and merge with corresponding data
for site in all_sites:
    name = os.path.basename(site)
    site_solar_path = os.path.join(solar_path, f"{name}.csv")
    site_weather_path = os.path.join(weather_path, name, f"{name}_combined.csv")
    site_output_path = os.path.join(output_path, f"{name}.csv")


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

    # Drop the old Year/Month/Day/Hour/Minute columns
    weather = weather.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute'])

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

print(f"✅ Merged data saved to:\n{output_path}")
print(f"Total merged rows: {len(merged)}")
