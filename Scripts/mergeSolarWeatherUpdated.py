import pandas as pd
from pathlib import Path

# === FILE PATHS ===
solar_path = Path(r"\Kingfisher_Solutions\Data\Solar Outut By Site\Calgary Fire Hall Headquarters\Calgary Fire Hall Headquarters-SOLAR-DATA.csv")
weather_path = Path(r"\Kingfisher_Solutions\Data\Solar Outut By Site\Calgary Fire Hall Headquarters\Calgary Fire Hall Headquarters_combined.csv")

# === LOAD SOLAR DATA ===
solar = pd.read_csv(solar_path)
solar.columns = solar.columns.str.strip()
solar['datetime'] = pd.to_datetime(solar['date'], format='%d/%m/%Y %H:%M', errors='coerce')
solar = solar.dropna(subset=['datetime'])

# === LOAD WEATHER DATA ===
weather = pd.read_csv(weather_path)
weather.columns = weather.columns.str.strip()

# Combine date columns into a single datetime
weather['datetime'] = pd.to_datetime(weather[['Year', 'Month', 'Day', 'Hour', 'Minute']], errors='coerce')
weather = weather.dropna(subset=['datetime'])

# Drop the old Year/Month/Day/Hour/Minute columns
weather = weather.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute'])

# === ROUND WEATHER TIMES UP TO NEAREST HOUR ===
weather['datetime'] = weather['datetime'].dt.ceil('H')

# === MERGE DATASETS ===
# We'll use "merge_asof" to attach the nearest *next* weather reading to each solar measurement
merged = pd.merge_asof(
    solar.sort_values('datetime'),
    weather.sort_values('datetime'),
    on='datetime',
    direction='forward'  # gives the next available weather reading (rounded up)
)

# === SAVE OUTPUT ===
output_path = solar_path.parent / "Calgary Fire Hall Headquarters_merged_solar_weather.csv"
merged.to_csv(output_path, index=False)

print(f"✅ Merged data saved to:\n{output_path}")
print(f"Total merged rows: {len(merged)}")
