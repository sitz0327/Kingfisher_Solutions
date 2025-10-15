# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 15:39:46 2025

@author: seani
"""

import pandas as pd
from pathlib import Path
import os
import sys

# Set pandas display options for clean console output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# === CONFIGURATION ===
# Set the base directory containing all site folders
BASE_DIR = Path(r"C:\Users\seani\Documents\anaconda_projects\Kingfisher_Solutions\Data\Solar Outut By Site")

# List of site folders to explicitly skip (e.g., the one you already completed)
EXCLUDE_SITES = [
    "Calgary Fire Hall Headquarters"
]

# === MERGING FUNCTION ===
def process_site(site_path: Path):
    """
    Processes a single site folder, attempts to find the solar and weather files,
    performs the merge, and saves the resulting CSV.
    """
    site_name = site_path.name
    print(f"\n--- Processing Site: {site_name} ---")

    # 1. Find required files
    # Solar data file (e.g., 'Calgary Fire Hall Headquarters-SOLAR-DATA.csv')
    solar_files = list(site_path.glob('*-SOLAR-DATA.csv'))

    # Weather data file (e.g., 'Calgary Fire Hall Headquarters_combined.csv')
    weather_files = list(site_path.glob('*_combined.csv'))

    if not solar_files:
        print(f"   ⚠️ SKIPPED: No *-SOLAR-DATA.csv found.")
        return
    if not weather_files:
        print(f"   ⚠️ SKIPPED: No *_combined.csv found.")
        return

    solar_path = solar_files[0]
    weather_path = weather_files[0]

    print(f"   🌞 Solar: {solar_path.name}")
    print(f"   ☁️ Weather: {weather_path.name}")
    
    try:
        # --- LOAD SOLAR DATA ---
        solar = pd.read_csv(solar_path)
        solar.columns = solar.columns.str.strip()
        
        # Original format from the user's initial script
        solar['datetime'] = pd.to_datetime(solar['date'], format='%d/%m/%Y %H:%M', errors='coerce')
        solar = solar.dropna(subset=['datetime'])
        
        if solar.empty:
            print("   ❌ FAILED: Solar data is empty or datetime conversion failed.")
            return

        # --- LOAD WEATHER DATA ---
        weather = pd.read_csv(weather_path)
        weather.columns = weather.columns.str.strip()

        # Combine date columns into a single datetime
        weather['datetime'] = pd.to_datetime(weather[['Year', 'Month', 'Day', 'Hour', 'Minute']], errors='coerce')
        weather = weather.dropna(subset=['datetime'])
        
        if weather.empty:
            print("   ❌ FAILED: Weather data is empty or datetime conversion failed.")
            return

        # Drop the old Year/Month/Day/Hour/Minute columns
        weather = weather.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute'])

        # === ROUND WEATHER TIMES UP TO NEAREST HOUR ===
        # This aligns the continuous weather data with the discrete solar events
        weather['datetime'] = weather['datetime'].dt.ceil('H')

        # === MERGE DATASETS ===
        # Sort both DataFrames before using merge_asof (required)
        merged = pd.merge_asof(
            solar.sort_values('datetime'),
            weather.sort_values('datetime'),
            on='datetime',
            direction='forward'  # Attach the nearest *next* weather reading
        )

        # === SAVE OUTPUT ===
        output_filename = f"{site_name}_merged_solar_weather.csv"
        output_path = site_path / output_filename
        merged.to_csv(output_path, index=False)

        print(f"   ✅ SUCCESS! Merged {len(merged)} rows.")
        print(f"   Saved to: {output_path.name}")

    except pd.errors.ParserError as e:
        print(f"   ❌ FAILED: Parsing error encountered. Check CSV formatting. Error: {e}")
    except KeyError as e:
        print(f"   ❌ FAILED: Missing required column {e}. Check input file headers.")
    except Exception as e:
        print(f"   ❌ FAILED: An unexpected error occurred. Error: {e}")


# === MAIN EXECUTION BLOCK ===
if __name__ == "__main__":
    if not BASE_DIR.is_dir():
        print(f"Error: Base directory not found or is not a directory: {BASE_DIR}")
        sys.exit(1)

    print(f"Starting recursive data merge from base directory: {BASE_DIR}")
    print(f"Excluding sites: {EXCLUDE_SITES}")
    print("-" * 50)

    site_folders = [p for p in BASE_DIR.iterdir() if p.is_dir()]
    processed_count = 0
    
    for site_path in site_folders:
        site_name = site_path.name

        # 1. Exclusion Rule: Skip sites explicitly listed
        if site_name in EXCLUDE_SITES:
            print(f"\n--- Site: {site_name} ---")
            print("   ➡️ SKIPPED: Site is in the EXCLUDE_SITES list.")
            continue

        # 2. Minimum File Count Rule: Skip folders with 1 or less files
        # This prevents processing empty or incomplete folders
        try:
            file_count = len(list(p for p in site_path.iterdir() if p.is_file()))
            if file_count <= 1:
                print(f"\n--- Site: {site_name} ---")
                print(f"   ➡️ SKIPPED: Folder contains only {file_count} file(s). Need at least two.")
                continue
        except Exception as e:
            print(f"   Error checking file count in {site_name}: {e}")
            continue

        # 3. Process the valid site folder
        process_site(site_path)
        processed_count += 1

    print("\n" + "=" * 50)
    print(f"✨ Script finished! Successfully processed {processed_count} site folder(s).")
    print("=" * 50)
