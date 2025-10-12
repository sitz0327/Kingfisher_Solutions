# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 14:22:05 2025

@author: seani
"""
import pandas as pd
import os
from typing import Optional

def split_solar_data_by_site(input_file_path: str, output_directory: Optional[str] = "split_sites") -> None:
    """
    Splits solar data from a single CSV into multiple CSVs,
    one for each unique site found in the 'name' column.
    """
    print(f"--- Starting data processing for file: {input_file_path}")

    # --- 1. Setup and Input Validation ---
    if not os.path.exists(input_file_path):
        print(f"[ERROR]: Input file not found at '{input_file_path}'. Please check the location.")
        return

    os.makedirs(output_directory, exist_ok=True)

    # --- 2. Read CSV and Error Handling ---
    try:
        df = pd.read_csv(input_file_path)
    except pd.errors.EmptyDataError:
        print("[ERROR]: The input file is empty.")
        return
    except Exception as e:
        print(f"[ERROR]: Reading CSV file failed: {e}")
        return

    # Clean column names by stripping whitespace
    df.columns = df.columns.str.strip()

    # Verify the essential grouping column exists
    if "name" not in df.columns:
        print("[ERROR]: The input file must contain a column named 'name' for grouping.")
        print(f"Found columns: {list(df.columns)}")
        return

    processed_count = 0
    total_groups = len(df["name"].unique())
    print(f"Found {total_groups} unique sites to process.")

    # --- 3. Split and Save Data ---
    for site_name, site_data in df.groupby("name"):
        # Create a safe filename by filtering out special characters
        safe_name = "".join(c for c in site_name if c.isalnum() or c in (' ', '_', '-')).strip()

        if not safe_name:
            print(f"[Warning]: Skipping a group with an unparsable site name: '{site_name}'")
            continue

        filename = f"{safe_name}-SOLAR-DATA.csv"
        filepath = os.path.join(output_directory, filename)

        # Write site-specific data to a new CSV file
        site_data.to_csv(filepath, index=False)
        print(f"Created ({processed_count + 1}/{total_groups}): {filepath}")
        processed_count += 1

    print(f"\n--- Done! Processed {processed_count} site files in '{output_directory}'.")


if __name__ == "__main__":
    # Prompt user for the input file path
    file_path = input("Enter the path to the solar data CSV file: ").strip()
    
    if file_path:
        split_solar_data_by_site(input_file_path=file_path)
    else:
        print("[ERROR]: No file path provided. Exiting.")
