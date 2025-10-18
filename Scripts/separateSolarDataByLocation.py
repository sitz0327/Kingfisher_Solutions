# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 14:22:05 2025

@author: seani
"""
import pandas as pd
from pathlib import Path
from typing import Optional


def split_solar_data_by_site(
    input_file_path: Path,
    output_directory: Optional[Path] = None
) -> None:
    """
    Splits solar data from a single CSV into multiple CSVs,
    one for each unique site found in the 'name' column.
    """

    print(f"--- Starting data processing for file: {input_file_path}")

    # --- 1. Setup and Input Validation ---
    if not input_file_path.exists():
        print(f"[ERROR]: Input file not found at '{input_file_path}'. Please check the location.")
        return

    # Define default output directory if not provided
    if output_directory is None:
        output_directory = input_file_path.parent / "SplitSites"

    output_directory.mkdir(exist_ok=True)
    print(f"Output directory: {output_directory}")

    # --- 2. Read CSV and Error Handling ---
    try:
        df = pd.read_csv(input_file_path)
    except pd.errors.EmptyDataError:
        print("[ERROR]: The input file is empty.")
        return
    except Exception as e:
        print(f"[ERROR]: Reading CSV file failed: {e}")
        return

    # Clean column names and the 'name' field
    df.columns = df.columns.str.strip()
    if "name" not in df.columns:
        print("[ERROR]: The input file must contain a column named 'name' for grouping.")
        print(f"Found columns: {list(df.columns)}")
        return

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


if __name__ == "__main__":
    # --- Define paths relative to the script location ---
    script_dir = Path(__file__).resolve().parent.parent
    input_file = script_dir / "Data" / "Solar_Energy_Production_20251008.csv"     # Example: ./data/solar_data.csv
    output_dir = script_dir / "Data"/ "SplitSites"                 # Output folder: ./split_sites

    split_solar_data_by_site(input_file_path=input_file, output_directory=output_dir)
