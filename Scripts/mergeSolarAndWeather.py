import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# === CONFIGURATION ===
# Define the path to the merged file
Path(r"\Kingfisher_Solutions\Data\split_sites")

# Identify key columns based on standard weather data structure
SOLAR_OUTPUT_COL = 'kWh'
IRRADIANCE_COL = 'Clearsky GHI' # Assuming 'Clearsky GHI' is the Global Horizontal Irradiance column
TEMPERATURE_COL = 'Temperature' # Assuming 'Temperature' is the ambient temperature column

def analyze_solar_data(file_path: Path):
    """
    Loads the merged data, performs correlation analysis, and generates key visualizations.
    """
    print(f"Loading data from: {file_path.name}")
    
    try:
        # Load data and ensure 'datetime' is the index for time-series plotting
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        
        # Clean datetime column (assuming it's already structured correctly from the merge script)
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime', SOLAR_OUTPUT_COL]).set_index('datetime')
        
        # 1. DATA CLEANING AND PREP
        
        # Filter for actual generation periods to remove night/inactive periods (where kWh == 0)
        # This focuses the analysis on operational hours.
        df_active = df[df[SOLAR_OUTPUT_COL] > 0].copy()
        
        print(f"Total records loaded: {len(df)}. Active generation records: {len(df_active)}")
        
        if df_active.empty:
            print("Error: No active solar generation records found after filtering.")
            return

        # 2. CORRELATION ANALYSIS
        print("\n--- Correlation Analysis: Top Factors affecting kWh ---")
        
        # Select numeric columns for correlation calculation
        numeric_df = df_active.select_dtypes(include=np.number)
        
        # Calculate correlation with Solar Output (kWh)
        correlations = numeric_df.corr()[SOLAR_OUTPUT_COL].sort_values(ascending=False)
        
        # Exclude self-correlation (kWh vs kWh)
        top_correlations = correlations[correlations.index != SOLAR_OUTPUT_COL]
        
        # Show top 5 strongest (absolute value) correlations
        top_5_correlations = top_correlations.abs().sort_values(ascending=False).head(5)
        
        print("Top 5 Variables Highly Correlated with Solar Output (kWh):")
        # Print the corresponding signed correlation values
        for col in top_5_correlations.index:
             print(f"  {col.ljust(20)}: {correlations[col]:.4f}")


        # 3. VISUALIZATIONS
        
        sns.set_style("whitegrid")
        
        # --- Visual 1: Time Series of Solar Output (kWh) ---
        plt.figure(figsize=(18, 6))
        # Plotting the raw data points as requested by user's example (style='.')
        df_active[SOLAR_OUTPUT_COL].plot(
            style='.', 
            color='#F8766D', 
            alpha=0.6, 
            label='Actual Power (kWh)'
        )
        # Add a smoothed line for trend visibility
        df_active[SOLAR_OUTPUT_COL].resample('D').mean().plot(
            color='#1f77b4', 
            linewidth=2, 
            label='Daily Average Trend'
        )
        
        plt.title('Time Series: Solar Power Generated (kWh) - ' + file_path.parent.name, fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Power Generated (kWh)", fontsize=12)
        plt.legend()
        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.show()
        
        
        # --- Visual 2: Scatter Plot: kWh vs. Primary Driver (Irradiance) ---
        plt.figure(figsize=(10, 8))
        # Use jointplot for a scatter plot with marginal histograms for distribution
        sns.jointplot(
            x=IRRADIANCE_COL, 
            y=SOLAR_OUTPUT_COL, 
            data=df_active, 
            kind='scatter', 
            color='#00BA38', 
            alpha=0.7
        )
        plt.suptitle(f'Relationship: Solar Output (kWh) vs. {IRRADIANCE_COL}', y=1.02, fontsize=16)
        plt.xlabel(f"{IRRADIANCE_COL}", fontsize=12)
        plt.ylabel("Power Generated (kWh)", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()


        # --- Visual 3: Scatter Plot: kWh vs. Secondary Factor (Temperature) ---
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=TEMPERATURE_COL, 
            y=SOLAR_OUTPUT_COL, 
            data=df_active, 
            hue='Cloud Type', # Optional: use another variable for color insight
            palette='viridis', 
            size=SOLAR_OUTPUT_COL, # Size markers by output volume
            alpha=0.6
        )
        
        plt.title(f'Relationship: Solar Output (kWh) vs. {TEMPERATURE_COL} (Colored by Cloud Type)', fontsize=16)
        plt.xlabel(f"{TEMPERATURE_COL}", fontsize=12)
        plt.ylabel("Power Generated (kWh)", fontsize=12)
        plt.legend(title='Cloud Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
    except FileNotFoundError:
        print(f"Error: File not found at the specified path: {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")

# Run the analysis
if __name__ == "__main__":
    solar_path = Path(r"\Kingfisher_Solutions\Data\split_sites")
    weather_path = Path(r"\Kingfisher_Solutions\Data\NSRDB")
    analyze_solar_data(solar_path, weather_path)
