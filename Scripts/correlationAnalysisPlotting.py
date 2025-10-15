# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 15:50:04 2025

@author: seani
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# === CONFIGURATION ===
# Define the path to the merged file
FILE_PATH = Path(r"C:\Users\seani\Documents\anaconda_projects\Kingfisher_Solutions\Data\Solar Outut By Site + Weather data combined\Bearspaw Water Treatment Plant\Bearspaw Water Treatment Plant_merged_solar_weather.csv")

# Identify key columns based on standard weather data structure
SOLAR_OUTPUT_COL = 'kWh'
IRRADIANCE_COL = 'Clearsky GHI' # Assuming 'Clearsky GHI' is the Global Horizontal Irradiance column
TEMPERATURE_COL = 'Temperature' # Assuming 'Temperature' is the ambient temperature column
# New column assumption based on correlation output:
PRIMARY_FEATURE = 'GHI' 

def build_prediction_model(df_active: pd.DataFrame, target_col: str, feature_cols: list):
    """
    Builds and evaluates a simple Linear Regression model for solar prediction.
    """
    print("\n--- Machine Learning: Linear Regression Model ---")

    # Drop any rows with NaN in the selected features/target before training
    model_df = df_active[[target_col] + feature_cols].dropna()
    
    if model_df.empty:
        print("Model skipped: Insufficient data after dropping NaNs in features.")
        return

    # Define features (X) and target (y)
    X = model_df[feature_cols]
    y = model_df[target_col]

    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Features used: {', '.join(feature_cols)}")
    print(f"Model Performance on Test Set:")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.2f} kWh (Lower is better)")
    print(f"  R-squared (R2) Score: {r2:.4f} (Closer to 1.0 is better)")
    
    # Plotting Model vs. Actual (simple scatter)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='#D39200')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction') # Diagonal line
    plt.xlabel('Actual Power Generated (kWh)')
    plt.ylabel('Predicted Power (kWh)')
    plt.title('ML Model: Actual vs. Predicted Solar Output')
    plt.legend()
    plt.show()

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
        
        # FIX: Explicitly convert the solar output column to numeric.
        # This resolves the 'str and int comparison' error.
        df[SOLAR_OUTPUT_COL] = pd.to_numeric(df[SOLAR_OUTPUT_COL], errors='coerce').fillna(0)
        
        # Filter for actual generation periods to remove night/inactive periods (where kWh == 0)
        # This focuses the analysis on operational hours.
        df_active = df[df[SOLAR_OUTPUT_COL] > 0].copy()
        
        print(f"Total records loaded: {len(df)}. Active generation records: {len(df_active)}")
        
        if df_active.empty:
            print("Error: No active solar generation records found after filtering.")
            return

        # 2. CORRELATION ANALYSIS (RE-CALCULATED FOR ML FEATURE SELECTION)
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
        feature_list = []
        for col in top_5_correlations.index:
             print(f"  {col.ljust(20)}: {correlations[col]:.4f}")
             feature_list.append(col)

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
            x=PRIMARY_FEATURE, # Using GHI as the highest correlated factor
            y=SOLAR_OUTPUT_COL, 
            data=df_active, 
            kind='scatter', 
            color='#00BA38', 
            alpha=0.7
        )
        plt.suptitle(f'Relationship: Solar Output (kWh) vs. {PRIMARY_FEATURE}', y=1.02, fontsize=16)
        plt.xlabel(f"{PRIMARY_FEATURE}", fontsize=12)
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
        
        # FIXED: Changed '$^{\circ}$C' to '\u00b0C' (Unicode degree symbol)
        plt.xlabel(f"{TEMPERATURE_COL} (\u00b0C)", fontsize=12) 
        
        plt.ylabel("Power Generated (kWh)", fontsize=12)
        plt.legend(title='Cloud Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

        # --- Visual 4: Monthly Performance Box Plot (New Visual) ---
        df_active['Month'] = df_active.index.month
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            x='Month', 
            y=SOLAR_OUTPUT_COL, 
            data=df_active, 
            palette='coolwarm',
            fliersize=3,
        )
        plt.title('Monthly Power Output Distribution (Seasonality)', fontsize=16)
        plt.xlabel("Month of Year", fontsize=12)
        plt.ylabel("Power Generated (kWh)", fontsize=12)
        plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.show()

        # --- Visual 5: Autocorrelation Plot (New Visual) ---
        # The first difference helps remove the trend for clearer autocorrelation
        pd.plotting.autocorrelation_plot(df_active[SOLAR_OUTPUT_COL].resample('H').mean().dropna(), ax=plt.figure(figsize=(12, 5)).gca(), color='#619CFF')
        plt.title('Autocorrelation of Hourly Solar Output', fontsize=16)
        plt.show()

        # 4. RUN MACHINE LEARNING MODEL
        # We'll use the top 4 non-self-correlated numeric features for the model
        ml_features = [col for col in feature_list if col != SOLAR_OUTPUT_COL][:4]
        build_prediction_model(df_active, SOLAR_OUTPUT_COL, ml_features)
        
    except FileNotFoundError:
        print(f"Error: File not found at the specified path: {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")

# Run the analysis
if __name__ == "__main__":
    # Ensure sklearn is available for ML
    try:
        analyze_solar_data(FILE_PATH)
    except ImportError:
        print("\nError: Scikit-learn (sklearn) is required for the machine learning section.")
        print("Please install it using: pip install scikit-learn")
