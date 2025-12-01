"""
Training data builder module for the SpeedLimit System.

This module implements the data loading, cleaning, labeling, merging, and
augmentation pipeline for neural network training. It processes sensor readings,
accident data, and environmental conditions to create a training dataset for
near-accident risk prediction.

The module follows the conceptual specification:
- Near-accident definition: SkidAngle > 15° OR CloseCarCm < 50 OR CloseGuardrailCm < 50
- Features: illuminance, water level, temperature, speed limit, and other weather features
- Data is merged by segment-hour (Month, Day, Hour)
- Augmentation includes oversampling, noise injection, and speed sweeps
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

# Augmentation configuration
TARGET_POSITIVE_RATIO = 0.20  # Target ratio of positive (near-accident) examples (20%)


def _find_csv_files(data_dir: Path) -> dict[str, Path]:
    """
    Automatically finds CSV files in the data directory.
    
    Searches for files matching the patterns:
    - Sensors*.csv
    - SensorTypes*.csv
    - SensorReadings*.csv
    - accidents*.csv (case-insensitive)
    
    Args:
        data_dir: Directory to search for CSV files
    
    Returns:
        Dictionary mapping file type to file path
    
    Raises:
        FileNotFoundError: If required files are not found
    """
    files = {}
    
    # Find Sensors file
    sensors_files = list(data_dir.glob("Sensors*.csv"))
    if not sensors_files:
        raise FileNotFoundError(f"No Sensors*.csv file found in {data_dir}")
    files['sensors'] = sensors_files[0]
    
    # Find SensorTypes file
    sensor_types_files = list(data_dir.glob("SensorTypes*.csv"))
    if not sensor_types_files:
        raise FileNotFoundError(f"No SensorTypes*.csv file found in {data_dir}")
    files['sensor_types'] = sensor_types_files[0]
    
    # Find SensorReadings file
    readings_files = list(data_dir.glob("SensorReadings*.csv"))
    if not readings_files:
        raise FileNotFoundError(f"No SensorReadings*.csv file found in {data_dir}")
    files['readings'] = readings_files[0]
    
    # Find accidents file (case-insensitive)
    accidents_files = list(data_dir.glob("*ccidents*.csv"))
    if not accidents_files:
        accidents_files = list(data_dir.glob("*ccident*.csv"))
    if not accidents_files:
        raise FileNotFoundError(f"No accidents*.csv file found in {data_dir}")
    files['accidents'] = accidents_files[0]
    
    return files


def _load_and_parse_csv(file_path: Path) -> pd.DataFrame:
    """
    Loads a CSV file with semicolon separator and comma decimal.
    
    Args:
        file_path: Path to the CSV file
    
    Returns:
        DataFrame with parsed data
    """
    df = pd.read_csv(file_path, sep=';', decimal=',', encoding='utf-8')
    return df


def _merge_sensor_readings(
    readings_df: pd.DataFrame,
    sensors_df: pd.DataFrame,
    sensor_types_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges sensor readings with sensor metadata and types.
    
    Creates a wide-format dataset with one row per segment-hour, with columns
    for each sensor type (illuminance, water level, temperature, etc.).
    
    Args:
        readings_df: Sensor readings data
        sensors_df: Sensor metadata (SensorID -> SensorTypeCode mapping)
        sensor_types_df: Sensor type definitions
    
    Returns:
        Merged DataFrame with segment-hour rows and sensor value columns
    """
    # Filter out speed limit sensor (SensorID = -1) from regular sensor readings
    # Speed limit will be handled separately
    regular_readings = readings_df[readings_df['Sensor'] != -1].copy()
    
    # Merge readings with sensor types
    merged = regular_readings.merge(sensors_df, left_on='Sensor', right_on='SensorID', how='left')
    merged = merged.merge(sensor_types_df, on='SensorTypeCode', how='left')
    
    # Remove rows where SensorTypeCode is missing (sensor not found in mapping)
    merged = merged.dropna(subset=['SensorTypeCode'])
    
    # Pivot to wide format: one row per segment-hour, columns for each sensor type
    # Group by Month, Day, Hour and pivot by SensorTypeCode
    pivot_df = merged.pivot_table(
        index=['Month', 'Day', 'Hour'],
        columns='SensorTypeCode',
        values='Value',
        aggfunc='mean'  # Average if multiple readings per hour
    ).reset_index()
    
    # Flatten column names (handle MultiIndex if present)
    if isinstance(pivot_df.columns, pd.MultiIndex):
        pivot_df.columns = [col[1] if col[1] else col[0] for col in pivot_df.columns]
    pivot_df.columns.name = None
    
    # Extract speed limit (sensorID = -1)
    speed_limit_df = readings_df[readings_df['Sensor'] == -1].copy()
    speed_limit_df = speed_limit_df.groupby(['Month', 'Day', 'Hour'])['Value'].first().reset_index()
    speed_limit_df.rename(columns={'Value': 'SpeedLimit'}, inplace=True)
    
    # Merge speed limit
    result = pivot_df.merge(speed_limit_df, on=['Month', 'Day', 'Hour'], how='left')
    
    return result


def _label_near_accidents(accidents_df: pd.DataFrame) -> pd.DataFrame:
    """
    Labels near-accidents based on the specification rules.
    
    Near-accident criteria:
    - SkidAngle > 15° (only if value is present and > 15)
    - CloseCarCm < 50 (only if value is present and < 50)
    - CloseGuardrailCm < 50 (only if value is present and < 50)
    
    Missing values are treated as safe (non-hazardous):
    - Missing SkidAngle: interpreted as "no skid detected" (safe, default = 0)
    - Missing CloseCarCm: interpreted as "far from car" (safe, default = 999)
    - Missing CloseGuardrailCm: interpreted as "far from guardrail" (safe, default = 999)
    
    Args:
        accidents_df: Accidents data
    
    Returns:
        DataFrame with near_accident label (1 = near-accident, 0 = safe)
    """
    accidents_df = accidents_df.copy()
    
    # Convert to numeric, handling missing values with safe defaults
    # SkidAngle: missing = no skid = safe, so use 0 (which is <= 15, so won't trigger rule)
    accidents_df['SkidAngle'] = pd.to_numeric(accidents_df['SkidAngle'], errors='coerce').fillna(0)
    
    # CloseCarCm and CloseGuardrailCm: missing = far away = safe, so use 999 (which is >= 50, so won't trigger rule)
    accidents_df['CloseCarCm'] = pd.to_numeric(accidents_df['CloseCarCm'], errors='coerce').fillna(999)
    accidents_df['CloseGuardrailCm'] = pd.to_numeric(accidents_df['CloseGuardrailCm'], errors='coerce').fillna(999)
    
    # Label near-accidents
    # Only triggers if:
    # - SkidAngle > 15 (and not missing, since missing = 0 which is <= 15)
    # - CloseCarCm < 50 (and not missing, since missing = 999 which is >= 50)
    # - CloseGuardrailCm < 50 (and not missing, since missing = 999 which is >= 50)
    accidents_df['near_accident'] = (
        (accidents_df['SkidAngle'] > 15) |
        (accidents_df['CloseCarCm'] < 50) |
        (accidents_df['CloseGuardrailCm'] < 50)
    ).astype(int)
    
    # Group by segment-hour and take maximum (if any accident in hour, label as near-accident)
    labeled = accidents_df.groupby(['Month', 'Day', 'Hour'])['near_accident'].max().reset_index()
    
    return labeled


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans invalid rows from the dataset.
    
    Removes rows with:
    - Missing essential readings (illuminance, water level, temperature, speed limit)
    - Nonsensical values (out of expected ranges)
    
    Args:
        df: DataFrame to clean
    
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Remove rows with missing essential features
    essential_cols = ['L', 'W', 'T', 'SpeedLimit']  # Light, Water, Temperature, SpeedLimit
    available_cols = [col for col in essential_cols if col in df.columns]
    
    if not available_cols:
        raise ValueError("No essential sensor columns found in data")
    
    df = df.dropna(subset=available_cols)
    
    # Remove nonsensical values
    # Light (L): 0 to 1000000 lux
    if 'L' in df.columns:
        df = df[(df['L'] >= 0) & (df['L'] <= 1000000)]
    
    # Water (W): 0 to 10000 micrometers
    if 'W' in df.columns:
        df = df[(df['W'] >= 0) & (df['W'] <= 10000)]
    
    # Temperature (T): -30 to 70 degrees Celsius
    if 'T' in df.columns:
        df = df[(df['T'] >= -30) & (df['T'] <= 70)]
    
    # Speed limit: reasonable range 30-130 km/h
    if 'SpeedLimit' in df.columns:
        df = df[(df['SpeedLimit'] >= 30) & (df['SpeedLimit'] <= 130)]
    
    return df


def _augment_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs data augmentation with target-driven oversampling and noise injection.
    
    Augmentation steps (in order):
    1. Target-driven oversampling: Oversample positive rows to reach TARGET_POSITIVE_RATIO (20%)
    2. Noise injection: Add Gaussian noise to weather features (L, W, T) for all rows
    
    Args:
        df: Original training data
    
    Returns:
        Augmented DataFrame with duplicates removed
    """
    df = df.copy()
    
    # Step 1: Target-driven oversampling
    # Check if positive ratio is below target threshold
    positive_count = (df['near_accident'] == 1).sum()
    negative_count = (df['near_accident'] == 0).sum()
    positive_ratio = positive_count / len(df) if len(df) > 0 else 0
    
    if positive_ratio < TARGET_POSITIVE_RATIO:
        # Calculate how many positive samples we need
        target_positive_count = int(negative_count * TARGET_POSITIVE_RATIO / (1 - TARGET_POSITIVE_RATIO))
        current_positive_count = positive_count
        
        if current_positive_count > 0:
            # Calculate oversampling factor
            oversample_factor = max(1, target_positive_count / current_positive_count)
            n_duplications = int(np.ceil(oversample_factor)) - 1  # -1 because we already have original
            
            # Extract positive rows
            positive_rows = df[df['near_accident'] == 1].copy()
            
            # Duplicate positive rows to reach target ratio
            oversampled_positive = [positive_rows]  # Start with original
            for _ in range(n_duplications):
                oversampled_positive.append(positive_rows.copy())
            
            # If still not enough, sample with replacement to reach exact target
            oversampled_positive_df = pd.concat(oversampled_positive, ignore_index=True)
            if len(oversampled_positive_df) < target_positive_count:
                additional_needed = target_positive_count - len(oversampled_positive_df)
                additional = positive_rows.sample(
                    n=additional_needed,
                    replace=True,
                    random_state=42
                )
                oversampled_positive_df = pd.concat([oversampled_positive_df, additional], ignore_index=True)
            
            # Combine oversampled positive with negative rows
            negative_rows = df[df['near_accident'] == 0].copy()
            df_oversampled = pd.concat([negative_rows, oversampled_positive_df], ignore_index=True)
            
            print(f"After oversampling: {len(df_oversampled)} rows")
            positive_count_oversampled = (df_oversampled['near_accident'] == 1).sum()
            negative_count_oversampled = (df_oversampled['near_accident'] == 0).sum()
            positive_ratio_oversampled = positive_count_oversampled / len(df_oversampled) if len(df_oversampled) > 0 else 0
            print(f"  Positive: {positive_count_oversampled} ({positive_ratio_oversampled:.2%}), Negative: {negative_count_oversampled}")
        else:
            # No positive rows to oversample
            df_oversampled = df.copy()
            print("No positive rows to oversample")
    else:
        # Positive ratio already meets or exceeds target, no oversampling needed
        df_oversampled = df.copy()
        print(f"Positive ratio ({positive_ratio:.2%}) already meets target ({TARGET_POSITIVE_RATIO:.2%}), skipping oversampling")
    
    # Step 2: Noise injection AFTER oversampling
    # Add slight Gaussian noise to weather-related features (L, W, T)
    noise_df = df_oversampled.copy()
    weather_cols = ['L', 'W', 'T']  # Light, Water, Temperature
    available_weather_cols = [col for col in weather_cols if col in noise_df.columns]
    
    for col in available_weather_cols:
        # Add 5% Gaussian noise
        noise = np.random.normal(0, noise_df[col].std() * 0.05, size=len(noise_df))
        noise_df[col] = noise_df[col] + noise
        # Clip to valid ranges
        if col == 'L':
            noise_df[col] = noise_df[col].clip(0, 1000000)
        elif col == 'W':
            noise_df[col] = noise_df[col].clip(0, 10000)
        elif col == 'T':
            noise_df[col] = noise_df[col].clip(-30, 70)
    
    # Combine all augmented data: oversampled + noise
    # Include both the oversampled data and the noise-injected version
    augmented_rows = [df_oversampled, noise_df]
    result = pd.concat(augmented_rows, ignore_index=True)
    
    # Remove duplicates
    result = result.drop_duplicates()
    
    print(f"After noise injection and deduplication: {len(result)} rows")
    final_positive = (result['near_accident'] == 1).sum()
    final_negative = (result['near_accident'] == 0).sum()
    final_ratio = final_positive / len(result) if len(result) > 0 else 0
    print(f"  Final distribution: Positive: {final_positive} ({final_ratio:.2%}), Negative: {final_negative}")
    
    return result


def build_training_data(output_path: str) -> Path:
    """
    Builds the complete training dataset from raw CSV files.
    
    This is the main function that orchestrates the entire data pipeline:
    1. Loads and parses CSV files
    2. Merges sensor readings into segment-hour format
    3. Labels near-accidents
    4. Cleans invalid data
    5. Performs augmentation
    6. Saves to output file
    
    Args:
        output_path: Path where the training dataset will be saved
    
    Returns:
        Path to the generated training data file
    
    Raises:
        FileNotFoundError: If required CSV files are not found
        ValueError: If data processing fails
    """
    # Determine data directory (assume data is in training/data relative to this file)
    current_file = Path(__file__)
    data_dir = current_file.parent / "data"
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    print(f"Loading data from: {data_dir}")
    
    # Find CSV files
    try:
        csv_files = _find_csv_files(data_dir)
        print(f"Found files: {csv_files}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise
    
    # Load CSV files
    print("Loading CSV files...")
    sensors_df = _load_and_parse_csv(csv_files['sensors'])
    sensor_types_df = _load_and_parse_csv(csv_files['sensor_types'])
    readings_df = _load_and_parse_csv(csv_files['readings'])
    accidents_df = _load_and_parse_csv(csv_files['accidents'])
    
    print(f"Loaded {len(readings_df)} sensor readings, {len(accidents_df)} accident records")
    
    # Merge sensor readings
    print("Merging sensor readings...")
    merged_df = _merge_sensor_readings(readings_df, sensors_df, sensor_types_df)
    print(f"Merged to {len(merged_df)} segment-hour rows")
    
    # Label near-accidents
    print("Labeling near-accidents...")
    labeled_accidents = _label_near_accidents(accidents_df)
    
    # Merge labels with sensor data
    final_df = merged_df.merge(labeled_accidents, on=['Month', 'Day', 'Hour'], how='left')
    final_df['near_accident'] = final_df['near_accident'].fillna(0).astype(int)
    
    print(f"Labeled data: {final_df['near_accident'].sum()} near-accidents out of {len(final_df)} rows")
    
    # Clean data
    print("Cleaning data...")
    cleaned_df = _clean_data(final_df)
    print(f"After cleaning: {len(cleaned_df)} rows")
    
    # Augment data
    print("Augmenting data...")
    augmented_df = _augment_data(cleaned_df)
    print(f"After augmentation: {len(augmented_df)} rows")
    
    # Ensure output directory exists
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    print(f"Saving training data to: {output_path}")
    augmented_df.to_csv(output_path_obj, index=False)
    
    print(f"Training data built successfully: {len(augmented_df)} rows saved to {output_path}")
    
    return output_path_obj


if __name__ == "__main__":
    # Default output path
    output_path = Path(__file__).parent.parent.parent / "data" / "nn_training_data.csv"
    build_training_data(str(output_path))

