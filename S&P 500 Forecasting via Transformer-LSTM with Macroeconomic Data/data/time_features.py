import pandas as pd
import numpy as np

def enhance_time_features(data, time_col, target_cols):

    # Copy and delete invalid rows
    enhanced_data = data.copy()

    invalid_rows = enhanced_data[enhanced_data[time_col].isna()]
    if not invalid_rows.empty:
        print("Invalid rows：")
        print(invalid_rows)

    enhanced_data.dropna(subset=[time_col], inplace=True)

    # Convert to string and remove whitespace
    enhanced_data[time_col] = enhanced_data[time_col].astype(str).str.strip()

    # Convert to datetime format
    enhanced_data[time_col] = pd.to_datetime(enhanced_data[time_col], format='%Y-%m')

    # Extract year
    enhanced_data['year'] = enhanced_data[time_col].dt.year

    # Calculate year offset
    base_year = 1982
    enhanced_data['year_offset'] = enhanced_data['year'] - base_year + 1

    # Extract month
    enhanced_data['month'] = enhanced_data[time_col].dt.month

    # Cyclical features (month frequency)
    enhanced_data['sin_month'] = np.sin(2 * np.pi * enhanced_data['month'] / 12)
    enhanced_data['cos_month'] = np.cos(2 * np.pi * enhanced_data['month'] / 12)

    # Add month index (starting from 0)
    enhanced_data['time_index'] = (
            (enhanced_data['year'] - enhanced_data['year'].min()) * 12
            + enhanced_data['month'] - 1
    )

    # Generate lagged features for target
    for tcol in target_cols:
        if tcol in enhanced_data.columns:
            for lag in range(1, 4):
                enhanced_data[f'{tcol}_lag_{lag}'] = enhanced_data[tcol].shift(lag)
        else:
            print(f"Warning: {tcol} not found in the data columns, cannot generate lagged features.")

    # Keep YYYY-MM format
    enhanced_data[time_col] = pd.to_datetime(enhanced_data[time_col], format='%Y-%m')
    enhanced_data[time_col] = enhanced_data[time_col].dt.strftime('%Y-%m')

    return enhanced_data



# for testing
if __name__ == "__main__":
    file_path = "raw data.xlsx"
    data = pd.read_excel(file_path)

    my_target_cols = ['S&P 500 Price']

    enhanced_data = enhance_time_features(
        data,
        time_col='Date',
        target_cols=my_target_cols
    )

    print(enhanced_data.head())

    output_path = "enhanced raw data.xlsx"
    enhanced_data.to_excel(output_path, index=False)
    print(f"Processed data saved to: {output_path}")