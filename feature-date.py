import pandas as pd
import numpy as np

def cyclical_encode_dates(df):
    """
    Enhances a DataFrame by converting specified date columns to datetime format, reporting missing values,
    and adding cyclical features for month and day of year.

    Parameters:
    - df (DataFrame): Input DataFrame with 'first_review', 'host_since', 'last_review' columns.

    Returns:
    - DataFrame: Modified DataFrame with new cyclical feature columns for month and day.
    
    New columns added for each date column include:
    - [column_name]_month, [column_name]_year, [column_name]_day_of_year,
      [column_name]_month_sin, [column_name]_month_cos, [column_name]_day_sin, [column_name]_day_cos.
    """
    date_columns = ['first_review', 'host_since', 'last_review']
    
    for col in date_columns:
        missing_count = df[col].isna().sum()
        print(f"Missing rows in {col}: {missing_count}")
        
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')  # Coerce errors will turn problematic parsing into NaT

    for col in date_columns:
        df[col + '_month'] = df[col].dt.month
        df[col + '_year'] = df[col].dt.year
        df[col + '_day_of_year'] = df[col].dt.dayofyear
        
        df[col + '_month_sin'] = np.sin(2 * np.pi * df[col + '_month'] / 12)
        df[col + '_month_cos'] = np.cos(2 * np.pi * df[col + '_month'] / 12)
        df[col + '_day_sin'] = np.sin(2 * np.pi * df[col + '_day_of_year'] / 365)
        df[col + '_day_cos'] = np.cos(2 * np.pi * df[col + '_day_of_year'] / 365)

    return df


if __name__ == "__main__":
    df = pd.read_csv('Airbnb_Data.csv')
    processed_df = cyclical_encode_dates(df)

    processed_df.to_csv('processed_Airbnb_Data.csv', index=False)
    print(processed_df.head())