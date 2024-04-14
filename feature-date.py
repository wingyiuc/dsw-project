import pandas as pd


def cyclical_encode_dates(df):
    """
    Enhances a DataFrame by converting specified date columns to datetime format and calculating the number
    of days passed since each date.

    Parameters:
    - df (DataFrame): Input DataFrame with 'first_review', 'host_since', 'last_review' columns.

    Returns:
    - DataFrame: Modified DataFrame with new columns showing days passed since each specified date.

    New columns added for each date column include:
    - [column_name]_days_since
    """
    date_columns = ['first_review', 'host_since', 'last_review']
    current_date = pd.to_datetime("today")
    
    for col in date_columns:
        missing_count = df[col].isna().sum()
        print(f"Missing rows in {col}: {missing_count}")
        
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')  # Coerce errors will turn problematic parsing into NaT

    for col in date_columns:
        df[col + '_days_since'] = (current_date - df[col]).dt.days


    return df


if __name__ == "__main__":
    df = pd.read_csv('Airbnb_Data.csv')
    processed_df = cyclical_encode_dates(df)

    processed_df.to_csv('processed_Airbnb_Data.csv', index=False)
    print(processed_df.head())