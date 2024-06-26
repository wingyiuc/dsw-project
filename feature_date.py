import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def cyclical_encode_dates(df):
    """
    Enhances a DataFrame by converting specified date columns to datetime format and calculating the number
    of days passed since each date. Missing values in the calculated days are replaced with the column mean.

    Parameters:
    - df (DataFrame): Input DataFrame with 'first_review', 'host_since', 'last_review' columns.

    Returns:
    - DataFrame: Modified DataFrame with new columns showing days passed since each specified date.
                 Missing values in these new columns are replaced with the mean of each column.

    New columns added for each date column include:
    - [column_name]_days_since
    """
    date_columns = ['first_review', 'host_since', 'last_review']
    current_date = pd.to_datetime("today")
    df = df[date_columns]
    
    for col in date_columns:
        missing_count = df[col].isna().sum()
        print(f"Missing rows in {col}: {missing_count}")
        
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')  # Coerce errors will turn problematic parsing into NaT

    for col in date_columns:
        df[col + '_days_since'] = (current_date - df[col]).dt.days
        mean_days_since = df[col + '_days_since'].mean()

        df[col + '_days_since'].fillna(mean_days_since, inplace=True)


    df.drop(columns=['first_review', 'host_since', 'last_review'], inplace=True)

    scaler = StandardScaler()
    df[['first_review_days_since', 'host_since_days_since', 'last_review_days_since']
       ] = scaler.fit_transform(df)

    return df


if __name__ == "__main__":
    df = pd.read_csv('Airbnb_Data.csv')
    processed_df = cyclical_encode_dates(df)

    processed_df.to_csv('processed_Airbnb_Data.csv', index=False)
    print(processed_df.head())