import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Observations
# Some numerical columns have string entries

# normalize things

NUMERICAL_COLUMNS = ['accommodates', 'bathrooms', 'host_response_rate', 'number_of_reviews', 'review_scores_rating', 'bedrooms', 'beds']


def remove_outliers(df, column_name):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)

    # Calculate IQR
    IQR = Q3 - Q1

    # Define upper and lower bounds for outlier detection
    lower_bound = Q1 - 2 * IQR
    upper_bound = Q3 + 2 * IQR

    # Filter the DataFrame to remove outliers
    df_filtered = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
    
    return df_filtered

def convert_column_types(df):
    df = df.copy()
    mean_value = df['bedrooms'].mean() # use floorspace to predict bedrooms?
    df['bedrooms'] = df['bedrooms'].fillna(mean_value)
    df['bedrooms'] = df['bedrooms'].astype(int)
    
    mean_value = df['beds'].mean()  # use bed_type to predict beds?
    df['beds'] = df['beds'].fillna(mean_value)
    df['beds'] = df['beds'].astype(int)
    
    df['host_response_rate'] = df['host_response_rate'].str.replace('%', '').astype(float)
    host_response_rate_mean = df['host_response_rate'].mean()
    df['host_response_rate'].fillna(host_response_rate_mean, inplace=True)

    bathroom_mode = df['bathrooms'].mode().iloc[0]
    print("mode of bathroom: ", df['bathrooms'].mode().iloc[0])
    df['bathrooms'].fillna(bathroom_mode, inplace=True)
    df['beds_per_bedroom'] = df['beds'] / (df['bedrooms'] + 1)
    beds_per_bedroom_mode = df['beds_per_bedroom'].mode().iloc[0]
    df['beds_per_bedroom'].fillna(beds_per_bedroom_mode, inplace=True)

    df['bed_and_bathrooms'] = df['bedrooms'] + df['bathrooms']
    scaler = StandardScaler()
    df['bed_and_bathrooms'] = scaler.fit_transform(
        df['bed_and_bathrooms'].to_numpy().reshape(-1, 1))
    
    return df


def min_max_normalize(column):
    return (column - column.min()) / (column.max() - column.min())

def normalize_left_skewed(df, column_name):
    df[column_name + "_normalized"] = min_max_normalize(np.sqrt(df[column_name]))
    # df['normalized_column'] = np.cbrt(df['left_skewed_column'])
    # df['normalized_column'] = df['left_skewed_column'] ** (1/3)
    return df

def normalize_right_skewed(df, column_name):
    df[column_name + "_normalized"] = min_max_normalize(np.log(df[column_name] + 1))
    return df
        

def process_numerical_columns(df):
    """
    Processes the numerical columns of Airbnb listing data.
    """
    df = convert_column_types(df)

    # Step 1: Remove outliers
    
    # accommodates, bathrooms, bedrooms and beds are skewed
    # they are correlated so removing outliers from one is suffiicent
    # df = remove_outliers(df, 'accommodates')

    # Step 2: Handle missing values (tbc)
    
    
    # Step 3: Normalize
    # df = normalize_left_skewed(df, 'accommodates')
    # df = normalize_left_skewed(df, 'bathrooms')
    df = normalize_right_skewed(df, 'host_response_rate')
    # df = normalize_left_skewed(df, 'number_of_reviews')
    df = normalize_right_skewed(df, 'review_scores_rating')
    df['normalized_rating'] = df['review_scores_rating_normalized'] * df['number_of_reviews'] / df['number_of_reviews'].sum()
    mean_normalized_rating = df['normalized_rating'].mean()
    df['normalized_rating'].fillna(mean_normalized_rating, inplace=True)
    # df = normalize_left_skewed(df, 'bedrooms')
    # df = normalize_left_skewed(df, 'beds')
    columns = ['accommodates', 'beds_per_bedroom', 'beds', 'bedrooms', 'bed_and_bathrooms',
               'bathrooms', 'host_response_rate', 'normalized_rating']

    return df[columns]


# columns = ['accommodates', 'beds', 'bedrooms', 'bathrooms', 'host_response_rate', 'normalized_rating']

"""
How to run the code: 

file_path = 'Airbnb_Data.csv'
df = pd.read_csv(file_path)
processed = process_numerical_columns(df)
print(processed.head(3))
"""
