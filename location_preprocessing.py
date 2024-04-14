import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def get_locations_features(df):
    scaler = MinMaxScaler()
    df = df[['latitude', 'longitude', 'city', 'neighbourhood']]
    df[['latitude', 'longitude']] = scaler.fit_transform(
        df[['latitude', 'longitude']])
    df = pd.get_dummies(df, columns=['city', 'neighbourhood'])
    return df, scaler


# Example usage:
# df = pd.read_csv("Airbnb_Data.csv")
# df = get_locations_features(df)