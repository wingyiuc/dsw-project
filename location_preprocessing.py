import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import prince
import matplotlib.pyplot as plt


def scale_columns(df, cols):
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df



def get_locations_features(df):
    # scaler = MinMaxScaler()
    original_df = df[['latitude', 'longitude', 'city', 'neighbourhood']].copy()

    # # Option 1: scale latitude and longitude by city
    # df = df[['latitude', 'longitude', 'city', 'neighbourhood']]
    # # df[['latitude', 'longitude']] = scaler.fit_transform(
    # #     df[['latitude', 'longitude']])
    # # df = pd.get_dummies(df, columns=['city', 'neighbourhood'])
    # df = pd.get_dummies(df, columns=['city']) 
    # df['city'] = original_df['city']
    # scaled_df = df.groupby('city').apply(
    #     scale_columns, cols=['latitude', 'longitude']).drop('city', axis=1)
    # scaled_df = scaled_df.reset_index().set_index('id')
    # # loc_df = pd.concat([city_dummies, scaled_df], axis=1)

    # Option 2: FAMD dimensionality reduction
    df['neighbourhood'].fillna("Unknown", inplace=True)
    location_df = df[['latitude', 'longitude', 'city', 'neighbourhood']]
    famd = prince.FAMD(n_components=2, n_iter=3, random_state=42)
    # famd_df = famd.fit_transform(location_df)  # Apply FAMD to the entire DataFrame
    famd.fit(location_df)
    famd_df = famd.transform(location_df)
    famd_df.columns = ['location_component_0', 'location_component_1']
    famd_df['city'] = original_df['city']

    coordinates = famd.row_coordinates(location_df)


    # Assuming 'city' is a column in location_df you want to use for color labels
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(coordinates[0], coordinates[1],
                        c=location_df['city'].factorize()[0], cmap='viridis')

    # Create a colorbar with the city names
    legend1 = ax.legend(*scatter.legend_elements(), title="Cities")
    ax.add_artist(legend1)

    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('Row Coordinates from FAMD Analysis')
    plt.show()
    return famd_df


# Example usage:
# df = pd.read_csv("Airbnb_Data.csv")
# df = get_locations_features(df)