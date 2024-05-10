import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def parser(s):
    # Remove the curly braces
    s = s.strip('{}')

    # Split the string by commas not inside quotes
    items = re.split(r',(?![^"]*"(?:(?:[^"]*"){2})*[^"]*$)', s)

    # Clean up the items by stripping whitespace and extra quotes
    cleaned_items = [item.strip().strip('"') for item in items]

    # Count the items
    count = len(cleaned_items)

    return count


def parse_amenities(df):
    scaler = StandardScaler()
    df['amenities_count'] = df['amenities'].apply(parser)
    df[['amenities_count']] = scaler.fit_transform(df['amenities_count'].to_numpy().reshape(-1, 1))
    return df[['amenities_count']]
