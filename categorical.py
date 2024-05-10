import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def one_hot_encode_categorical(df):
    categorical_group = ['property_type', 'room_type', 'bed_type', 'cancellation_policy', 'cleaning_fee', 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable']

    # drop missing value 
    # df_dropped = df.dropna(subset=categorical_group)
    df[categorical_group].fillna("Unknown", inplace=True)

    # Setting up the OneHotEncoder
    encoder = OneHotEncoder(dtype=bool, sparse_output=False)
    transformer = ColumnTransformer([('one_hot_encoder', encoder, categorical_group)], remainder='drop')

    # Applying one-hot encoding
    encoded_array = transformer.fit_transform(df)
    encoded_df = pd.DataFrame(encoded_array, columns=transformer.get_feature_names_out())
    return encoded_df

