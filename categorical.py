import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

def one_hot_encode_categorical(df):
    categorical_group = ['property_type', 'room_type', 'bed_type', 'cancellation_policy']
    bool_group = ['cleaning_fee', 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable']
    df[categorical_group].fillna("Unknown", inplace=True)

    def clean_to_bool(value):
        if pd.isna(value):
            return 'NaN'
        elif str(value).strip().lower() in ['true', 't', '1', 'yes']:
            return True
        elif str(value).strip().lower() in ['false', 'f', '0', 'no']:
            return False
        else:
            return 'NaN'  # Treat any unrecognized value as NaN

    encoding_map = {True: 1, False: 0, 'NaN': 2}

    for col in bool_group:
        df[col] = df[col].apply(clean_to_bool).map(encoding_map)

    # Setting up the OneHotEncoder
    encoder = OneHotEncoder(dtype=bool, sparse_output=False)
    transformer = ColumnTransformer([('one_hot_encoder', encoder, categorical_group)], remainder='drop')
    one_hot_encoded = transformer.fit_transform(df[categorical_group])

    pca = PCA(n_components=4)
    reduced_dimensions = pca.fit_transform(one_hot_encoded)
    df[categorical_group] = reduced_dimensions
    res = pd.concat([df[categorical_group], df[bool_group]],  axis=1)
    return res

