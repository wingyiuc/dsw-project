import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

from na_filling_utils import linear_reg_fillna

def one_hot_encode_categorical(df):
    df = df.copy()
    categorical_group = ['property_type', 'room_type', 'bed_type', 'cancellation_policy']
    bool_group = ['cleaning_fee', 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable']

    def clean_to_bool(value):
        if pd.isna(value):
            return 'NaN'
        elif str(value).strip().lower() in ['true', 't', '1', 'yes']:
            return True
        elif str(value).strip().lower() in ['false', 'f', '0', 'no']:
            return False
        else:
            return 'NaN'  # Treat any unrecognized value as NaN

    encoding_map = {True: 1, False: 0, 'NaN': pd.NA}

    for col in bool_group:
        df[col] = df[col].apply(clean_to_bool).map(encoding_map)
        
    # linear reg
    # host_has_profile_pic, host_identity_verified
    host_has_profile_pic_nonna = linear_reg_fillna(df, ['property_type', 'room_type', 'cleaning_fee', 'instant_bookable'], 'host_has_profile_pic')
    host_identity_verified_nonna = linear_reg_fillna(df, ['accommodates', 'bathrooms', 'first_review', 'host_response_rate', 'host_since', 'number_of_reviews', 'review_scores_rating', 'bedrooms', 'beds'], 'host_identity_verified')
    df['host_has_profile_pic'] = host_has_profile_pic_nonna
    df['host_identity_verified'] = host_identity_verified_nonna
    

    # Setting up the OneHotEncoder
    encoder = OneHotEncoder(dtype=bool, sparse_output=False)
    transformer = ColumnTransformer([('one_hot_encoder', encoder, categorical_group)], remainder='drop')
    one_hot_encoded = transformer.fit_transform(df[categorical_group])

    pca = PCA(n_components=4)
    reduced_dimensions = pca.fit_transform(one_hot_encoded)
    df[categorical_group] = reduced_dimensions
    res = pd.concat([df[categorical_group], df[bool_group]],  axis=1)
    return res

