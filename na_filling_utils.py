import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.impute import SimpleImputer

CATEGORICAL_COLUMNS = ['property_type', 'room_type', 'bed_type', 'cancellation_policy', 'cleaning_fee', 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable']
DATE_COLUMNS = ['first_review', 'host_since', 'last_review']
NUMERICAL_COLUMNS = ['accommodates', 'bathrooms', 'host_response_rate', 'number_of_reviews', 'review_scores_rating', 'bedrooms', 'beds']


def process_date(df, column):
    current_date = pd.to_datetime("today")
    df[column] = pd.to_datetime(df[column], errors='coerce')  # Coerce errors will turn problematic parsing into NaT
    df[column + '_days_since'] = (current_date - df[column]).dt.days
    df[column] = df[column + '_days_since']
    
def process_numerical(df, column):
    if column == 'host_response_rate':
        df[column] = df[column].str.replace('%', '')
    df[column] = df[column].astype(float)
    
def process_categorical(df, column):
    return pd.get_dummies(df[column], prefix=column)

def encode_df(df, X_columns, y_column):
    df = df.copy()
    all_columns = X_columns.copy()
    all_columns.append(y_column)
    
    X = []
    for column in X_columns:
        if column in CATEGORICAL_COLUMNS:
            X.append(process_categorical(df, column))
        elif column in DATE_COLUMNS:
            process_date(df, column)
            X.append(df[column])
        elif column in NUMERICAL_COLUMNS:
            process_numerical(df, column)
            X.append(df[column])
            
    y = df[y_column]
    if y_column == "host_has_profile_pic":
        y = df[y_column].replace({'t': 1, 'f': 0})
    elif y_column == "host_identity_verified":
        y = df[y_column].replace({'t': 1, 'f': 0})
    elif y_column in DATE_COLUMNS:
        process_date(df, y_column)
        y = df[y_column]

    X = pd.concat(X, axis=1)

    df2 = pd.concat([X, y], axis=1).dropna()
    
    X_train, y_train = df2.iloc[:, :-1], df2.iloc[:, -1]
    
    return X_train, y_train, X
    
    
def linear_reg_fillna(df, X_columns, y_column):
    
    X_train, y_train, X = encode_df(df, X_columns, y_column)
    model = LinearRegression()
    
    model.fit(X_train, y_train)
    
    miss_mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    miss_mean_imputer = miss_mean_imputer.fit(X)
    imputed_X = miss_mean_imputer.transform(X.values)
    
    y_pred = model.predict(imputed_X)
    
    # use original y if exist. use y_pred if none are na.
    
    y_pred = np.where(X.isna().any(axis=1), y_train.mean(), y_pred)
    
    return pd.Series(np.where(df[y_column].isna(), y_pred, df[y_column]))
    
