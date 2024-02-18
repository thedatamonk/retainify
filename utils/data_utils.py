import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE


def convert_total_charges_to_float(df):

    df['TotalCharges'] = df['TotalCharges'].map(lambda x: np.nan if isinstance(x, str) and x.strip() == '' else x)

    before_nan_indices = df[pd.isna(df['TotalCharges'])].index.tolist()

    print('Index Positions with empty spaces before interpolation: ', before_nan_indices)

    # Fill indices with NaN with the last seen non-null value
    # I know that this is not the best way to do this but we will try to
    # improve this later
    
    df["TotalCharges"] = df['TotalCharges'].ffill(inplace=False)

    after_nan_indices = df[pd.isna(df['TotalCharges'])].index.tolist()

    print('Index Positions with empty spaces after interpolation: ', after_nan_indices)

    # Finally, convert all the non-empty string float values to float
    df['TotalCharges'] = df['TotalCharges'].astype(float)

    return df

def encode_categorical_feats(df, categorical_cols):

    # This will store the LabelEncoder instances for each column
    # We can use these later to perform an inverse transform (encoded labels -> original labels)
    encoder_dict = {}
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()

            df[col] = le.fit_transform(df[col])

            encoder_dict[col] = le

        else:
            print (f"Column {col} not found in dfframe")
    
    return df, encoder_dict


def scale_numeric_feats(df, numeric_cols):
    scaler = MinMaxScaler()

    for col in numeric_cols:
        if col in df.columns:
            df[col] = scaler.fit_transform(df[[col]])

    return df


async def process_data(df):

    df = convert_total_charges_to_float(df=df)
    df = df.drop(columns = ['customerID'], inplace = False)

    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_cols = [col for col in df.columns if col not in numeric_cols]

    
    df, label_encoder_dicts = encode_categorical_feats(df, categorical_cols=categorical_cols)

    df = df.drop(columns=['PhoneService', 'gender','StreamingTV','StreamingMovies','MultipleLines','InternetService'], inplace=False)

    df = scale_numeric_feats(df, numeric_cols=numeric_cols)

    return df



def balance_data(x_train, y_train):

    print (f"Target distribution before balancing with SMOTE\n", y_train.value_counts())

    # Initialize oversampler
    over_sampler = SMOTE(sampling_strategy=1, random_state=42)

    # Apply SMOTE on training data only
    x_train_smote, y_train_smote = over_sampler.fit_resample(x_train, y_train)

    print (f"Target distribution after balancing with SMOTE\n", y_train_smote.value_counts())

    return x_train_smote, y_train_smote





