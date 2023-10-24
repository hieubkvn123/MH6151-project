import os
import pickle
import pathlib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

LABEL_ENC_FOLDER = './encoders'
NUM_ENC_FOLDER = './encoders'

if(not os.path.exists(LABEL_ENC_FOLDER)):
    pathlib.Path(LABEL_ENC_FOLDER).mkdir(parents=True, exist_ok=True)

def preproc_education(x):
    if(x == 'unknown') : return 0
    elif(x == 'primary') : return 1
    elif(x == 'secondary') : return 2
    elif(x == 'tertiary') : return 3
    
def preproc_month(x):
    if(x == 'jan') : return 1
    elif(x == 'feb') : return 2
    elif(x == 'mar') : return 3
    elif(x == 'apr') : return 4
    elif(x == 'may') : return 5
    elif(x == 'jun') : return 6
    elif(x == 'jul') : return 7
    elif(x == 'aug') : return 8
    elif(x == 'sep') : return 9
    elif(x == 'oct') : return 10
    elif(x == 'nov') : return 11
    elif(x == 'dec') : return 12
    
def preproc_binary(x):
    if(x == 'no') : return 0
    elif(x == 'yes') : return 1

def preproc_df_for_tree_algos(df):
    ### Label encoder paths ###
    encoders = {
        'job' : os.path.join(LABEL_ENC_FOLDER, 'job_labelenc.pkl'),
        'marital' : os.path.join(LABEL_ENC_FOLDER, 'marital_labelenc.pkl'),
        'contact' : os.path.join(LABEL_ENC_FOLDER, 'contact_labelenc.pkl'),
        'poutcome' : os.path.join(LABEL_ENC_FOLDER, 'poutcome_labelenc.pkl'),
    }

    ### Initialize label encoders ##
    job_labelenc = LabelEncoder() if not os.path.exists(encoders['job']) else pickle.load(open(encoders['job'], 'rb'))
    marital_labelenc = LabelEncoder() if not os.path.exists(encoders['marital']) else pickle.load(open(encoders['marital'], 'rb'))
    contact_labelenc = LabelEncoder() if not os.path.exists(encoders['contact']) else pickle.load(open(encoders['contact'], 'rb'))
    poutcome_labelenc = LabelEncoder() if not os.path.exists(encoders['poutcome']) else pickle.load(open(encoders['poutcome'], 'rb'))

    ### Preprocess dataframe ##
    # Rename last column
    df = df.rename(columns={'y' : 'subscription'})

    # Remove duration
    df = df.drop(columns='duration')

    # All preprocessing steps
    df['job'] = job_labelenc.fit_transform(df['job'])
    df['marital'] = marital_labelenc.fit_transform(df['marital'])
    df['contact'] = contact_labelenc.fit_transform(df['contact'])
    df['poutcome'] = poutcome_labelenc.fit_transform(df['poutcome'])
    df['education'] = df['education'].apply(preproc_education)
    df['month'] = df['month'].apply(preproc_month)
    df['default'] = df['default'].apply(preproc_binary)
    df['housing'] = df['housing'].apply(preproc_binary)
    df['loan'] = df['loan'].apply(preproc_binary)
    df['subscription'] = df['subscription'].apply(preproc_binary)

    # Save label encoders
    with open(encoders['job'], 'wb') as f:
        pickle.dump(job_labelenc, f)
    with open(encoders['marital'], 'wb') as f:
        pickle.dump(marital_labelenc, f)
    with open(encoders['contact'], 'wb') as f:
        pickle.dump(contact_labelenc, f)
    with open(encoders['poutcome'], 'wb') as f:
        pickle.dump(poutcome_labelenc, f)

    return df

def preproc_df_for_bayes_algos(df):
    # Label encode the categorical data
    df = preproc_df_for_tree_algos(df)

    # Onehot encode
    categorical_columns = [
        'job', 'marital', 'contact', 'poutcome', 'education',
        'month', 'default', 'housing', 'loan'
    ]
    numeric_columns = [x for x in df.columns if x not in categorical_columns + ['subscription', 'Unnamed: 0']]

    # Standard scaler
    for col in numeric_columns:
        enc_file = os.path.join(NUM_ENC_FOLDER, f'{col}_standardscaler.pkl')
        if(not os.path.exists(enc_file)):
            # Do the scaling
            scaler = StandardScaler()
            df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))

            # Save the encoder
            with open(enc_file, 'wb') as f:
                pickle.dump(scaler, f)
        else:
            with open(enc_file, 'rb') as f:
                scaler = pickle.load(f)
                df[col] = scaler.transform(df[col].values.reshape(-1, 1))
        
    # One-hot encode cat columns
    df = pd.get_dummies(df, columns=categorical_columns)

    return df
