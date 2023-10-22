import os
import sys
import pickle
import pandas as pd
from argparse import ArgumentParser
from sklearn.preprocessing import LabelEncoder

from src.preproc import *
from src.model_utils import *
import src.hyperparams_opt as hyperparams_opt

# Some constants
LABEL_ENC_FOLDER = './encoders'
CHECKPOINT_FOLDER = './checkpoints'
if(not os.path.exists(LABEL_ENC_FOLDER)):
    os.mkdir(LABEL_ENC_FOLDER)
if(not os.path.exists(CHECKPOINT_FOLDER)):
    os.mkdir(CHECKPOINT_FOLDER)

def preprocessing_df(df):
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


if __name__ == '__main__':
    ### Argument parser ###
    parser = ArgumentParser()
    parser.add_argument('--output_file', type=str, required=False, help='Path to output file')
    parser.add_argument('--oversampling', required=False, action='store_true', help='Oversampling or not')
    args = vars(parser.parse_args())

    ### 1. Load data ###
    train_df = pd.read_csv('data/bank-train.csv', sep=',')
    test_df  = pd.read_csv('data/bank-test.csv', sep=',')

    ### 2. Preprocessing dataframes ###
    # Load train + test dataframes
    train_df = preprocessing_df(train_df) 
    test_df  = preprocessing_df(test_df)

    # Get feature + target columns
    target_col = 'subscription'
    feat_cols = [col for col in train_df.columns if col != 'subscription']

    ### 3. Hyper-parameters tuning ###
    # Set stdout
    sys.stdout = open(args['output_file'], 'w') if args['output_file'] is not None else sys.stdout

    # Prepare for hyper-parameters tuning
    metrics = hyperparams_opt.metrics
    model_class = hyperparams_opt.models['random_forest']['model_class']
    hyperparams = hyperparams_opt.models['random_forest']['hyperparams']
    ckpt_filename = hyperparams_opt.models['random_forest']['ckpt_filename']
    results, best_param = hyperparams_tuning(train_df, model_class, hyperparams, metrics, feat_cols, target_col, target_metric='mcc')
    print(f'\nBest hyper-parameters set:\n{best_param}')

    ### 4. Testing ###
    # Train the model with best param
    model = model_class(**best_param)
    model.fit(train_df[feat_cols], train_df[target_col])

    # Evaluate on test dataframe
    prediction = model.predict(test_df[feat_cols])
    results = {
        key : metrics[key](prediction, test_df[target_col])
        for key in metrics.keys()
    }
    print(f'\nTest results :\n{results}')

    # Checkpoint the model
    ckpt_file = os.path.join(CHECKPOINT_FOLDER, ckpt_filename)
    with open(ckpt_file, 'wb') as f:
        pickle.dump(model, f)
    print(f'\nCheckpoint saved to {ckpt_file}')

