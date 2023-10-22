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
CHECKPOINT_FOLDER = './checkpoints'
if(not os.path.exists(CHECKPOINT_FOLDER)):
    os.mkdir(CHECKPOINT_FOLDER)

if __name__ == '__main__':
    ### Argument parser ###
    parser = ArgumentParser()
    parser.add_argument('--output_file', type=str, required=False, help='Path to output file')
    parser.add_argument('--oversampling', required=False, action='store_true', help='Oversampling or not')
    args = vars(parser.parse_args())

    if(args['oversampling']):
        CHECKPOINT_FOLDER = os.path.join(CHECKPOINT_FOLDER, 'with_oversampling')
        os.mkdir(CHECKPOINT_FOLDER)

    ### 1. Load data ###
    train_df = pd.read_csv('data/bank-train.csv', sep=',')
    test_df  = pd.read_csv('data/bank-test.csv', sep=',')

    ### 2. Preprocessing dataframes ###
    # Load train + test dataframes
    train_df = preproc_df_for_tree_algos(train_df) 
    test_df  = preproc_df_for_tree_algos(test_df)

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

