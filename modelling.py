import os
import sys
import pickle
import pathlib
import pandas as pd
from collections import Counter
from argparse import ArgumentParser
from imblearn.over_sampling import RandomOverSampler

from src.preproc import *
from src.model_utils import *
import src.hyperparams_opt as hyperparams_opt

# Some constants
CHECKPOINT_FOLDER = './checkpoints'
if(not os.path.exists(CHECKPOINT_FOLDER)):
    pathlib.Path(CHECKPOINT_FOLDER).mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    ### Argument parser ###
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, required=False, 
            choices=['decision_tree', 'random_forest', 'gradient_boost_tree', 'adaboost'],
            help='Model to start training', default='decision_tree')
    parser.add_argument('--output_file', type=str, required=False, help='Path to output file')
    parser.add_argument('--oversampling', required=False, action='store_true', help='Oversampling or not')
    args = vars(parser.parse_args())

    ### 1. Load data ###
    train_df = pd.read_csv('data/bank-train.csv', sep=',')
    test_df  = pd.read_csv('data/bank-test.csv', sep=',')

    ### 2. Preprocessing dataframes ###
    # Load train + test dataframes
    train_df = preproc_df_for_tree_algos(train_df) 
    test_df  = preproc_df_for_tree_algos(test_df)

    # Get feature + target columns
    target_col = 'subscription'
    feat_cols = [col for col in train_df.columns if col != 'subscription'] # Can replace with feature selection

    # Get features and targets
    X_train, Y_train = train_df[feat_cols], train_df[target_col]
    X_test, Y_test = test_df[feat_cols], test_df[target_col]

    # Apply over-sampling if applicable
    if(args['oversampling']):
        # Create a separate folder
        CHECKPOINT_FOLDER = os.path.join(CHECKPOINT_FOLDER, 'with_oversampling')
        pathlib.Path(CHECKPOINT_FOLDER).mkdir(parents=True, exist_ok=True)

        # Oversampling
        sampler = RandomOverSampler(sampling_strategy=0.5)
        X_train, Y_train = sampler.fit_resample(X_train, Y_train)
        print(f'Class distribution after resampling : {Counter(Y_train)}')

    ### 3. Hyper-parameters tuning ###
    # Set stdout
    sys.stdout = open(args['output_file'], 'w') if args['output_file'] is not None else sys.stdout

    # Prepare for hyper-parameters tuning
    metrics = hyperparams_opt.metrics
    model_class = hyperparams_opt.models[args['model_name']]['model_class']
    hyperparams = hyperparams_opt.models[args['model_name']]['hyperparams']
    ckpt_filename = hyperparams_opt.models[args['model_name']]['ckpt_filename']
    results, best_param = hyperparams_tuning(X_train, Y_train, model_class, hyperparams, metrics, target_metric='mcc', model_name=args['model_name'])
    print(f'\nBest hyper-parameters set:\n{best_param}')

    ### 4. Testing ###
    # Train the model with best param
    model = model_class(**best_param)
    model.fit(X_train, Y_train)

    # Evaluate on test dataframe
    prediction = model.predict(X_test)
    results = {
        key : metrics[key](prediction, Y_test)
        for key in metrics.keys()
    }
    print(f'\nTest results :\n{results}')

    # Checkpoint model
    ckpt_file = os.path.join(CHECKPOINT_FOLDER, ckpt_filename)
    with open(ckpt_file, 'wb') as f:
        pickle.dump(model, f)
    print(f'\nCheckpoint saved to {ckpt_file}')
