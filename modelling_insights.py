import os
import pickle
import pathlib
import pandas as pd
import src.hyperparams_opt as hyperparams_opt
from src.preproc import preproc_df_for_bayes_algos, preproc_df_for_tree_algos

# Some constants
CHECKPOINT_FOLDER = './checkpoints'
pathlib.Path(CHECKPOINT_FOLDER).mkdir(parents=True, exist_ok=True)
pathlib.Path(os.path.join(CHECKPOINT_FOLDER, 'default')).mkdir(parents=True, exist_ok=True)

def test(model, X_test, Y_test):
    prediction = model.predict(X_test)
    results = {
        key : hyperparams_opt.final_report_metrics[key](prediction, Y_test)
        for key in hyperparams_opt.final_report_metrics.keys()
    }
    return results

if __name__ == '__main__':

    for model_name, item in hyperparams_opt.models.items():
        # Report model name
        print(f'\n{model_name}')

        # Read the raw data
        train_df = pd.read_csv('data/bank-train.csv', sep=',')
        test_df  = pd.read_csv('data/bank-test.csv', sep=',')

        # Preproc dataset
        if(model_name not in ['voting_naive_bayes', 'stacking_naive_bayes', 'gaussian_naive_bayes']):
            train_df = preproc_df_for_tree_algos(train_df.copy()) 
            test_df  = preproc_df_for_tree_algos(test_df.copy())
        else:
            train_df = preproc_df_for_bayes_algos(train_df.copy()) 
            test_df  = preproc_df_for_bayes_algos(test_df.copy())

        # Get feature + target columns
        target_col = 'subscription'
        feat_cols = [col for col in train_df.columns if col not in ['subscription']]

        # Get features and targets
        X_train, Y_train = train_df.copy()[feat_cols], train_df.copy()[target_col]
        X_test, Y_test = test_df.copy()[feat_cols], test_df.copy()[target_col]

        # Train the default model
        if(model_name not in ['voting_naive_bayes', 'stacking_naive_bayes']):
            default_model = item['model_class']()
        else:
            default_model = item['model_class'](
                cat_cols = [
                    'job_0', 'job_1', 'job_2', 'job_3', 'job_4', 'job_5',
                    'job_6', 'job_7', 'job_8', 'job_9', 'job_10', 'job_11', 'marital_0',
                    'marital_1', 'marital_2', 'contact_0', 'contact_1', 'contact_2',
                    'poutcome_0', 'poutcome_1', 'poutcome_2', 'poutcome_3', 'education_0',
                    'education_1', 'education_2', 'education_3', 'month_1', 'month_2',
                    'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8',
                    'month_9', 'month_10', 'month_11', 'month_12', 'default_0', 'default_1',
                    'housing_0', 'housing_1', 'loan_0', 'loan_1'
                ],
                num_cols = [
                    'age', 'day', 'campaign', 'pdays', 'previous'
                ]
            )
        ckpt_file = os.path.join(CHECKPOINT_FOLDER, 'default', item['ckpt_filename'])
        if(not os.path.exists(ckpt_file)):
            # Train
            default_model.fit(X_train, Y_train)

            # Checkpoint model
            with open(ckpt_file, 'wb') as f:
                pickle.dump(default_model, f)
        else:
            with open(ckpt_file, 'rb') as f:
                default_model = pickle.load(f)

        # Test the default model
        results = test(default_model, X_test, Y_test)
        print(f' -- Test results for default model :\n{results}')

        # Get the tuned model
        ckpt_file = os.path.join(CHECKPOINT_FOLDER, 'tuned', item['ckpt_filename'])
        with open(ckpt_file, 'rb') as f:
            tuned_model = pickle.load(f)
        results = test(tuned_model, X_test, Y_test)
        print(f' -- Test results for tuned model :\n{results}')

        # Get the tuned model after oversampling
        ckpt_file = os.path.join(CHECKPOINT_FOLDER, 'tuned_with_oversampling', item['ckpt_filename'])
        with open(ckpt_file, 'rb') as f:
            tuned_and_os_model = pickle.load(f)
        results = test(tuned_and_os_model, X_test, Y_test)
        print(f' -- Test results for tuned + oversampled model :\n{results}')
