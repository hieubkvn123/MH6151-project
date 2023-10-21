import numpy as np
from itertools import product
from sklearn.model_selection import StratifiedShuffleSplit

def k_fold_validation(data, models, metrics, feat_cols, target_col, k=5):
    # Extract features + targets
    features = data[feat_cols]
    targets = data[target_col]
    
    # Initialize splitter
    splitter = StratifiedShuffleSplit(n_splits=k)
    summary = { key : { mkey : [] for mkey in metrics.keys() } for key in models.keys() }

    # Start k-fold cross valiation
    for key, model in models.items():
        print('--------------------------------------------------------------------------------------')
        print(f'Cross valiation for {key}')
        for i, (train_index, val_index) in enumerate(splitter.split(features, targets)):
            X_train, X_val = features.loc[train_index], features.loc[val_index]
            Y_train, Y_val = targets.loc[train_index], targets.loc[val_index]

            # Fit all models
            model.fit(X_train.values, Y_train.values)
            
            # Let all models make predictions on the validation dataset
            pred = model.predict(X_val.values)
            
            # Calculate performance metrics
            print(f' -- Split #{i+1}, performance metrics ', end='')
            performance_metrics_str = []
            for mkey, metrics_fn in metrics.items():
                val = metrics_fn(Y_val.values, pred)

                # Add these accuracies to the respective list
                summary[key][mkey].append(val)
            
                # Report
                performance_metrics_str.append(f'{mkey} = {val:.2f}')
            performance_metrics_str = ', '.join(performance_metrics_str)
            print(performance_metrics_str)


    # Print out the summary to decide the best model to pick
    print('======================================================================================')
    summarized_metrics = {}
    for model in summary.keys():
        print(f'Performance metrics of model {model}')
        summarized_metrics[model] = {}
        for mkey, metrics in summary[model].items():
            mean = np.mean(metrics)
            std  = np.std(metrics)
            summarized_metrics[model][mkey] = mean
            print(f' -- {mkey} : Mean={mean:.4f}, STD={std:.4f}')
        print('--------------------------------------------------------------------------------------')

    return summarized_metrics            
            
def hyperparams_tuning(df, model_class, hyperparams, metrics, feat_cols, target_col, target_metric=None):
    # Check the target metric
    if(target_metric is None):
        target_metric = metrics.keys()[0]

    # Get the parameter keys and values as separate lists
    param_keys = list(hyperparams.keys())  
    param_values = [hyperparams[key] for key in param_keys]

    # Generate all permutations of parameter values
    param_permutations = list(product(*param_values))

    # Loop thru all hyper-params set
    results = {}
    best_metric = 0
    best_param = None
    for param in param_permutations:
        param_dict = {x : y for x, y in zip(hyperparams.keys(), param)}
        model = model_class(**param_dict)

        print('\n\n--------------------------------------------------------------------------------------')
        print(f'Parameters set : {param_dict}')
        summarized_metrics = k_fold_validation(df, {'model' : model}, metrics, feat_cols, target_col)
        results[param] = summarized_metrics

        # Reset best metric and best param
        if(best_metric < summarized_metrics['model'][target_metric]):
            best_metric = summarized_metrics['model'][target_metric]
            best_param = param_dict

    return results, best_param
