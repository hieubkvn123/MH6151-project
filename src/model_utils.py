import numpy as np
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
        print('-----------------------------------------------------------------')
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
    print('=================================================================')
    for model in summary.keys():
        print(f'Performance metrics of model {model}')
        for mkey, metrics in summary[model].items():
            mean = np.mean(metrics)
            std  = np.std(metrics)
            print(f' -- {mkey} : Mean={mean:.4f}, STD={std:.4f}')
        print('-----------------------------------------------------------------')
            