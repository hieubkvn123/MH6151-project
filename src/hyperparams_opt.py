from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score

models = {
    'random_forest' : {
        'model_class' : RandomForestClassifier,
        'hyperparams' : {
            'n_estimators' : [25 * i for i in range(1, 6)],
            'max_depth' : [10, 20]
        }
    },
    'decision_tree' : {
        'model_class' : DecisionTreeClassifier,
        'hyperparams' : {
            'criterion' : ['gini', 'entropy', 'log_loss'],
            'max_depth' : [10, 20],
            'min_samples_split' : [2, 5, 10]
        }
    }
}

metrics = {
    'mcc' : matthews_corrcoef,
    'accuracy' : accuracy_score,
    'f1_score' : lambda y, pred : f1_score(y, pred, pos_label='yes')
}