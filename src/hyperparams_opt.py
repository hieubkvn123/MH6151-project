from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score

models = {
    'random_forest' : {
        'model_class' : RandomForestClassifier,
        'hyperparams' : {
            'n_estimators' : [25 * i for i in range(1, 6)],
            'max_depth' : [10, 20]
        },
        'ckpt_filename' : 'random_forest.pkl'
    },
    'decision_tree' : {
        'model_class' : DecisionTreeClassifier,
        'hyperparams' : {
            'criterion' : ['gini', 'entropy'],
            'max_depth' : [10, 20],
            'min_samples_split' : [2, 5, 10]
        },
        'ckpt_filename' : 'decision_tree.pkl'
    },
    'gradient_boost_tree' : {
        'model_class' : GradientBoostingClassifier,
        'hyperparams' : {
            'learning_rate' : [0.1, 0.2, 0.5],
            'n_estimators' : [50, 100, 150]
        },
        'ckpt_filename' : 'gradient_boost_tree'
    }
}

metrics = {
    'mcc' : matthews_corrcoef,
    'accuracy' : accuracy_score,
    'f1_score' : lambda y, pred : f1_score(y, pred, pos_label='yes')
}