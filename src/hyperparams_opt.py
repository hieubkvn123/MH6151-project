import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score

models = {
    'decision_tree' : {
        'model_class' : DecisionTreeClassifier,
        'hyperparams' : {
            'criterion' : ['gini', 'entropy'],
            'max_depth' : [5, 10, 20],
            'min_samples_split' : [2, 5, 10]
        },
        'ckpt_filename' : 'decision_tree.pkl'
    },
    'random_forest' : {
        'model_class' : RandomForestClassifier,
        'hyperparams' : {
            'criterion' : ['gini', 'entropy', 'log_loss'],
            'max_depth' : [5, 10, 20],
            'n_estimators' : [25 * i for i in range(1, 6)],
        },
        'ckpt_filename' : 'random_forest.pkl'
    },
    'gradient_boost_tree' : {
        'model_class' : GradientBoostingClassifier,
        'hyperparams' : {
            'learning_rate' : [0.1, 0.2, 0.5],
            'n_estimators' : [25 * i for i in range(1, 6)],
            'min_samples_split' : [2, 5, 10]
        },
        'ckpt_filename' : 'gradient_boost_tree.pkl'
    },
    'adaboost' : {
        'model_class' : AdaBoostClassifier,
        'hyperparams' : {
            'estimator' : [DecisionTreeClassifier()],
            'n_estimators' : [25 * i for i in range(1, 6)],
            'learning_rate' : [0.1, 0.2, 0.5]
        },
        'ckpt_filename' : 'adaboost.pkl'
    },
    'xgboost' : {
        'model_class' : xgb.XGBClassifier,
        'hyperparams' : {
            'booster' : ['gbtree', 'gblinear', 'dart'],
            'max_depth' : [5, 10, 20],
            'eta' : [0.2, 0.3, 0.5]
        },
        'ckpt_filename' : 'xgboost.pkl'
    }
}

metrics = {
    'mcc' : matthews_corrcoef,
    'accuracy' : accuracy_score,
    'f1_score' : f1_score 
}
