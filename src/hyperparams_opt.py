import xgboost as xgb
from src.bayes import BayesClassifier
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
    },
    'naive_bayes' : {
        'model_class' : BayesClassifier,
        'hyperparams' : {
            'cat_cols' : [
                [
                    'job_0', 'job_1', 'job_2', 'job_3', 'job_4', 'job_5',
                    'job_6', 'job_7', 'job_8', 'job_9', 'job_10', 'job_11', 'marital_0',
                    'marital_1', 'marital_2', 'contact_0', 'contact_1', 'contact_2',
                    'poutcome_0', 'poutcome_1', 'poutcome_2', 'poutcome_3', 'education_0',
                    'education_1', 'education_2', 'education_3', 'month_1', 'month_2',
                    'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8',
                    'month_9', 'month_10', 'month_11', 'month_12', 'default_0', 'default_1',
                    'housing_0', 'housing_1', 'loan_0', 'loan_1'
                ]
            ],
            'num_cols' : [
                [
                    'age', 'day', 'campaign', 'pdays', 'previous'
                ]
            ]
        },
        'ckpt_filename' : 'naive_bayes.pkl'
    }
}

metrics = {
    'mcc' : matthews_corrcoef,
    'accuracy' : accuracy_score,
    'f1_score' : f1_score 
}
