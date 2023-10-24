from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier, StackingClassifier

class VotingBayesClassifier:
    def __init__(self, cat_cols, num_cols, voting='hard'):
        cat_feature_selector = ColumnTransformer([('selector', 'passthrough', cat_cols)], remainder='drop')
        num_feature_selector = ColumnTransformer([('selector', 'passthrough', num_cols)], remainder='drop')

        self.cat_features_classifier = Pipeline([
            ('selector', cat_feature_selector),
            ('classifier', BernoulliNB())
        ])

        self.num_features_classifier = Pipeline([
            ('selector', num_feature_selector),
            ('classifier', GaussianNB())
        ])

        self.classifier = VotingClassifier(estimators=[
            ('cat_classifier', self.cat_features_classifier),
            ('num_classifier', self.num_features_classifier)
        ], voting=voting)


    def fit(self, X, y):
        return self.classifier.fit(X, y)
    
    def predict(self, X):
        return self.classifier.predict(X)

class StackingBayesClassifier:
    def __init__(self, cat_cols, num_cols, stack_method='auto'):
        cat_feature_selector = ColumnTransformer([('selector', 'passthrough', cat_cols)], remainder='drop')
        num_feature_selector = ColumnTransformer([('selector', 'passthrough', num_cols)], remainder='drop')

        self.cat_features_classifier = Pipeline([
            ('selector', cat_feature_selector),
            ('classifier', BernoulliNB())
        ])

        self.num_features_classifier = Pipeline([
            ('selector', num_feature_selector),
            ('classifier', GaussianNB())
        ])

        self.classifier = StackingClassifier(estimators=[
            ('cat_classifier', self.cat_features_classifier),
            ('num_classifier', self.num_features_classifier)
        ], stack_method=stack_method)


    def fit(self, X, y):
        return self.classifier.fit(X, y)
    
    def predict(self, X):
        return self.classifier.predict(X)
    