from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier

class BayesClassifier:
    def __init__(self, cat_cols, num_cols):
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
        ], voting='soft')

    def fit(self, X, y):
        return self.classifier.fit(X, y)
    
    def predict(self, X):
        return self.classifier.predict(X)
    