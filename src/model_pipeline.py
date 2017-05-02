import pandas as pd
from sklearn import ensemble
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import svm


class ModelPipeline(object):

    def __init__(self, df=None):
        self.df = df

    def get_model_scores(self, json_logistic=None, json=None, ylabel=None):
        '''
        input: cleaned/engineered json file
        output: return f1_macro scores from models
        '''
        # models to run
        # logistic = linear_model.LogisticRegression(penalty='l1')
        sgd = linear_model.SGDClassifier(loss="log")
        svm = svm.LinearSVC()
        randomforest = ensemble.RandomForestClassifier()

        # save down sequence of models to run for future reference
        model_sequence = ['sgd', 'svm', 'randomforest']

        # define a pipeline
        pipe = Pipeline([('sgd', sgd),
                        ('svm', svm),
                        ('randomforest', randomforest)])
        # convert json into df
        df = pd.read_json(json)

        # create X, y, then do train_test_split
        y = df.pop(ylabel)
        X = df
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        # fit models with train set
        pipe.fit(X_train, y_train)
        scores = cross_val_score(pipe, X_train, y_train,
                                 cv=5, scoring='f1_macro')
        return(zip(model_sequence, scores))
