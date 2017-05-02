import pandas as pd
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import svm


class ModelPipeline(object):

    def __init__(self, df=None):
        self.df = df

    def get_baseline_scores(self, df, ylabel):
        '''
        inputs: cleaned/engineered df and ylabel name
        outputs: list of baseline scores (precision, recall, f1)
        '''
        y_true = df[ylabel]
        df['yhat_baseline'] = 1
        y_pred = df['yhat_baseline']
        baseline_precision = metrics.precision_score(y_true, y_pred)
        baseline_recall = metrics.recall_score(y_true, y_pred)
        baseline_f1 = metrics.f1_score(y_true, y_pred)
        return([('baseline_precision', baseline_precision),
               ('baseline_recall', baseline_recall),
               ('baseline_f1', baseline_f1)])

    def run_logistic_separately(self, df, list_vars_to_drop, ylabel):
        for item in list_vars_to_drop:
            df.pop(item)
        df_logistic = df
        y = df_logistic.pop(ylabel)
        X = df_logistic
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)
        logistic = linear_model.LogisticRegression(penalty='l1')
        logistic.fit(X_train, y_train)
        log_score = cross_val_score(logistic, X_train, y_train,
                                    cv=5, scoring='f1_macro')
        return(zip('logistic', log_score))

    def split_df(self, df, ylabel):
        # create X, y, then do train_test_split
        y = df.pop(ylabel)
        X = df
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)
        return X_train, X_test, y_train, y_test

    def get_othermodels_scores(self, df, ylabel):
        '''
        input: cleaned/engineered df
        output: return f1_macro scores from models
        '''
        # set models to run in pipeline
        sgd = linear_model.SGDClassifier(loss="log")
        svm = svm.LinearSVC()
        randomforest = ensemble.RandomForestClassifier()
        # save down sequence of models to run for future reference
        model_sequence = ['sgd', 'svm', 'randomforest']
        # define a pipeline
        pipe = Pipeline([('sgd', sgd),
                        ('svm', svm),
                        ('randomforest', randomforest)])
        # call in split df
        X_train, X_test, y_train, y_test = self.split_df(df, ylabel)
        # fit models with train set
        pipe.fit(X_train, y_train)
        scores = cross_val_score(pipe, X_train, y_train,
                                 cv=5, scoring='f1_macro')
        return(zip(model_sequence, scores))

    def parameter_tuning(self, df, ylabel):
        pass
