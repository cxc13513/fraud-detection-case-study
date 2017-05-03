import pandas as pd
from main import clean_all
import numpy as np
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import svm


class ModelPipeline(object):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def get_baseline_scores(self, X, y):
        '''
        inputs: cleaned/engineered df and ylabel name
        outputs: list of baseline scores (precision, recall, f1)
        '''
        y_true = y
        df['yhat_baseline'] = 1
        y_pred = df['yhat_baseline']
        baseline_precision = metrics.precision_score(y_true, y_pred)
        baseline_recall = metrics.recall_score(y_true, y_pred)
        baseline_f1 = metrics.f1_score(y_true, y_pred)
        return([('baseline_precision', baseline_precision),
               ('baseline_recall', baseline_recall),
               ('baseline_f1', baseline_f1)])

    def run_logistic_separately(self, X, list_vars_to_drop, y):
        for item in list_vars_to_drop:
            df.pop(item)
        df_logistic = df
        y = df_logistic.pop(ylabel)
        X = df_logistic
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)
        logistic = linear_model.LogisticRegression(penalty='l1')
        logistic.fit(X_train, y_train)
        log_score = cross_val_score(logistic, X, y,
                                    cv=5, scoring='f1_macro')
        return(zip('logistic', log_score))

    def get_othermodels_scores(self, X, y):
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
        # fit models with train set
        pipe.fit(X, y)
        scores = cross_val_score(pipe, X, y,
                                 cv=5, scoring='f1_macro')
        return(zip(model_sequence, scores))

    def rmsle(y_hat, y):
        target = y
        predictions = y_hat
        log_diff = np.log(predictions+1) - np.log(target+1)
        rmsle_raw = np.sqrt(np.mean(log_diff**2))
        rmsle_scorer = make_scorer(rmsle_raw, greater_is_better=False)
        return rmsle_scorer

    def parameter_tuning(pipeline, params, train_x, train_y):
        gscv = GridSearchCV(pipeline,
                            params,
                            n_jobs=-1,
                            verbose=True,
                            scoring=rmsle_scorer
                            )
        clf = gscv.fit(train_x, train_y)
        best_params = clf.best_params_
        best_rmsle = clf.best_score_
        return best_params, best_rmsle
