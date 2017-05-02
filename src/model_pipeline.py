import pandas as pd
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import svm


class ModelPipeline(object):

    def __init__(self, json=None, json_logistic=None, ylabel=None):
        self.json = json
        self.json_logistic = json_logistic
        self.ylabel = ylabel

    def transform_json_df(self):
        cleaned_engineered_df = pd.read_json(self.json)
        return cleaned_engineered_df

    def get_baseline(self, cleaned_engineered_df=None, ylabel=None):
        y_true = cleaned_engineered_df[ylabel]
        cleaned_engineered_df['yhat_baseline'] = 1
        y_pred = cleaned_engineered_df['yhat_baseline']
        baseline_precision = metrics.precision_score(y_true, y_pred)
        baseline_recall = metrics.recall_score(y_true, y_pred)
        baseline_f1 = metrics.f1_score(y_true, y_pred)
        return([('baseline_precision:', baseline_precision),
               ('baseline_recall:', baseline_recall),
               ('baseline_f1:', baseline_f1)])

    def get_model_scores(self, cleaned_engineered_df=None, ylabel=None):
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

        # create X, y, then do train_test_split
        y = cleaned_engineered_df.pop(ylabel)
        X = cleaned_engineered_df
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        # fit models with train set
        pipe.fit(X_train, y_train)
        scores = cross_val_score(pipe, X_train, y_train,
                                 cv=5, scoring='f1_macro')
        return(zip(model_sequence, scores))

    def parameter_tuning(self, df_logistic=None, df=None, ylabel=None):
        pass
