from sklearn import ensemble
from sklearn.grid_search import GridSearchCV
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


class ModelPipeline(object):

    def __init__(self, X=None, y=None):
        self.X = X
        self.y = y

    def get_baseline_scores(self):
        '''Calculate baseline scores.

        inputs: cleaned/engineered df and ylabel name
        outputs: list of baseline scores (precision, recall, f1)
        '''
        y_true = self.y
        self.X['yhat_baseline'] = 1
        y_pred = self.X['yhat_baseline']
        baseline_precision = metrics.precision_score(y_true, y_pred)
        baseline_recall = metrics.recall_score(y_true, y_pred)
        baseline_f1 = metrics.f1_score(y_true, y_pred)
        return([('baseline_precision', baseline_precision),
               ('baseline_recall', baseline_recall),
               ('baseline_f1', baseline_f1)])

    def run_logistic_separately(self):
        logistic = linear_model.LogisticRegression(penalty='l1')
        logistic.fit(self.X, self.y)
        log_score = cross_val_score(logistic, self.X, self.y,
                                    cv=5, scoring='f1_macro')
        return(zip('logistic', log_score))

    def get_othermodels_scores(self):
        '''Calculate scores for all modesl except Logisitic.

        input: cleaned/engineered df
        output: return f1_macro scores from models
        '''
        # set models to run in pipeline
        sgd = linear_model.SGDClassifier(loss="log", alpha=0.0001,
                                         learning_rate='optimal')
        svc = LinearSVC(C=1)
        randomforest = ensemble.RandomForestClassifier(max_depth=1,
                                                       max_features='auto',
                                                       n_estimators=150)
        adaboost = ensemble.AdaBoostClassifier(learning_rate=0.75,
                                               n_estimators=50)
        # save down sequence of models to run for future reference
        model_sequence = ['sgd', 'svc', 'randomforest', 'adaboost']
        # define a pipeline
        pipe = Pipeline([('sgd', sgd),
                        ('svc', svc),
                        ('randomforest', randomforest),
                        ('adaboost', adaboost)])
        # fit models with train set
        pipe.fit(self.X, self.y)
        f1scores = cross_val_score(pipe, self.X, self.y,
                                   cv=5, scoring='f1_weighted')
        f1results = zip(model_sequence, f1scores)
        recallscores = cross_val_score(pipe, self.X, self.y,
                                       cv=5, scoring='recall_weighted')
        recallresults = zip(model_sequence, recallscores)
        precisionscores = cross_val_score(pipe, self.X, self.y,
                                          cv=5, scoring='precision_weighted')
        precisionresults = zip(model_sequence, precisionscores)
        return("f1:", f1results,
               "recall:", recallresults,
               "precision:", precisionresults)

    def predict(self, X_test, y_test):
        X_all_training = self.X.append(X_test, ignore_index=True)
        y_all_training = self.y.append(y_test, ignore_index=True)
        randomforest = ensemble.RandomForestClassifier(max_depth=1,
                                                       max_features='auto',
                                                       n_estimators=150)
        randomforest.fit(X_all_training, y_all_training)
        f1scores = cross_val_score(randomforest,
                                   X_all_training, y_all_training,
                                   cv=5, scoring='f1_weighted')
        recallscores = cross_val_score(randomforest,
                                       X_all_training, y_all_training,
                                       cv=5, scoring='f1_weighted')
        precisionscores = cross_val_score(randomforest, X_all_training,
                                          y_all_training, cv=5,
                                          scoring='f1_weighted')
        print("f1:", f1scores,
              "recall:", recallscores,
              "precision:", precisionscores)
        return randomforest

    def parameter_tuning(self, pipeline, params):
        # set models to run in pipeline
        sgd = linear_model.SGDClassifier(loss='log',
                                         learning_rate='optimal', penalty='l1')
        svc = LinearSVC()
        randomforest = ensemble.RandomForestClassifier()
        adaboost = ensemble.AdaBoostClassifier()
        pipeline = Pipeline([('sgd', sgd),
                            ('svc', svc),
                            ('randomforest', randomforest),
                            ('adaboost', adaboost)])
        params = dict(sgd__alpha=[0.0001, 0.001, 0.01],
                      svc__C=[1, 10],
                      randomforest__n_estimators=[150, 300, 450],
                      randomforest__max_depth=[1, 3, None],
                      randomforest__max_features=['auto', 'sqrt', 'log2'],
                      adaboost__n_estimators=[50, 100, 150],
                      adaboost__learning_rate=[0.5, 0.75, 1.0])
        gscv = GridSearchCV(pipeline,
                            params,
                            n_jobs=-1,
                            verbose=True,
                            cv=3,
                            scoring='recall_weighted')
        gscv.fit(self.X, self.y)
        best_model = gscv.best_estimator_
        best_params = gscv.best_params_
        best_recall_score = gscv.best_score_
        return best_model, best_params, best_recall_score
