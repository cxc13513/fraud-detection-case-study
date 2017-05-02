from sklearn.base import TransformerMixin
from sklearn.cross_validation import PredefinedSplit
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer


class CustomMixin(TransformerMixin):
    def get_params(self, **kwargs):
        return dict()

    def set_params(self, **kwargs):
        for key in self.get_params():
            setattr(self, key, kwargs[key])



p = Pipeline([
        ('filter', FilterColumns()),
        ('type_change', DataType()),
        ('replace_outliers', ReplaceOutliers()),
        ('compute_age', ComputeAge()),
        ('nearest_average', ComputeNearestMean()),
        ('columns', ColumnFilter()),
        ('lm', LinearRegression())
    ])

 def rmsle(y_hat, y):
        target = y
        predictions = y_hat
        log_diff = np.log(predictions+1) - np.log(target+1)
        return np.sqrt(np.mean(log_diff**2))

    # GridSearch
    params = {'nearest_average__window': [3, 5, 7]}

    # Turns our rmsle func into a scorer of the type required
    # by gridsearchcv.
    rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

    gscv = GridSearchCV(p, params,
                        scoring=rmsle_scorer,
                        cv=cross_val)
    clf = gscv.fit(df.reset_index(), y)

    print 'Best parameters: %s' % clf.best_params_
    print 'Best RMSLE: %s' % clf.best_score_

    test = pd.read_csv('data/test.csv')
    test = test.sort_values(by='SalesID')

    test_predictions = clf.predict(test)
    test['SalePrice'] = test_predictions
    outfile = 'data/solution_benchmark.csv'
    test[['SalesID', 'SalePrice']].to_csv(outfile, index=False)

'''
PARAMETER TUNING

# specify CV #folds
numFolds = 10
kf = KFold(n_splits=num_folds, shuffle=True)
scores = []
for train, test in kf.split(X):
  model.fit(train[train_idx], train_y[train_idx])
  scores.append(model.score(train[cv_idx], train_y[cv_idx]))

print("Score: {}".format(np.mean(scores)))



good pipeline foc: http://scikit-learn.org/stable/modules/pipeline.html

if include gridsearch in here:
https://learn.galvanize.com/content/gSchool/dsi-curriculum/master/regression-case-study/solutions/pair/soln1/heavy_equipment_model.py
# define parameters for pipeline models
# set parameter ranges for models
parameters = {
    'logistic__C': np.logspace(1, 3, 5),
    'logistic__n_jobs':
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': (0.00001, 0.000001),
    'clf__penalty': ('l2', 'elasticnet'),
    #'clf__n_iter': (10, 50, 80),
}

def grid_search(model, feature_dict, train_x, train_y):
   gscv = GridSearchCV(
       model,
       feature_dict,
       n_jobs=-1,
       verbose=True,
       scoring='f1'
   )

 f1_beta
'''
