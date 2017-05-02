from basic_df_test import create_basic_df_test_pipeline
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# models to run
logistic = linear_model.LogisticRegression(penalty='l1')
sgd = linear_model.SGDClassifier(loss="log")
randomforest = ensemble.RandomForestClassifier()

# save down sequence of models to run for future reference
model_sequence = ['logistic', 'sgd', 'randomforest']

# define a pipeline
pipe = Pipeline([('logistic', logistic), ('sgd', sgd),
                ('randomforest', randomforest)])

# pull in latest version of data to use, drop all NaNs
path = '~/Desktop/Cat/school/fraud-detection/data.json'
data = create_basic_df_test_pipeline(path)
data = data.dropna()

# create X, y, then do train_test_split
y = data.pop('fraud')
X = data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# fit models with train set
pipe.fit(X_train, y_train)
scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='f1_macro')
print(zip(model_sequence, scores))









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
