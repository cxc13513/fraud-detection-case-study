"""
Module containing model fitting code for a web application that implements a
text classification model.

When run as a module, this will load a csv dataset, train a classification
model, and then pickle the resulting model object to disk.
"""
import cPickle as pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix as cm
import numpy as np


class TextClassifier(object):
    """A text classifier model:
        - Vectorize the raw text into features.
        - Fit a naive bayes model to the resulting features.
    """

    def __init__(self):
        self._vectorizer = TfidfVectorizer(stop_words='english')
        self._classifier = MultinomialNB()

    def fit(self, X, y):
        """Fit a text classifier model.

        Parameters
        ----------
        X: A numpy array or list of text fragments, to be used as predictors.
        y: A numpy array or python list of labels, to be used as responses.

        Returns
        -------
        self: The fit model object.
        """
        x_vectorized=self._vectorizer.fit_transform(X)
        self._classifier.fit(x_vectorized,y)
        return self

    def predict_proba(self, X):
        """Make probability predictions on new data."""
        x_vectorized=self._vectorizer.transform(X)
        return self._classifier.predict_proba(x_vectorized)

    def predict(self, X):
        """Make predictions on new data."""
        x_vectorized=self._vectorizer.transform(X)
        return self._classifier.predict(x_vectorized)

    def score(self, X, y):
        """Return a classification accuracy score on new data."""
        x_vectorized=self._vectorizer.transform(X)
        return self._classifier.score(x_vectorized, y)

    def confusion_matrix(self, X, y):
        """Return a confusion matrix with format:
                      -----------
                      | TP | FP |
                      -----------
                      | FN | TN |
                      -----------
        Parameters
        ----------
        y_true : ndarray - 1D
        y_pred : ndarray - 1D

        Returns
        -------
        ndarray - 2D
        """
        x_vectorized=self._vectorizer.transform(X)
        y_pred=self._classifier.predict(x_vectorized)
        [[tn, fp], [fn, tp]] = cm(y, y_pred)
        print '-----------'
        print '| TP | FP |'
        print '-----------'
        print '| FN | TN |'
        print '-----------'
        return np.array([[tp, fp], [fn, tn]])

def get_data(filename):
    """Load raw data from a file and return training data and responses.

    Parameters
    ----------
    filename: The path to a csv file containing the raw text data and response.

    Returns
    -------
    X: A numpy array containing the text fragments used for training.
    y: A numpy array containing labels, used for model response.
    """
    text_df=pd.read_csv(filename)
    body_text=text_df['body']
    label=text_df['section_name']
    return body_text, label

# if __name__ == '__main__':
#     X, y = get_data("data/articles.csv")
#     tc = TextClassifier()
#     tc.fit(X, y)
#     with open('data/model.pkl', 'w') as f:
#         pickle.dump(tc, f)
