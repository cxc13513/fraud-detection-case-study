#profit_curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

'''
Typical usage:

from profit_curve import profit_curve
prof_curve=profit_curve(your_probabilities, labels)
profit_curve.set_save_path('your_file.png') if you want to save file instead of plt.show()
profit_curve.run_profit_analysis()
'''


class profit_curve(object):

    def __init__(self, y_hat=None, y_true=None):
        self.y_hat=y_hat
        self.y_true=y_true
        self.default_profit_matrix=np.array([[0,-.1],[-1,0]])
        self.profits = None
        self.thresholds= None
        self.profit_matrix=None
        self.save_path=None
        self.roc_save_path=None

    def set_save_path(self, save_path):
        if type(save_path)==str:
            self.save_path = save_path
        else:
            print 'save path was not a string. was it also valid path/filename?'

    def set_roc_path(self, roc_save_path):
        if type(roc_save_path)==str:
            self.roc_save_path = roc_save_path
        else:
            print 'save path was not a string. was it also valid path/filename?'

    def set_y(self, y_hat, y_true):
        if len(y_hat)==len(y_true):
            self.y_hat=y_hat
            self.y_true=y_true
        else:
            print 'check lengths of data'

    def set_cost_benefit(self, cost_benefit_matrix):
        '''Input: numpy array
        coefficients that multiply in this order
        [[TP, FP], [FN, TN]]'''
        self.cost_benefit_matrix=cost_benefit_matrix

    # def select_elements(self, seq):
    #     """Reduce data to only 1000 of its data points"""
    #     return np.array(seq[::int(len(seq)/1000)])


    def standard_confusion_matrix(self, y_true, y_pred):
        """Make confusion matrix with format:
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
        [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred)
        return np.array([[tp, fp], [fn, tn]])

    def profit_curve(self, y_hat, y_true, cost_benefit_matrix=None):
        y_hat=self.y_hat
        y_true=self.y_true
        if cost_benefit_matrix==None:
            cost_benefit_matrix=self.default_profit_matrix

        '''
        default cost_benefit matrix is simply a penalty value of -1 for uncaught
        fraud and -.1 for accidentally calling something fraud that isn't'''


        """Function to calculate list of profits based on supplied cost-benefit
        matrix and prediced probabilities of data points and thier true labels.

        Parameters
        ----------
        cost_benefit    : ndarray - 2D, with profit values corresponding to:
                                              -----------
                                              | TP | FP |
                                              -----------
                                              | FN | TN |
                                              -----------
        predicted_probs : ndarray - 1D, predicted probability for each datapoint
                                        in labels, in range [0, 1]
        labels          : ndarray - 1D, true label of datapoints, 0 or 1

        Returns
        -------
        profits    : ndarray - 1D
        thresholds : ndarray - 1D
        """
        # Reduce number of total data points to something manageable on geological timescale.
        # predicted_probs=self.select_elements(y_hat)
        # labels=self.select_elements(y_true)
        predicted_probs=y_hat
        labels=y_true
        n_obs = float(len(labels))
        # Make sure that 1 is going to be one of our thresholds
        maybe_one = [] if 1 in predicted_probs else [1]
        thresholds = maybe_one + sorted(list(y_hat), reverse=True)
        # thresholds =  y_hat.copy()
        # thresholds[::-1].sort()
        profits = []
        for threshold in thresholds:
            y_predict = predicted_probs >= threshold
            confusion_matrix = self.standard_confusion_matrix(labels, y_predict)
            threshold_profit = np.sum(confusion_matrix * cost_benefit_matrix) / n_obs
            profits.append(threshold_profit)
        self.profits=np.array(profits)
        return np.array(profits), np.array(thresholds)

    def plot_profits(self, profits=None, thresholds=None, save_path=None):
        plt.clf()
        if profits==None:
            profits = self.profits
        if thresholds==None:
            thresholds = self.thresholds
        """Plotting function to compare profit curves of different models.

        Parameters
        ----------
        model_profits : list((model, profits, thresholds))
        save_path     : str, file path to save the plot to. If provided plot will be
                             saved and not shown.
        """

        plt.plot(thresholds, profits)

        plt.title(self.save_path)
        plt.xlabel("Thresholds")
        plt.ylabel("Profit")
        # plt.legend(loc='best')
        if save_path:
            plt.savefig(save_path)
            plt.clf()
        else:
            plt.show()
            plt.clf()

    def run_profit_analysis(self):
        if self.profit_matrix==None:
            cost_benefit=self.default_profit_matrix
        else:
            cost_benefit=self.profit_matrix
        y_hat=self.y_hat
        y_true=self.y_true
        """doesn't use max profit finder"""
        profs, thresh= self.profit_curve(y_hat, y_true, cost_benefit)
        if self.save_path:
            set_save_path= self.save_path
        else:
            set_save_path=None
        self.plot_profits(profs, thresh, save_path=set_save_path)

    def roc(self):
        return roc_curve(self.y_true, self.y_hat)

    def run_roc(self):
        plt.clf()
        fpr,tpr,thr=self.roc()
        plt.plot(fpr, tpr)
        plt.title(self.roc_save_path)
        plt.xlabel("False Positive Rates")
        plt.ylabel("True Positive Rates")
        # plt.legend(loc='best')
        if self.roc_save_path:
            plt.savefig(self.roc_save_path)
            plt.clf()
        else:
            plt.show()
            plt.clf()

    # def plot_model_profits_all(model_profits, thresholds):
    #     fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True)
    #     for ax in enumerate(axes.flat):
    #         ax.plt.plot(thresholds[i], model_profits[i])
    #     # Operate on just the top row of axes:
    #     for ax, label in zip(axes[0, :], ['A', 'B', 'C']):
    #         ax.set_title(label, size=20)
    #     # Operate on just the first column of axes:
    #     for ax, label in zip(axes[:, 0], ['D', 'E']):
    #         ax.set_ylabel(label, size=20)
    #     plt.show()
