from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product
from IMLearn.metrics.loss_functions import misclassification_error

def mis_weighted(y_true, y_pred):
    vec_miss=y_true*y_pred
    vec_miss[vec_miss>0] = 0
    m = y_true.shape[0]
    num_miss = np.abs(float(np.sum(vec_miss))) #values that are incorrect
    return num_miss / m

class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        d = X.shape[1]
        err_plus,err_minus = [], []
             
        for feature in range(d):
            err_plus.append(self._find_threshold(X[:,feature],y,1))
            err_minus.append(self._find_threshold(X[:,feature],y,-1))
        min_plus, min_minus = min(err_plus, key = lambda t: t[1]), min(err_minus, key = lambda t: t[1])
        if min_plus[1]> min_minus[1]:
            
            self.sign_ = -1
            self.j_ = err_minus.index(min_minus)
            self.threshold_= min_minus[0]
        else:
            self.sign_ = 1
            self.j_ = err_plus.index(min_plus)     
            self.threshold_= min_plus[0]
        if self.threshold_ == min(X[:,self.j_]):
            self.threshold_ = -np.inf
        if self.threshold_ == max(X[:,self.j_]):
            self.threshold_ = np.inf

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """

        result = np.copy(X[:,self.j_])
        result[X[:,self.j_]>=self.threshold_] = self.sign_
        result[X[:,self.j_]<self.threshold_] = -self.sign_
        return result


    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        array_miss = []
        val_lable = np.vstack((values, labels))
        val_lable = val_lable[:, val_lable[0].argsort()]
        labels  =val_lable[1, :]
        values  =val_lable[0, :]
        m = labels.shape[0]
        answer=np.ones(m)*sign
        error = mis_weighted(labels, answer)
        for index in range(m):
            answer[index] = -sign
            if np.sign(labels[index]) == answer[index]:
                error -= np.abs(labels[index])
            else:
                error+= np.abs(labels[index])
            array_miss.append(error)
        argmin_error = np.argmin(array_miss)
        return values[argmin_error], array_miss[argmin_error]
        


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return mis_weighted(y,self._predict(X))
