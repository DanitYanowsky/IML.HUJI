import numpy as np
from IMLearn.base import BaseEstimator
from typing import Callable, NoReturn
from IMLearn.metrics.loss_functions import misclassification_error



class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None
        
    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        ##init:
        m = X.shape[0]
        D = np.zeros([m])
        D.fill(1/m)
        self.models_= []
        self.weights_ = []
        
        ## loop:
        for iter in range(self.iterations_):
            model_t = self.wl_()
            model_t._fit(X, y* D)
            vec_miss=y*model_t._predict(X)
            vec_miss[vec_miss>0] = 0
            epsilon = np.abs(float(np.sum(D*vec_miss)))
            w_t = 0.5 * np.log((1/epsilon)-1)
            D = D*(np.exp(-y*w_t*model_t._predict(X)))
            D = D / np.sum(D)
            self.weights_.append(w_t)
            self.models_.append(model_t)
        self.D_ = D
        

    def _predict(self, X):
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.partial_predict(X, self.iterations_)
        vec_answer = np.zeros([X.shape[0]])
        vec_sum = np.zeros([X.shape[0]])
        for index in range(self.iterations_):
            vec_sum+=self.models_[index]._predict(X)*self.weights_[index]
        vec_answer[vec_sum>=0] = 1
        vec_answer[vec_sum<0] = -1   
        return vec_answer         

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
        return misclassification_error(y, self._predict(X))

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        
        vec_answer = np.zeros([X.shape[0]])
        vec_sum = np.zeros([X.shape[0]])
        for index in range(T):
            vec_sum+=self.models_[index]._predict(X)*self.weights_[index]
        vec_answer[vec_sum>=0] = 1
        vec_answer[vec_sum<0] = -1   
        return vec_answer
        temp = self.iterations_
        self.iterations_=T
        vec_answer = self._predict(X)
        self.iterations_ = temp
        return vec_answer
    
    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """

        return misclassification_error(y, self.partial_predict(X,T))
