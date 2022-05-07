from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from IMLearn.learners import MultivariateGaussian


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.multi_k_array_ = None
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, self.pi_=np.unique(y, return_counts=True)
        m=X.shape[0]
        self.pi_ = self.pi_ / m
        _X = np.column_stack((X, y))
        K=self.classes_.shape[0]
        if len(X.shape)==1:
            d = 1
        else:
            d=X.shape[1]
        self.mu_=np.zeros((K,d))
        self.vars_=np.zeros((K,d))
        self.multi_k_array = []
               
        for k in range(K):
            k_multi=MultivariateGaussian()
            X_k = _X[_X[:, -1]==self.classes_[k]] ##Data specific to k
            k_multi.fit(X_k[:,:-1])
            self.mu_[k,:] = k_multi.mu_ ##insert mu_k to the array
            self.vars_[k,:] = np.var(X_k[:,:-1], axis=0)
            self.multi_k_array.append(k_multi)   ##insert the instance to the array          
        self.fitted_=True
     
    def _predict(self, X: np.ndarray) -> np.ndarray:
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
        return np.argmax(self.likelihood(X), axis=1) ##axis=1 is max on each line

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        m_samples = X.shape[0]
        k_classes = self.classes_.shape[0]
        likelihoods = np.zeros((m_samples, k_classes))
        for k in range(k_classes): ##for every class calculate pdf
            exp_value = -((X-self.mu_[k])**2)/(2*self.vars_[k])
            const = -0.5*np.log(2 * np.pi * self.vars_[k])
            value_to_sum =const+exp_value
            likelihoods[:, k] = np.sum(value_to_sum, 1) +np.log(self.pi_[k])
        return likelihoods

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
        from ...metrics import misclassification_error
        return misclassification_error(y,self.predict(X))
