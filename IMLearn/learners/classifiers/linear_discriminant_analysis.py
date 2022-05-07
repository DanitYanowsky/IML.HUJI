from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from IMLearn.learners import MultivariateGaussian


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.multi_k_array_ = None
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, self.pi_=np.unique(y, return_counts=True)
        d = X.shape[1]
        K=(self.classes_).shape[0]
        m = X.shape[0] 
        
        self.pi_ = self.pi_ / m
        _X = np.column_stack((X, y))
        self.mu_=np.zeros((K,d))
        self.multi_k_array = [] 
        
        ## calculate the mu:
        for k in range(K):
            k_multi=MultivariateGaussian()
            X_k = _X[_X[:, -1]==self.classes_[k]] ##Data specific to k
            k_multi.fit(X_k[:,:-1]) ##TODO slice the y??? 
            self.mu_[k] = k_multi.mu_ ##insert mu_k to the array            
            self.multi_k_array.append(k_multi)   ##insert the instance to the array
        self.fitted_=True
        
        ##calculate the cov:
        sum =0
        for i in range(m):
            k_index =int(_X[i,-1])
            vec = _X[i,:-1] - self.mu_[k_index,:]
            vec = vec.reshape((d,1))
            sum+=vec@vec.T
        self.cov_ = 1/m*(sum)
        self._cov_inv = np.linalg.inv(self.cov_)
            
        

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
        m_samples, d_features = X.shape
        k_classes = self.classes_.shape[0]
        likelihoods = np.zeros((m_samples, k_classes))
        for x_i in range(m_samples):
            for k in range(k_classes):
                exp_value=np.exp(-0.5*((X[x_i]-self.mu_[k]).T)@(self._cov_inv)@(X[x_i]-self.mu_[k]))
                const = 1/np.sqrt(((2*np.pi)**d_features)*np.linalg.det(self.cov_))
                likelihoods[x_i, k] = (1/const)*exp_value
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
