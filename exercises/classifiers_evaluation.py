from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class
    Parameters
    ----------
    filename: str
        Path to .npy data file
    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used
    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class
    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X,y= load_dataset("datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        def callback(instance: Perceptron,X_array,y_array):
            losses.append(instance._loss(X_array,y_array))
        # Plot figure
        pre_algos = Perceptron(callback=callback)
        pre_algos._fit(X,y)
        np_iter = np.arange(1, len(losses)+1)
        go.Figure([go.Scatter(x=np_iter, y=losses, mode='markers+lines'),],
        layout=go.Layout(title=r"Losses as function of iterations", 
                xaxis_title="Iterations", 
                yaxis_title="losses")).show()



def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X,y= load_dataset("datasets/" + f)
        lda = LDA()
        gnb = GaussianNaiveBayes()
        
        
        # Fit models and predict over training set
        lda._fit(X,y)
        gnb._fit(X,y)
        lda_predict = lda._predict(X)
        gnb_predict = gnb._predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        lda_accuracy = accuracy(y, lda_predict)
        gnb_accuracy = accuracy(y, gnb_predict)
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(f"Gaussian Naive Bias classifier, accuracy = {gnb_accuracy}",
                                            f"LDA classifier, accuracy = {lda_accuracy}"))
        
        # Add `X` dots specifying fitted Gaussians' means
        feature_1 = X[:, 0]
        feature_2 = X[:, 1]
        K = lda.classes_.shape[0]
        fig.add_trace(go.Scatter(x=feature_1, y=feature_2, mode="markers", showlegend=False,
                                 marker=dict(color=gnb_predict, symbol=y, opacity=.8)), row=1, col=1)
        fig.add_trace(go.Scatter(x=feature_1, y=feature_2, mode="markers", showlegend=False,
                                 marker=dict(color=lda_predict, symbol=y, opacity=.8)), row=1, col=2)
        fig.update_layout(
            title="Gaussian Naive Bias and LDA classifiers",
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",)

        for k in range(K):
            lda_mu_k = lda.mu_[k] 
            lda_cov_k = lda.cov_  
            fig.add_trace(get_ellipse(lda_mu_k, lda_cov_k), row=1, col=2)

            gnb_mu_k = (gnb.mu_)[k]  
            gnb_cov_k = np.diag(gnb.vars_[k])  
            fig.add_trace(get_ellipse(gnb_mu_k, gnb_cov_k), row=1, col=1)

            a_lda = lda_mu_k[0]  # x of center
            b_lda = lda_mu_k[1]  # y of center
            fig.add_trace(go.Scatter(x=[a_lda], y=[b_lda], mode="markers", showlegend=False,
                                     marker=dict(color="black", symbol="x", opacity=.8, size=12)), row=1, col=2)

            a_gnb = gnb_mu_k[0]  # x of center
            b_gnb = gnb_mu_k[1]  # y of center
            fig.add_trace(go.Scatter(x=[a_gnb], y=[b_gnb], mode="markers", showlegend=False,
                                     marker=dict(color="black", symbol="x", opacity=.8, size=12)), row=1, col=1)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
