from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    Y1   = np.random.normal(10, 1, size=1000)
    X = np.linspace(-10, 10, 1000)

    make_subplots(rows=1, cols=2)\
        .add_traces([go.Scatter(x=X, y=Y1, mode='lines', marker=dict(color="black"), showlegend=False),
                    ], rows=[1,1], cols=[1,2])\
        .add_traces([go.Scatter(x=X, y=Y1,   mode='markers', marker=dict(color="red"),  name="$\\mathcal{N}\\left(0,1\\right)$"),], 
        rows=1, cols=[1,1,2,2])\
        .update_layout(title_text=r"$\text{(2) Generating Data From Probabilistic Model}$", height=300)\
        .show()
    # Question 2 - Empirically showing sample mean is consistent
    raise NotImplementedError()

    # Question 3 - Plotting Empirical PDF of fitted model
    raise NotImplementedError()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
