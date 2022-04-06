from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"

def proccess(matrix: pd.array):
    exclude_feutrues=["lat", "long","date"] ##Features that are not relevant for the regression
    matrix = matrix.loc[:, ~matrix.columns.isin(exclude_feutrues)]
    matrix=matrix.apply(pd.to_numeric, errors='coerce')
    matrix = matrix.dropna().drop_duplicates()
    return(matrix)

    
    

def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    table = pd.read_csv(filename, index_col=0)
    return proccess(table)



def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    Y, X= y.to_numpy(), X.to_numpy()
    y_sigma= np.sqrt(np.var(y))
    corr_lst = []
    for i in range(X.shape[1]):
        
        cov = np.cov(X[:, i],y)
        xi_sigma = np.sqrt(np.array(np.var(X[:,i])))
        corr_i=cov/(xi_sigma*y_sigma)
        corr_lst.append(corr_i[0,1])
    
    
    # for j in range(X.shape[1]):
    #     go.Figure([go.Scatter(x=X[:,j], y=Y, mode='markers', name=r'$\widehat\mu-$\mu$'),],
    #         layout=go.Layout(title=r"$\text{Abs distance between the estimated and true value of Exp, as function of number of samples}$", 
    #                 xaxis_title="$m\\text{ number of samples}$", 
    #                 yaxis_title="r$\\text{ |est_mu-mu|}$",
    #                 height=300)).show()
            


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    matrix = load_data('datasets/house_prices.csv')
    # print(matrix)

    # # Question 2 - Feature evaluation with respect to response
    # pd.get_dummies(matrix, columns=["zipcode"])
    # feature_evaluation(matrix, matrix["price"])
    # Question 3 - Split samples into training- and testing sets.
    frac = 0.75
    y= matrix["price"]
    new_x = matrix.drop("price", axis=1)
    train_X, train_y, test_X, test_y = split_train_test(new_x, y)
    # split_data = split_train_test(matrix.drop("price", axis=1), matrix["price"], frac)
    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
