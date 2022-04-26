from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
from IMLearn.metrics import loss_functions as loss


from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"

def proccess(matrix: pd.array):
    exclude_feutrues=["sqft_lot15","sqft_lot","lat", "long","date","yr_renovated"] ##Features that are not relevant for the regression
    matrix = matrix.loc[:, ~matrix.columns.isin(exclude_feutrues)]
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    matrix = matrix[matrix['yr_built'].values > 1000]
    matrix = matrix[matrix['price'].values > 100]


    matrix = matrix.dropna()
    y=matrix['price']
    X=matrix.loc[:,~matrix.columns.isin(["price"])]
    return(X,y)

    

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
            
    sqft_living=X[:,2]
    go.Figure([go.Scatter(x=sqft_living, y=Y, mode='markers', name=r'$\price$'),],
        layout=go.Layout(title=r"$\text{Price as function of sqft living}$", 
                xaxis_title="$\\text{ sqft}$", 
                yaxis_title="r$\\text{ price}$")).show()
    
    yr_built=X[:,10]
    go.Figure([go.Scatter(x=yr_built, y=Y, mode='markers', name=r'$\yr_built$'),],
        layout=go.Layout(title=r"$\text{Price as function of Year of built}$", 
                xaxis_title="$\\text{ yr_built}$", 
                yaxis_title="r$\\text{ price}$")).show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data('datasets/house_prices.csv')

    # # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y)

    
    # Question 3 - Split samples into training- and testing sets.
    X = pd.get_dummies(X, columns=["zipcode"])

    train_X, train_y, test_X, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    train_X['price'] = train_y
    model = LinearRegression()
    average_loss = np.zeros(91)
    std_loss = np.zeros(91)
    for i in range(10,101):
        loss_i = np.zeros(10)
        for j in range(10):
            p_train = train_X.sample(frac= i/100)
            p_X = p_train.drop(["price"], axis=1)
            model.fit(p_X, p_train["price"])
            loss_i[j]=model._loss(test_X, test_y)
        average_loss[i-10] = np.mean(loss_i)
        std_loss[i-10] = (np.std(loss_i))
    ms = np.arange(100)

    go.Figure([go.Scatter(x=ms, y=average_loss, mode='markers+lines', name=r'\text{mean loss}$'),
              go.Scatter(x=ms, y=average_loss - (2 * std_loss), fill=None, mode="lines", line=dict(color="lightgrey"),
                    showlegend=False),
              go.Scatter(x=ms, y=average_loss + 2 * std_loss, fill='tonexty', mode="lines", line=dict(color="lightgrey"),
                    showlegend=False)],
        layout=go.Layout(title=r"$\text{Loss as function of percentage of samples}$", 
                xaxis_title="$\\text{ Percentage of samples from the data}$", 
                yaxis_title="r$\\text{ loss}$")).show()
        
            
        
