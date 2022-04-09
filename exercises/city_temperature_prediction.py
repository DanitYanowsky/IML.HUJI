import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, index_col=0, parse_dates=["Date"])
    df = df.dropna()
    day_of_year = df['Date'].dt.dayofyear
    df['DayOfYear'] = day_of_year
    df = df[df['Temp'].values > -20]
    return(df)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data('datasets/City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    israel_df = df.loc[df["City"] == "Tel Aviv"]
    g1 = px.scatter(x=israel_df["DayOfYear"], y=israel_df["Temp"], color=israel_df["Year"].astype(str))
    g1.update_traces(marker={'size': 3})
    # g1.show()
    ##to write titles in g1
    
    group_month = israel_df.groupby('Month')
    std_temp = (group_month["Temp"]).std()
    g2 = px.bar(std_temp, title="std of Temp for each month")
    # g2.show()

    # Question 3 - Exploring differences between countries
    country_month = df.groupby(['Month', 'Country'])

    country_month = country_month.agg({"Temp": ["mean", "std"]}).reset_index()
    country_month.columns = ["Month", "Country", "mean", "std"]

    g_temp = px.line(country_month, x="Month", y="mean", color=country_month["Country"].astype(str),
                        title="Mean Temp of the countries", error_y="std").show()

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(israel_df["DayOfYear"], israel_df["Temp"])
    train_X, train_y, test_X, test_y= train_X.to_numpy(), train_y.to_numpy(), test_X.to_numpy(), test_y.to_numpy()
    loss_array = np.zeros(10)
    for k in range(1,11):
        pol_reg = PolynomialFitting(k)
        pol_reg._fit(train_X, train_y)
        loss =pol_reg._loss(test_X, test_y)
        loss_array[k-1]= loss
    g4 = px.bar(loss_array, title="")
    g4.show()
    ##change the num of x
        
    # Question 5 - Evaluating fitted model on different countries
    pol_reg = PolynomialFitting(5)
    pol_reg._fit(israel_df["DayOfYear"], israel_df["Temp"])
    loss_array = np.zeros(3)
    cities = ["Amsterdam", "Amman", "Capetown"]
    for j in range(3):
        df_j = df.loc[df["City"] == cities[j]]
        loss =pol_reg._loss(df_j["DayOfYear"], df_j["Temp"])
        loss_array[j]= loss

    g5 = px.bar(x=["Netherlands", "Jordan", "South Africa"],y=loss_array, title="")
    g5.show()

