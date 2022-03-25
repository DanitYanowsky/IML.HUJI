from turtle import title
from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    uni_inst = UnivariateGaussian()
    mu=10
    sigma=1
    m =1000
    X = np.random.normal(mu, sigma, size=m) # our samples
    uni_inst.fit(X)
    est_mu = uni_inst.mu_
    est_var= uni_inst.var_
    print(est_mu, est_var)
        
    # Question 2 - Empirically showing sample mean is consistent
    abs_mean = []
    for i in range(1,101):
        index = i*10
        uni_inst.fit(X[:index])
        abs_mean.append(np.abs(mu-uni_inst.mu_)) 

    ms = np.linspace(10, 1000, 100).astype(np.int)
    go.Figure([go.Scatter(x=ms, y=abs_mean, mode='markers+lines', name=r'$\widehat\mu-$\mu$'),],
            layout=go.Layout(title=r"$\text{Abs distance between the estimated and true value of Exp, as function of number of samples}$", 
                    xaxis_title="$m\\text{ number of samples}$", 
                    yaxis_title="r$\\text{ |est_mu-mu|}$",
                    height=300)).show()

    
    # Question 3 - Plotting Empirical PDF of fitted model
    pdf_values = uni_inst.pdf(X)
    go.Figure([go.Scatter(x=X, y=pdf_values, mode='markers', name=r'$\widehat\mu-$\mu$'),],
        layout=go.Layout(title=r"$\text{Pdf values, as function of samples}$", 
                xaxis_title="$m\\text{ sample}$", 
                yaxis_title="r$\\text{ pdf}$")).show()

def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    multi_inst = MultivariateGaussian()
    mu=np.array([0,0,4,0])
    cov=np.array([[1,0.2,0,0.5],
                  [0.2,2,0,0],
                  [0,0,1,0],
                  [0.5,0,0,1]])
    m =1000
    X = np.random.multivariate_normal(mu, cov, size=m) # our samples
    multi_inst.fit(X)
    est_mu = multi_inst.mu_
    est_cov= multi_inst.cov_
    print("mu= ",est_mu)
    print("cov= ", est_cov)

    # Question 5 - Likelihood evaluation
    size = 200
    F1 = np.linspace(-10,10,size)
    F3 = np.linspace(-10,10,size)
    mu_log=np.array(np.meshgrid(F1,0,F3,0)).T.reshape(size*size,4)
    log_value = np.apply_along_axis(MultivariateGaussian.log_likelihood, 1, mu_log, multi_inst.cov_,X)
    log_value = log_value.reshape(size,size)
    trace = go.Heatmap(x = F1, y = F3, z = log_value, type = 'heatmap', colorscale = 'Viridis')
    
    data = [trace]
    fig = go.Figure(data = data, layout=go.Layout(title=r'Heatmap of log-likelihood as function of f1 and f3',
                       xaxis_title="F1", yaxis_title="F3"))
    fig.show()

    # Question 6 - Maximum likelihood
    i,j = np.unravel_index(log_value.argmax(), log_value.shape)
    return(F1[i], F3[j])

def Q3_quiz():
    Y = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
        -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])
    print(UnivariateGaussian.log_likelihood(1,1,Y))
    print(UnivariateGaussian.log_likelihood(10,1,Y))
    return
        
if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
    # Q3_quiz()

