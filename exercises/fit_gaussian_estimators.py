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
        print(uni_inst.mu_)
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
                yaxis_title="r$\\text{ pdf}$",
                height=300)).show()

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
    log_value = np.apply_along_axis(multi_inst.log_likelihood, 1, mu_log, multi_inst.cov_,X).reshape(size,size)
    
    trace = go.Heatmap(x = F1, y = F3, z = log_value, type = 'heatmap', colorscale = 'Viridis')
    data = [trace]
    fig = go.Figure(data = data)
    fig.show()

    # Question 6 - Maximum likelihood
    i,j= np.unravel_index(log_value.argmax(), log_value.shape)
    print(i,j)
    return

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
