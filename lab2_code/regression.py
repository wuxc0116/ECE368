import numpy as np
import matplotlib.pyplot as plt
import util
from util import *
from itertools import product


def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)

    Inputs:
    ------
    beta: hyperparameter in the proir distribution

    Outputs: None
    -----
    """
    ### TODO: Write your code here

    x_axis = np.linspace(-1, 1, 100)
    y_axis = np.linspace(-1, 1, 100)

    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    x_set = np.array(list(product(x_axis, y_axis))) # cartesian product

    mean_vec = np.array([0, 0])
    covariance_mat = np.array([[beta, 0], [0, beta]])
    gauss_density = util.density_Gaussian(mean_vec, covariance_mat, x_set)
    prior = np.transpose(np.reshape(gauss_density, (100, 100)))

    plt.xlabel('a0')
    plt.ylabel('a1')
    plt.title('prior distribution')
    #plot the contour
    plt.contour(x_grid, y_grid, prior, colors='blue')
    #plot the actual point
    plt.plot(-0.1, -0.5, 'ro', label='real value')
    plt.legend()
    plt.show()

    return


def posteriorDistribution(x, z, beta, sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)

    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise

    Outputs:
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    ### TODO: Write your code here
    data_size = len(x)

    X = np.column_stack((np.ones((data_size, 1)), x))
    Cov = sigma2 * np.linalg.inv(np.matmul(X.T, X) + (sigma2 / beta) * np.identity(X.shape[1]))
    mu = (1 / sigma2) * np.matmul(Cov, np.matmul(X.T, z))
    mu = (np.transpose(mu)).squeeze()

    x_space = np.linspace(-1, 1, 100)
    y_space = np.linspace(-1, 1, 100)

    x_set = np.array(list(product(x_space, y_space)))
    x_grid, y_grid = np.meshgrid(x_space, y_space)

    gauss_density = density_Gaussian(mu, Cov, x_set)
    gauss_density_reshape = np.reshape(gauss_density, (100, 100)).T

    plt.contour(x_grid, y_grid, gauss_density_reshape, 10, colors='blue')
    plt.plot(-0.1, -0.5, 'ro', label='real value')

    plt.xlabel('a0')
    plt.ylabel('a1')

    if data_size == 1:
        plt.title('p(a|x1,z1)')
    elif data_size == 5:
        plt.title('p(a|x1,z1,..., x5,z5)')
    elif data_size == 100:
        plt.title('p(a|x1,z1,..., x100,z100)')

    plt.legend()
    plt.show()

    return (mu, Cov)


def predictionDistribution(x, beta, sigma2, mu, Cov, x_train, z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results

    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot

    Outputs: None
    -----
    """
    ### TODO: Write your code here
    data_size = len(x)
    x_array = np.array(x)
    X = np.column_stack((np.ones((data_size, 1)), x_array))

    # calculate the covariance matrix
    Xtrans_Cov = np.matmul(X, Cov) # X_transpose * Cov
    sigma2_Z = sigma2 + np.matmul(Xtrans_Cov, X.T) # sigma^2 + X_transpose * Cov * X

    std_diviation = np.sqrt(np.diag(sigma2_Z))
    mu = np.matmul(X,  mu)

    plt.errorbar(x, mu, yerr=std_diviation, ecolor='black', capsize=2, color='blue', label='prediction')
    plt.scatter(x_train, z_train, label='data Samples', color='red')
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.xlabel('x(input)')
    plt.ylabel('z(prediction)')
    if data_size == 1:
        plt.title('p(z|x,x1,z1)')
    elif data_size == 5:
        plt.title('p(z|x,x1,z1,..., x5,z5)')
    elif data_size == 100:
        plt.title('p(z|x,x1,z1,..., x100,z100)')
    plt.legend()
    plt.show()

    return


if __name__ == '__main__':
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction
    x_test = [x for x in np.arange(-4, 4.01, 0.2)]

    # known parameters
    sigma2 = 0.1
    beta = 1

    # number of training samples used to compute posterior
    ns = 5

    # used samples
    x = x_train[0:ns]
    z = z_train[0:ns]

    # prior distribution p(a)
    priorDistribution(beta)

    # posterior distribution p(a|x,z)
    mu, Cov = posteriorDistribution(x, z, beta, sigma2)

    # distribution of the prediction
    predictionDistribution(x_test, beta, sigma2, mu, Cov, x, z)