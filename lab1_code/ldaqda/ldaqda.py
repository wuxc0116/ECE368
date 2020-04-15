import numpy as np
import matplotlib.pyplot as plt
import util

def discrimAnalysis(x, y):
    """
    Estimate the parameters in LDA/QDA and visualize the LDA/QDA models

    Inputs
    ------
    x: a N-by-2 2D array contains the height/weight data of the N samples

    y: a N-by-1 1D array contains the labels of the N samples

    Outputs
    -----
    A tuple of five elments: mu_male,mu_female,cov,cov_male,cov_female
    in which mu_male, mu_female are mean vectors (as 1D arrays)
             cov, cov_male, cov_female are covariance matrices (as 2D arrays)
    Besides producing the five outputs, you need also to plot 1 figure for LDA
    and 1 figure for QDA in this function
    """
    ### TODO: Write your code here


    x_m = x[np.where(y == 1)[0]]
    x_f = x[np.where(y == 2)[0]]
    print(x_m)
    print("+++++++++++")
    print(x_f)

    mu = np.mean(x, axis=0)
    mu_male = np.mean(x_m, axis=0)
    mu_female = np.mean(x_f, axis=0)

    N = x.shape[0]
    N_male = x_m.shape[0]
    N_female = x_f.shape[0]

    cov = np.transpose(x - mu) @ (x - mu) * (1 / N)
    cov_male = np.transpose(x_m - mu_male) @ (x_m - mu_male) * (1 / N_male)
    cov_female = np.transpose(x_f - mu_female) @ (x_f - mu_female) * (1 / N_female)

    print(mu_male, mu_female,cov,cov_male,cov_female)


    return (mu_male,mu_female,cov,cov_male,cov_female)


def misRate(mu_male,mu_female,cov,cov_male,cov_female,x,y):
    """
    Use LDA/QDA on the testing set and compute the misclassification rate

    Inputs
    ------
    mu_male,mu_female,cov,cov_male,mu_female: parameters from discrimAnalysis

    x: a N-by-2 2D array contains the height/weight data of the N samples

    y: a N-by-1 1D array contains the labels of the N samples

    Outputs
    -----
    A tuple of two elements: (mis rate in LDA, mis rate in QDA )
    """
    ### TODO: Write your code here


    return (mis_lda, mis_qda)


if __name__ == '__main__':

    # load training data and testing data
    x_train, y_train = util.get_data_in_file('trainHeightWeight.txt')
    x_test, y_test = util.get_data_in_file('testHeightWeight.txt')

    # parameter estimation and visualization in LDA/QDA
    mu_male,mu_female,cov,cov_male,cov_female = discrimAnalysis(x_train,y_train)

    # misclassification rate computation
    mis_LDA,mis_QDA = misRate(mu_male,mu_female,cov,cov_male,cov_female,x_test,y_test)







