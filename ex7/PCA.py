import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np


def loadData(path):
    mat = loadmat(path)
    print(mat.keys())
    return mat


def plotData(X, y):
    plt.scatter(X, y)
    plt.show()


def plotProject(data, data_recover):
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(data[:, 0], data[:, 1])
    plt.scatter(data_recover[:, 0], data_recover[:, 1])
    for i in range(len(data)):
        plt.plot([data[i, 0], data_recover[i, 0]], [data[i, 1], data_recover[i, 1]], linestyle=':', c='y')
    plt.show()

def PCA(data, D=1):
    # compute the covariance matrix.
    cov = (data.T @ data) / len(data)

    # SVD
    u, s, v = np.linalg.svd(cov)
    print(u)

    # Dimensionality Reduction
    u_reduce = u[:, :D]
    data_reduce = data @ u_reduce

    # there is a lit difference compared with the PDF(1.481), maybe because of the precision
    print(data_reduce[:5])

    # Recover te data
    data_recover = data_reduce @ u_reduce.T
    print(data_recover)
    plotData(data_recover[:, 0], data_recover[:, 1])

    plotProject(data, data_recover)

def main():
    path = 'data/ex7data1.mat'
    rawData = loadData(path)['X']
    data = np.array(rawData)
    print(data.shape)
    plotData(data[:, 0], data[:, 1])

    # normalization
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    PCA(data)


if __name__ == '__main__':
    main()



