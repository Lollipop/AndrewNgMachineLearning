import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np



def loadData(path):
    mat = loadmat(path)
    print(mat.keys())
    return mat


def plotData(data, num=10, height=32, width=32):
    indexs = np.random.randint(0, data.shape[0]-1, 100)
    samples = data[indexs]
    print(samples.shape)
    fig, ax = plt.subplots(nrows=num, ncols=num, sharex=True, sharey=True, figsize=(10, 10))
    for r in range(num):
        for c in range(num):
            tmp = samples[num * r + c].reshape(height, width).T
            ax[r, c].imshow(tmp)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    plt.show()


def PCA(data, D=1):
    # compute the covariance matrix.
    cov = (data.T @ data) / len(data)

    # SVD
    u, s, v = np.linalg.svd(cov)

    # Dimensionality Reduction
    u_reduce = u[:, :D]
    print(u_reduce.shape)
    plotData(u_reduce.T)

    data_reduce = data @ u_reduce
    print(data_reduce.shape)
    plotData(data_reduce, height=10, width=10)

    # Recover te data
    data_recover = data_reduce @ u_reduce.T
    print(data_recover.shape)
    plotData(data_recover)



def main():
    path = 'data/ex7faces.mat'
    data = loadData(path)['X']
    print(data.shape)
    plotData(data)

    # pre-process the data
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    print(data.shape)
    PCA(data, D=100)


if __name__ == '__main__':
    main()

