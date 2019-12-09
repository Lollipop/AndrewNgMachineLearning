from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def loadData(path):
    mat = loadmat(path)
    print(mat.keys())
    return mat


def plotData(data):
    plt.scatter(data[:, 0], data[:, 1], c='b', s=50)
    plt.show()


def K_means(K, data, iters=10):
    # random initialization
    indexs = np.random.choice(np.arange(len(data)), K)
    print(indexs)

    converge_process = []

    # random centroid
    centroids = data[indexs]

    # centroids = np.array([[3, 3], [6, 2], [8, 5]])
    converge_process.append(centroids)
    # print(centroids)
    idx = np.zeros(len(data))

    print(converge_process)

    for iter in range(iters):
        for i in range(len(data)):
            min = np.Inf
            for k in range(len(centroids)):
                tmp = np.sum((data[i] - centroids[k])**2)
                if min > tmp:
                    idx[i] = k
                    min = tmp

        for k in range(len(centroids)):
            tmp = data[idx == k]
            centroids[k] = np.sum(tmp, axis=0) / len(tmp)

        converge_process.append(np.copy(centroids))

    fig, ax = plt.subplots(figsize=(10, 10))

    for k in range(K):
        m = len(converge_process)
        x = np.zeros(m)
        y = np.zeros(m)
        for i in range(m):
            x[i] = converge_process[i][k][0]
            y[i] = converge_process[i][k][1]

        ax.scatter(x, y, marker='+', s=500)

    for k in range(len(centroids)):
        tmp = data[idx == k]
        x = tmp[:, 0]
        y = tmp[:, 1]
        ax.scatter(x, y)

    plt.show()
    return idx, centroids


def main():
    path = 'data/ex7data2.mat'
    data = loadData(path)['X']
    # plotData(data)
    _, centroids = K_means(3, data)

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(data)
    centroids_ = kmeans.cluster_centers_
    labels = kmeans.predict(data)
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(data[:, 0], data[:, 1], c=labels)
    plt.show()

    print(centroids)
    print(centroids_)


if __name__ == '__main__':
    main()
