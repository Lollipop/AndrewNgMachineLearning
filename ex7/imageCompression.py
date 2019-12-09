import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as image
from scipy.io import loadmat
from sklearn.cluster import KMeans


def showImage(path):
    img = image.imread(path)
    plt.imshow(img)
    plt.show()


def loadData(path):
    mat = loadmat(path)
    print(mat.keys())
    return mat


def K_means(K, data, iters=300, isCompression=True):
    # random initialization
    indexs = np.random.randint(0, len(data) - 1, K)

    # random centroid
    centroids = data[indexs]

    idx = np.zeros(len(data))

    isConverge = np.zeros(centroids.shape)

    iter = 0
    while np.sum(isConverge - centroids) < 1e-8 and iter < iters:
        if iter % 50 == 0:
            print('iter={}/{}'.format(iter, iters))
        iter += 1
        for i in range(len(data)):
            min = np.Inf
            for k in range(len(centroids)):
                tmp = np.sum((data[i] - centroids[k])**2)
                if min > tmp:
                    idx[i] = k
                    min = tmp

        isConverge = np.copy(centroids)

        for k in range(len(centroids)):
            tmp = data[idx == k]
            centroids[k] = np.sum(tmp, axis=0) / len(tmp)

    if isCompression:
        return centroids[idx.astype(int)]

    return data


def main():
    # show the image
    # imgPath = 'data/bird_small.png'
    # showImage(imgPath)

    dataPath = 'data/bird_small.mat'
    rawData = loadData(dataPath)
    #  you need to do this otherwise the color would be weird after clustring
    # but i don't know the reason
    rawData = rawData['A'] / 255
    height, width, depth = rawData.shape
    data = np.reshape(rawData, (height * width, depth))

    # Because the for-loop and random initialization, it sometimes will be slow and get a bad result.
    # So it can work better, if you can implement the vectorization, or you can
    # choose the best param among different initial params.

    compressedImage = K_means(16, data, iters=300, isCompression=True)
    compressedImage = compressedImage.reshape((height, width, depth))
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(rawData)
    ax[1].imshow(compressedImage)
    plt.show()

    # using the lib to do the compression
    model = KMeans(n_clusters=16, n_init=100, n_jobs=-1)
    model.fit(data)
    centroids_ = model.cluster_centers_
    labels = model.predict(data)
    compressedImage = centroids_[labels].reshape((height, width, depth))
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(rawData)
    ax[1].imshow(compressedImage)
    plt.show()


if __name__ == '__main__':
    main()

