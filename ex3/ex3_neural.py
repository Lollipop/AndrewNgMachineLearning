import numpy as np
from scipy.io import loadmat
from sklearn.metrics import classification_report#这个包是评价报告


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def main():
    theta_path = 'ex3weights.mat'
    thetas = loadmat(theta_path)
    print(thetas['Theta1'].shape, thetas['Theta2'].shape)
    theta1 = np.array(thetas['Theta1'])
    theta2 = np.array(thetas['Theta2'])

    data_path = 'ex3data1.mat'
    data = loadmat(data_path)

    # 这里不能把 10 替换成 0， 因为网络训练就是用的 10
    X = np.array(data['X'])
    y = np.array(data['y'])

    X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)

    layer1 = X @ theta1.T
    layer1 = np.insert(layer1, 0, values=np.ones(layer1.shape[0]), axis=1)
    layer1 = sigmoid(layer1)
    print(layer1.shape)

    layer2 = sigmoid(layer1 @ theta2.T)
    print(layer2.shape)
    print(layer2)

    # matlab下标从 1 开始，因此需要加 1
    y_preds = np.argmax(layer2, axis=1) + 1
    print(y_preds.shape)

    correct = (y.ravel() == y_preds).astype(int)
    accuracy = np.mean(correct)
    print(accuracy)

    # using the lib for reporting
    print(classification_report(y, y_preds))





if __name__ == '__main__':
    main()