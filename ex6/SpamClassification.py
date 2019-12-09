from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


def processData(data):
    pass


def loadData(path):
    mat = loadmat(path)
    print(mat.keys())
    return mat


def main():
    trainPath = 'data/spamTrain.mat'
    trainData = loadData(trainPath)
    X, y = trainData['X'], trainData['y'].ravel()

    testPath = 'data/spamTest.mat'
    testData = loadData(testPath)
    Xtest, ytest = testData['Xtest'], testData['ytest'].ravel()

    # train a SVM
    svc = svm.SVC()
    svc.fit(X, y)
    pred = svc.predict(Xtest)
    print(metrics.classification_report(ytest, pred))

    # train a Logistic Classification
    logistic = LogisticRegression()
    logistic.fit(X, y)
    pred = logistic.predict(Xtest)
    print(metrics.classification_report(ytest, pred))


if __name__ == '__main__':
    main()
