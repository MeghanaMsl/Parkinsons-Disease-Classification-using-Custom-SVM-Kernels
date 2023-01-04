"""
Author: Meghana Yalamarty
Date: 11-03-2021
Description: Parkinson's disease classification using custom defined SVM kernels combined using closure properties
"""

import itertools
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def gaussianKernel(X1, X2, sigma=0.1):
    """ Function to calculate the gram matrix for Gaussian Kernel"""
    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            x1 = x1.flatten()
            x2 = x2.flatten()
            gram_matrix[i, j] = (np.exp(- np.sum( np.power((x1 - x2),2) ) / float( 2*(sigma**2) ) ) )

    return gram_matrix

def linearKernel(X1, X2,):
    """ Function to calculate the gram matrix for Linear Kernel"""
    return np.dot(X1, X2.T)


def mother_wavelet(x):
    return np.cos(1.75*x)*np.exp(-x**2/2)


def waveletKernel(X1, X2, a=15, c=None, h=mother_wavelet):
    """ Function to calculate the gram matrix for Wavelet Kernel"""
    gram_matrix = np.ones((X1.shape[0], X2.shape[0]))
    for d in range(X1.shape[1]):
        column_1 = X1[:, d].reshape(-1, 1)
        column_2 = X2[:, d].reshape(-1, 1)
        if c is None:
            gram_matrix *= h((column_1 - column_2.T) / a)
        else:
            gram_matrix *= h((column_1 - c) / a) * h((column_2.T - c) / a)
    return gram_matrix


def anovaKernel(X1, X2, sigma=3.0, d=1):
    """ Function to calculate the gram matrix for ANOVA Kernel"""
    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i in range(X1.shape[1]):
        column_1 = X1[:, i].reshape(-1, 1)
        column_2 = X2[:, i].reshape(-1, 1)
        gram_matrix += np.exp(-sigma * (column_1 - column_2.T) ** 2) ** d
    return gram_matrix


def laplaceKernel(X1, X2, sigma=5.0):
    """ Function to calculate the gram matrix for Laplace Kernel"""
    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i in range(X1.shape[1]):
        column_1 = X1[:, i].reshape(-1, 1)
        column_2 = X2[:, i].reshape(-1, 1)
        gram_matrix += np.exp(-abs(column_1 - column_2.T)/ float(sigma))
    return gram_matrix


def addKernels(X1, X2, k1, k2):
    """ Function to add two kernel functions"""
    return k1(X1,X2) + k2(X1,X2)


def multiplyKernels(X1, X2, k1, k2):
    """ Function to multiply two kernel functions"""
    return k1(X1, X2) * k2(X1, X2)


def offsetKernel(X1, X2, k1, c):
    """ Function to offset two kernel functions by a constant value c"""
    return k1(X1, X2) + c


def scaleKernel(X1, X2, k1, c):
    """ Function to scale two kernel functions by a constant value c"""
    return (k1(X1, X2)) * c


def kernelPower(X1, X2, k1, c):
    """ Function to raise a kernel function by a constant power c"""
    return (k1(X1, X2)) ** c


def calculate_metrics(y_pred, y_test):
    """ Function to calculate the accuracy metrics of the dataset"""
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cnf_matrix.ravel()

    # Calculate the accuracy metrics using the values returned from a confusion matrix
    print(f"\nAccuracy of the selected kernel/s is: {round(metrics.accuracy_score(y_test, y_pred), 3)}")
    print(f"Precision: {round(tp/(tp+fp),3)}")
    print(f"Sensitivity/ recall: {round(tp/(tp+fn),3)}")
    print(f"specificity/selectivity: {round(tn/(tn+fp),3)}")
    print(f"F1 score: {round(metrics.f1_score(y_test, y_pred),3)} ")

    class_names = ['without Parkinsons', 'with Parkinsons']
    plt.figure()
    generate_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')
    plt.show()

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name = 'custom kernel')
    display.plot()
    plt.show()


def generate_confusion_matrix(cnf_matrix, classes, normalize=False, title='Confusion matrix'):
    """ Function to plot the confusion matrix of the dataset"""
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(title)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    threshold = cnf_matrix.max() / 2.

    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > threshold else "black")

    plt.tight_layout()
    plt.ylabel('True values')
    plt.xlabel('Predicted values')

    return cnf_matrix


def main():
    df = pd.read_csv("parkinsons.data")
    target = df["status"].tolist()
    df.drop('status', 1, inplace=True)
    data = df.loc[:, df.columns != 'name']

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=8)

    X = np.array(data)
    y = np.array(target)
    accuracy_custom = []
    predicted_targets = np.array([])
    actual_targets = np.array([])

    kernel_mapping = {1:gaussianKernel,
                      2:waveletKernel,
                      3:linearKernel,
                      4:anovaKernel,
                      5:laplaceKernel}

    print(f"Custom SVM kernel implementaion with kernel closure properties for UCI Parkinson's vocal dataset")
    print(f"""Please select the closure property of kernel to be applied from below list: \n
        1) Add two kernels
        2) Multiply two kernels
        3) Scale kernel with a constant value
        4) offset kernel with a constant value
        5) Raise kernel to a constant power
        6) View pre-calculated results for laplace kernel raised to power 3""")

    operation_type = int(input(f"Enter the required option from list above:"))

    if operation_type in [1,2]:
        print(f"""Please select any two kernels from the below available list to add/ multiply any two kernels:\n
            1) Gaussian Kernel
            2) Wavelet Kernel
            3) Linear Kernel
            4) ANOVA Kernel
            5) Laplace Kernel""")

        kernel1= int(input(f"Enter option for kernel 1 from above list: "))
        kernel2= int(input(f"Enter option for kernel 2 from above list: "))

    if operation_type in [3,4,5]:
        print(f"""Please select a kernel from the below available list:\n
                    1) Gaussian Kernel
                    2) Wavelet Kernel
                    3) Linear Kernel
                    4) ANOVA Kernel
                    5) Laplace Kernel""")
        kernel1 = int(input(f"Enter option for kernel from above list: "))
        constant = int(input(f"Enter constant value to offset/ scale/ raise kernel to a constant power: "))

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


        clf = SVC(kernel="precomputed")

        if operation_type == 1:
            model = clf.fit(addKernels(X_train, X_train, kernel_mapping[kernel1], kernel_mapping[kernel2]), y_train)
            p = model.predict(addKernels(X_test, X_train, kernel_mapping[kernel1], kernel_mapping[kernel2]))

        if operation_type == 2:
            model = clf.fit(multiplyKernels(X_train, X_train, kernel_mapping[kernel1], kernel_mapping[kernel2]), y_train)
            p = model.predict(multiplyKernels(X_test, X_train, kernel_mapping[kernel1], kernel_mapping[kernel2]))

        if operation_type == 3:
            model = clf.fit(scaleKernel(X_train, X_train, kernel_mapping[kernel1], constant), y_train)
            p = model.predict(scaleKernel(X_test, X_train, kernel_mapping[kernel1], constant))

        if operation_type == 4:
            model = clf.fit(offsetKernel(X_train, X_train, kernel_mapping[kernel1], constant), y_train)
            p = model.predict(offsetKernel(X_test, X_train, kernel_mapping[kernel1], constant))

        if operation_type == 5:
            model = clf.fit(kernelPower(X_train, X_train, kernel_mapping[kernel1], constant), y_train)
            p = model.predict(kernelPower(X_test, X_train, kernel_mapping[kernel1], constant))

        if operation_type == 6:
            model = clf.fit(kernelPower(X_train, X_train, laplaceKernel, 3), y_train)
            p = model.predict(kernelPower(X_test, X_train, laplaceKernel, 3))


        predicted_targets = np.append(predicted_targets, p)
        actual_targets = np.append(actual_targets, y_test)

    calculate_metrics(predicted_targets, actual_targets)


if __name__ == '__main__':
    main()