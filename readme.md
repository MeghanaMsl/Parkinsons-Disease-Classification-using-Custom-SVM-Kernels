# Parkinson's Disease Classification using Custom defined Support Vector Machine kernels and Closure Properties

---

## Overview
This Project focuses on the use of custom SVM kernels and combining them using closure properties of kernels to understand how the accuracies of a classification task change as compared to using the default provided SVM kernels such as Linear, Polynomial or Sigmoid kernels.

## Dataset
The dataset used for this project is taken from the Machine Learning repository, University of California,
Irvine (UCI) website. This dataset is composed of a total of 195 voice recordings which consist of a range of
biomedical voice measurements from 31 people, 23 with Parkinson's disease (PD) and 8 healthy individuals.
The feature set extracted from the voice recordings consists of 16 dysphonia measurements. Each column in
the table is a particular voice measure, and each row corresponds to one of 195 voice recordings from these
individuals. The main aim of the data is to discriminate healthy people from those with PD.

## Implementation
The prediction of Parkinson's Disease (PD) has been carried out by exploring various combinations of SVM kernels using a Stratified
5-fold cross validation technique. The stratified k-fold cross validation technique has been chosen to ensure
that we maintain the same ratio of PD and without PD records in the train and test splits of the data. The
models have been assessed based on the below metrics:
1. Accuracy = (TP+TN)/(TP+TN+FP+FN)
2. Precision = TP/(TP+FP)
3. Sensitivity = TP/(TP+FN)
4. Specificity = TN/(TN+FP)
5. F1 Score

To implement the chosen kernels we have pre-calculated gram matrices for every kernel based on its
respective function and used it to fit the SVM model on the training data.
Also, the following functions have been written to facilitate the combination/ scaling of the existing kernel
functions:
- Adding two kernels
- Multiplying two kernels
- Scaling a selected kernel with a constant value
- Offsetting a selected kernel with a constant value
- Raising a kernel to a constant power

The above functions can be performed on the below available kernel functions:
- Linear Kernel
- Gaussian Kernel
- Laplace Kernel
- Wavelet Kernel
- ANOVA Kernel

The execution of the program allows the user to choose from any of the above list of closure properties and
kernels to define a new kernel to analyze its performance on the PD dataset. Also, the user can go ahead and
choose to view the results for a predefined kernel that has shown optimal accuracy for this dataset (Laplace
Kernel raised to power 3 has led to an accuracy of 0.94).

### Quick recap of Key SVM Concepts to better understand the working
- Support Vector Machines are supervised learning models that enable in analyzing classification and
regression tasks. 
- They work by transforming the data into higher dimensional space using a nonlinear function mapping. 
- The transformation of data into higher dimensional space may allow the classes to become linearly separable
which usually does not happen in the original dimension space.
- In SVM the mapping of data into higher dimensional space is not done explicitly by function transformations
but is instead done using Kernel functions. This is done by representing the kernel as a Gram Matrix which follows a certain
set of conditions defined by Mercer's theorem.
- Complex kernel functions that abide by Mercer's conditions can be easily developed using the closure
properties of kernels. 
- The closure properties state that any simple kernel when added/ multiplied/ scaled/
offset-ed with any other kernel would result in a valid kernel.

## Results
The best accuracy (0.94) has been achieved by Laplace Kernel raised to power 3 and has increased the accuracy rate by 7% from the default in-built kernels provided in scikit-learn package. 
In the future, more combinations of kernels can be tried out with hyperparameter tuning perfomed 
for individual kernel parameters.



