---
title: Machine Learning Library
date: 2023-11-07 00:00:00 +0800
categories: [ML]
tags: [ML]
math: true
---

#

https://scikit-learn.org/stable/modules/linear_model.html
Linear Models
The following are a set of methods intended for regression in which the target value is expected to be a linear combination of the features.
Ordinary Least Squares
LinearRegression fits a linear model with coefficients w to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.

```
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
reg.coef_
```

Non-Negative Least Squares
It is possible to constrain all the coefficients to be non-negative, which may be useful when they represent some physical or naturally non-negative quantities (e.g., frequency counts or prices of goods). LinearRegression accepts a boolean positive parameter: when set to True Non-Negative Least Squares are then applied.

Ridge regression and classification
1.1.2.1. Regression
Ridge regression addresses some of the problems of Ordinary Least Squares by imposing a penalty on the size of the coefficients. The ridge coefficients minimize a penalized residual sum of squares:

from sklearn import linear*model
reg = linear_model.Ridge(alpha=.5)
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
reg.coef*
reg.intercept\_

Note that the class Ridgeallows for the user to specify that the solver be automatically chosen by setting solver="auto". When this option is specified, Ridgewill choose between the "lbfgs", "cholesky", and "sparse_cg"solvers. Ridgewill begin checking the conditions shown in the following table from top to bottom. If the condition is true, the corresponding solver is chosen.

Solver
Condition
‘lbfgs’
The positive=Trueoption is specified.
‘cholesky’
The input array X is not sparse.
‘sparse_cg’
None of the above conditions are fulfilled.

1.1.2.2. Classification

The Ridge regressor has a classifier variant: RidgeClassifier. This classifier first converts binary targets to {-1, 1} and then treats the problem as a regression task, optimizing the same objective as above. The predicted class corresponds to the sign of the regressor’s prediction. For multiclass classification, the problem is treated as multi-output regression, and the predicted class corresponds to the output with the highest value.

It might seem questionable to use a (penalized) Least Squares loss to fit a classification model instead of the more traditional logistic or hinge losses. However, in practice, all those models can lead to similar cross-validation scores in terms of accuracy or precision/recall, while the penalized least squares loss used by the RidgeClassifierallows for a very different choice of the numerical solvers with distinct computational performance profiles.

The RidgeClassifiercan be significantly faster than e.g. LogisticRegressionwith a high number of classes because it can compute the projection matrix only once.

Lasso

The Lasso is a linear model that estimates sparse coefficients. It is useful in some contexts due to its tendency to prefer solutions with fewer non-zero coefficients, effectively reducing the number of features upon which the given solution is dependent. For this reason, Lasso and its variants are fundamental to the field of compressed sensing. Under certain conditions, it can recover the exact set of non-zero coefficients (see Compressive sensing: tomography reconstruction with L1 prior (Lasso)).

from sklearn import linear_model
reg = linear_model.Lasso(alpha=0.1)
reg.fit([[0, 0], [1, 1]], [0, 1])
reg.predict([[1, 1]])

Kernel ridge regression
