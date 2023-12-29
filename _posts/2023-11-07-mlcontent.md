---
title: Machine Learning Notes
date: 2023-11-07 00:00:00 +0800
categories: [ML]
tags: [ML]
math: true
img_path: /img/ml/
---

#

each algo optimization function there working and comparison


l1 l2 regularization

# Linear Regression

http://www.stat.yale.edu/Courses/1997-98/101/linreg.htm#:~:text=A%20linear%20regression%20line%20has,y%20when%20x%20%3D%200). 

<div align="center">
  <img src="image.png" alt="Alt text" width="500" height="300" />
</div>

## Residuals
Once a regression model has been fit to a group of data, examination of the residuals (the deviations from the fitted line to the observed values) allows the modeler to investigate the validity of his or her assumption that a linear relationship exists. Plotting the residuals on the y-axis against the explanatory variable on the x-axis reveals any possible non-linear relationship among the variables, or might alert the modeler to investigate lurking variables. In our example, the residual plot amplifies the presence of outliers.

<!-- ![Alt text](images/residuals.png) -->

## Lurking Variables
If non-linear trends are visible in the relationship between an explanatory and dependent variable, there may be other influential variables to consider. A lurking variable exists when the relationship between two variables is significantly affected by the presence of a third variable which has not been included in the modeling effort. Since such a variable might be a factor of time (for example, the effect of political or economic cycles), a time series plot of the data is often a useful tool in identifying the presence of lurking variables.

## Extrapolation
Whenever a linear regression model is fit to a group of data, the range of the data should be carefully observed. Attempting to use a regression equation to predict values outside of this range is often inappropriate, and may yield incredible answers. This practice is known as extrapolation. Consider, for example, a linear model which relates weight gain to age for young children. Applying such a model to adults, or even teenagers, would be absurd, since the relationship between age and weight gain is not consistent for all age groups.
## ROC Curve
A Receiver Operating Characteristic (ROC) curve is a graphical representation of the performance of a binary classification model at various classification thresholds. It illustrates the trade-off between the true positive rate (sensitivity) and the false positive rate (1 - specificity) as the discrimination threshold is varied.

Here's how the ROC curve works:

1. **Binary Classification**:
   - The ROC curve is commonly used for binary classification problems, where the goal is to distinguish between two classes (positive and negative).

2. **True Positive Rate (Sensitivity)**:
   - The true positive rate (TPR), also known as sensitivity or recall, is the proportion of actual positive instances correctly predicted by the model.
   - \(TPR = \frac{TP}{TP + FN}\)

3. **False Positive Rate (1 - Specificity)**:
   - The false positive rate (FPR), also known as the complement of specificity, is the proportion of actual negative instances incorrectly predicted as positive by the model.
   - \(FPR = \frac{FP}{FP + TN}\)

4. **Threshold Variation**:
   - The ROC curve is created by varying the discrimination threshold of the classifier and plotting the TPR against the FPR at each threshold.
   - As the threshold increases, the TPR usually decreases, and the FPR also decreases.

5. **Random Classifier Line**:
   - The ROC curve of a random classifier (one that makes predictions irrespective of the input features) is represented by a diagonal line from the bottom left to the top right (the line y = x).

6. **Ideal Classifier Point**:
   - The ideal classifier would have a TPR of 1 and an FPR of 0, resulting in a point at the top left corner of the ROC curve.

7. **Area Under the ROC Curve (AUC-ROC)**:
   - The AUC-ROC value provides a single scalar measure of the performance of a binary classification model. A perfect classifier has an AUC-ROC value of 1, while a random classifier has an AUC-ROC of 0.5.
   - AUC-ROC measures the area under the ROC curve.

8. **Choosing the Threshold**:
   - The choice of the threshold depends on the specific requirements of the classification task. A higher threshold may prioritize specificity, while a lower threshold may prioritize sensitivity.

9. **Use in Model Evaluation**:
   - ROC curves are widely used to evaluate the performance of classifiers, especially in situations where the class distribution is imbalanced.

the ROC curve provides a visual representation of the performance of a binary classification model across different discrimination thresholds. It is a valuable tool for understanding the trade-off between true positive rate and false positive rate and for selecting an appropriate threshold based on the specific needs of the application.


## Gradient descent 

It can be used for any optimization linear regresion as well as deep learning.

In linear regression,
$$f_{w,b}(x^{(i)}) = w*x^{(i)}+b $$

you utilize input training data to fit the parameters $w$,$b$ by minimizing a measure of the error between our predictions $f_{w,b}(x^{(i)})$ and the actual data $y^{(i)}$. The measure is called the $cost$, $J(w,b)$. In training you measure the cost over all of our training samples $x^{(i)},y^{(i)}$

$$ J_{w,b} = \frac{1}{2m}\sum\limits_{i=1}^{m-1} (f_{w,b}(x^{(i)})  - y^{(i)})^2  $$

![Desktop View](gd.png){: w="300" h="200" }


gradient descent defined as 

$$ 
\text{repeat until converge } 
\begin{cases}
w := w - \alpha \frac{\partial J(w,b)}{\partial w} \\
b := b - \alpha \frac{\partial J(w,b)}{\partial b}
\end{cases}
$$
                                                   

if $\frac{\partial J(w,b)}{\partial w}$ is slope when 
    +ve  -> w decreases (moves towards left) 
    -ve  -> w increases (moves towards right) 

the gradient descent is with gradient of cost w.r.t to w and b

$$ 
\text{repeat until converge } 
\begin{cases}
w := w - \alpha \frac{1}{m}\sum\limits_{i=1}^{m-1} (f_{w,b}(x^{(i)})  - y^{(i)})x^{(i)} \\
b := b - \alpha \frac{1}{m}\sum\limits_{i=1}^{m-1} (f_{w,b}(x^{(i)})  - y^{(i)})
\end{cases}
$$
                                               
```python
import numpy as np

def model_function(x,w,b):
    return np.dot(x,w)+b

def cost_function(x,y,w,b):
    m = x.shape[0]
    f_wb = model_function(x,w,b)
    total_loss =  (np.sum((f_wb - y)**2))/(2*m)
    return total_loss

def compute_gradient(x,y,w,b):
    m = x.shape[0]
    f_wb = model_function(x,w,b)
    dj_db = (1/m)*np.sum((f_wb - y))
    dj_dw = (1/m)*np.sum(x.T*(f_wb - y))
    return dj_dw,dj_db

def compute_gradient_descent(x,y,w,b,alpha,iterations=100):
    m = x.shape[0]

    for i in range(iterations):
        dj_dw,dj_db = compute_gradient(x,y,w,b)

        w = w - alpha *(1/m)* dj_dw
        b = b - alpha *(1/m)* dj_db

        if i%100==0:
            print(i,cost_function(x,y,w,b))
    return w,b

X_train = np.array([[2104, 5, 1, 45], 
                    [1416, 3, 2, 40], 
                    [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

w = np.random.rand(X_train.shape[1])
# w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

# w = np.zeros_like(w_init)
b = 0
alpha = 5.0e-7
# dj_db,dj_dw = compute_gradient(x_train,y_train,w,b)
w_n ,b_n = compute_gradient_descent(X_train,y_train,w,b,alpha,1000)
print(w_n ,b_n)
for i in range(X_train.shape[0]):
    print(f"prediction: {np.dot(X_train[i], w_n) + b_n:0.2f}, target value: {y_train[i]}")
```

Result:
```python
0 209314.1336159363
100 671.6664448665141
200 671.6661571235436
300 671.6658693815885
400 671.6655816406517
500 671.6652939007305
600 671.6650061618273
700 671.6647184239407
800 671.6644306870704
900 671.6641429512182
[ 0.2083132  -0.60111553 -0.18031452 -0.17953791] -0.0011102253224113373
prediction: 427.02, target value: 460
prediction: 285.62, target value: 232
prediction: 169.82, target value: 178
```

#

## Polynomial Regression


generate the data for polynomial regression
  
**Feature Scaling**

Feature scaling is a method used to normalize the range of independent variables or features of data. In data processing, it is also known as data normalization and is generally performed during the data preprocessing step.

min-max scaling

$$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

standardization(standard scores (also called z scores))

$$x_{std} = \frac{x - \mu}{\sigma}$$

where $\mu$ is the mean (average) and $\sigma$ is the standard deviation from the mean 

**Evaluate model**

Mean Squared Error (MSE)

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y_i})^2$$
$$or$$
$$MSE = \frac{1}{2n}\sum_{i=1}^{n}(y_i - f(x_i))^2$$
where 
- $y_i$ is the true value and $\hat{y_i}$ is the predicted value 
- $n$ is the number of observations

Root Mean Squared Error (RMSE)

$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y_i})^2}$$

R-squared (Coefficient of Determination)

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$
$$ SS_{res} = \sum_{i=1}^{n}(y_i - \hat{y_i})^2$$
$$ SS_{tot} = \sum_{i=1}^{n}(y_i - \bar{y_i})^2$$

where 
- $SS_{res}$ is the sum of squares of residuals 
- $SS_{tot}$ is the total sum of squares
- $y_i$ is the true value
- $\bar{y_i}$ is the mean of $y_i$
- $\hat{y_i}$ is the predicted value
- $n$ is the number of observations


**Adding Polynomial Feature**

Polynomial regression is a form of regression analysis in which the relationship between the independent variable $x$ and the dependent variable $y$ is modelled as an $n$th degree polynomial in $x$.Rising the degree of the polynomial, we can get a more complex model.And the model will be more flexible and can fit the data better.

$$y = b + w_1x + w_2x^2 + w_3x^3 + ... + w_dx^d$$

where
- $y$ is the target
- $b$ is the bias
- $w_1$ is the weight of feature $x$


```python
# Import necessary libraries
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Assuming you have already split your data into X_train, X_test, y_train, and y_test

# Initiate polynomial features
poly = PolynomialFeatures(2, include_bias=False)

# Transform the x data for proper fitting (for single variable type it returns,[1,x,x**2])
X_train_poly = poly.fit_transform(X_train[['X3 distance to the nearest MRT station']])
X_test_poly = poly.transform(X_test[['X3 distance to the nearest MRT station']])

# Create a scaler object
scaler = StandardScaler()

# Fit the scaler to the training data and transform
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

# Create a linear regression model
model = LinearRegression()
start_time = time.time()

# Train the model
model.fit(X_train_scaled, y_train)

time_taken_with_scaling = "{:.6f}".format(time.time() - start_time)
print("--- %s seconds ---" % (time_taken_with_scaling))

# Make predictions
y_pred = model.predict(X_test_scaled)

# accuracy
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))

# Plot the predictions on a scatter plot
plt.scatter(X_test_scaled[:, 1], y_test, color='blue')  # Use X_test_scaled[:, 1] for the x-axis
plt.scatter(X_test_scaled[:, 1], y_pred, color='red')    # Use X_test_scaled[:, 1] for the x-axis
plt.xlabel('X3 distance to the nearest MRT station')
plt.ylabel('Y house price of unit area')
plt.show()

```

```python
--- 0.001193 seconds ---
Mean squared error: 69.36

```


```python
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Initialize lists to store training and testing errors
training_errors = []
testing_errors = []
maxdegree = 10
# Loop through degrees 1 to 12 and store the training and testing errors
for degree in range(1, maxdegree):
    # Initiate polynomial features
    poly = PolynomialFeatures(degree, include_bias=False)

    # Transform the x data for proper fitting (for single variable type it returns,[1,x,x**2])
    X_train_poly = poly.fit_transform(X_train[['X3 distance to the nearest MRT station']])
    X_test_poly = poly.transform(X_test[['X3 distance to the nearest MRT station']])

    # Create a scaler object
    scaler = StandardScaler()

    # Fit the scaler to the training data and transform
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly)

    # Create a linear regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train_scaled, y_train)

    # Make predictions on both training and testing sets
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Calculate mean squared errors for both training and testing sets
    mse_train = mean_squared_error(y_train, y_train_pred)/2
    mse_test = mean_squared_error(y_test, y_test_pred)/2

    # Append errors to the lists
    training_errors.append(mse_train)
    testing_errors.append(mse_test)

# Plot the training and testing errors against degree
plt.plot(range(1, maxdegree), training_errors, color='blue', label='Training')
plt.plot(range(1, maxdegree), testing_errors, color='red', label='Testing')
plt.xlabel('Degree of Polynomial')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

degree = np.argmin(testing_errors) + 1
print(f"Lowest CV MSE is found in the model with degree= {degree} and training error= {training_errors[degree-1]} and testing error= {testing_errors[degree-1]}")

```


```python

# same process to choose bet diff neural network models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
def build_bc_models():

    tf.random.set_seed(20)

    model_1_bc = Sequential(
        [
            Dense(25, activation = 'relu'),
            Dense(15, activation = 'relu'),
            Dense(1, activation = 'sigmoid')
        ],
        name='model_1_bc'
    )

    model_2_bc = Sequential(
        [
            Dense(20, activation = 'relu'),
            Dense(12, activation = 'relu'),
            Dense(12, activation = 'relu'),
            Dense(20, activation = 'relu'),
            Dense(1, activation = 'sigmoid')
        ],
        name='model_2_bc'
    )

    model_3_bc = Sequential(
        [
            Dense(32, activation = 'relu'),
            Dense(16, activation = 'relu'),
            Dense(8, activation = 'relu'),
            Dense(4, activation = 'relu'),
            Dense(12, activation = 'relu'),
            Dense(1, activation = 'sigmoid')
        ],
        name='model_3_bc'
    )

    models_bc = [model_1_bc, model_2_bc, model_3_bc]
    
    return models_bc

models_bc = build_bc_models()

poly = PolynomialFeatures(degree=1, include_bias=False)

# Transform the x data for proper fitting (for single variable type it returns,[1,x,x**2])
X_train_poly = poly.fit_transform(X_train[['X3 distance to the nearest MRT station']])
X_test_poly = poly.transform(X_test[['X3 distance to the nearest MRT station']])

# Create a scaler object
scaler = StandardScaler()

# Fit the scaler to the training data and transform
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)
# Initialize lists to store training and testing errors
training_errors = []

testing_errors = []

# Loop through models and store the training and testing errors

for model in models_bc:
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=100, verbose=0)

    # Make predictions on both training and testing sets
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Calculate mean squared errors for both training and testing sets
    mse_train = mean_squared_error(y_train, y_train_pred)/2
    mse_test = mean_squared_error(y_test, y_test_pred)/2

    # Append errors to the lists
    training_errors.append(mse_train)
    testing_errors.append(mse_test)

# Plot the training and testing errors against degree
plt.plot(range(1, len(models_bc)+1), training_errors, color='blue', label='Training')
plt.plot(range(1, len(models_bc)+1), testing_errors, color='red', label='Testing')
plt.xlabel('Model Number')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

model_number = np.argmin(testing_errors) + 1
print(f"Lowest CV MSE is found in the model number= {model_number} and training error= {training_errors[model_number-1]} and testing error= {testing_errors[model_number-1]}")

```


**Diagnose a model via Bias and Variance**

Bias is the difference between the average prediction of our model and the correct value which we are trying to predict. A model with high bias pays very little attention to the training data and oversimplifies the model. It always leads to high error on training and test data.

Variance is the variability of model prediction for a given data point or a value which tells us spread of our data. A model with high variance pays a lot of attention to training data and does not generalize on the data which it hasn’t seen before. As a result, such models perform very well on training data but have high error rates on test data.

level of performance of a model:

human-level performance > training set performance > validation set performance > test set performance


- High bias and low variance: model is underfitting
- Low bias and high variance: model is overfitting
- Low bias and low variance: model is good
- High bias and high variance: model is bad

table 

|  | High Variance | Low Variance |
| --- | --- | --- |
| High Bias | Overfitting | Underfitting |
| Low Bias | Good | Good |



# 

## Bias

- **Definition**:
  - Bias refers to the error introduced by approximating a real-world problem, which may be extremely complex, by a much simpler model.
  - High bias implies that the model is too simplistic and unable to capture the underlying patterns in the data.

- **Characteristics**:
  - Models with high bias tend to oversimplify the relationships in the data and may perform poorly on both the training and unseen data.
  - Commonly associated with underfitting.

- **Examples**:
  - A linear regression model applied to a dataset with a nonlinear underlying pattern may exhibit high bias.

## Variance

- **Definition**:
  - Variance refers to the model's sensitivity to small fluctuations or noise in the training data.
  - High variance implies that the model is capturing not only the underlying patterns but also the noise in the data.

- **Characteristics**:
  - Models with high variance may perform well on the training data but poorly on unseen data, as they adapt too closely to the specific training dataset.
  - Commonly associated with overfitting.

- **Examples**:
  - A complex polynomial regression model applied to a dataset with some random noise may exhibit high variance.

## Bias-Variance Trade-Off

- **Trade-Off**:
  - There is often a trade-off between bias and variance. Increasing model complexity tends to decrease bias but increase variance, and vice versa.
  - The goal is to find the right level of model complexity that minimizes both bias and variance, resulting in optimal predictive performance on new, unseen data.

- **Underfitting and Overfitting**:
  - **Underfitting**: Occurs when a model is too simple, leading to high bias and poor performance on both training and test data.
  - **Overfitting**: Occurs when a model is too complex, capturing noise in the training data and leading to high variance. Performance on training data may be good, but it generalizes poorly to new data.

- **Model Evaluation**:
  - The bias-variance trade-off is crucial when evaluating models. Models should be assessed not only on their performance on training data but also on their ability to generalize to new, unseen data.

- **Regularization**:
  - Techniques such as regularization are used to control the trade-off between bias and variance by penalizing overly complex models.

Understanding the bias-variance trade-off is fundamental for selecting appropriate machine learning models, tuning hyperparameters, and achieving models that generalize well to new, unseen data.



# Decision Tree

## Introduction

Decision trees are a type of supervised learning algorithm that can be used for both classification and regression tasks. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

## Decision tree Steps

1. Calculate the entropy of the target.
2. Calculate the entropy of the target for each feature.
3. Calculate the information gain for each feature.
4. Choose the feature with the largest information gain as the root node.
5. Repeat steps 1 to 4 for each branch until you get the desired tree depth.

## Decision tree for classification

Entropy is the measure of impurity in a bunch of examples. The entropy of a set $S$ is defined as:

$$
H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)
$$

where $p_i$ is the proportion of the ith class.

The entropy is 0 if all samples at a node belong to the same class, and the entropy is maximal if we have a uniform class distribution. For example, in a binary class setting, the entropy is 0 if $p_1 = 1$ or $p_2 = 0$. If the classes are distributed uniformly with $p_1 = p_2 = 0.5$, the entropy is 1. Therefore, we can say that the entropy reaches its maximum value if the classes are uniformly distributed.

The following equation shows how to calculate the entropy of a dataset $D$:

$$
H(D) = -\sum_{i=1}^{c} p_i \log_2(p_i)
$$

where 
- $p_i$ is the proportion of the ith class
- $c$ is the number of classes
- $y$ is the class label.

For $y = 0$ and $y = 1$ (binary class setting), we can rewrite the equation as follows:

$$
H(D_1) = -p_1 \log_2(p_1) - (1 - p_1) \log_2(1 - p_1)
$$

where 
- $ p_1 $ is the proportion of the positive class
- $p_2 = 1 - p_1$ is the proportion of the negative class
- $D_1$ is the dataset of the left node.

The information gain is the entropy of the dataset before the split minus the weighted entropy after the split by an attribute. The following equation shows how to calculate the information gain $IG$ for a decision tree:

$$
IG(D_p, f) = H(D_p) - \sum_{j=1}^{m} \frac{N_j}{N_p} H(D_j)
$$

$$
IG(D_p, f) = H(D_p) - \sum_{j=1}^{m} \frac{N_j}{N_p} \left(-p_{j1} \log_2(p_{j1}) - p_{j2} \log_2(p_{j2})\right)
$$
where 
- $f$ is the feature to perform the split
- $D_p$ and $D_j$ are the dataset of the parent and $j$th child node
- $N_p$ is the total number of samples at the parent node
- $N_j$ is the number of samples in the $j$th child node
- $m$ is the number of child nodes

For $y = 0$ and $y = 1$ and $m = 2$(binary class setting) and two child nodes, we can rewrite the equation as follows:
$$ IG(D_p, f) = H(D_p) - \sum_{j=1}^{2} \frac{N_j}{N_p} H(D_j) $$
$$ IG(D_p, f) = H(D_p) - (\frac{N_{left}}{N_p} H(D_{left}) + \frac{N_{right}}{N_p} H(D_{right})) $$
$$ IG(D_p, f) = H(D_p) - (\frac{N_{left}}{N_p} \left(-p_{left1} \log_2(p_{left1}) - (1 - p_{left1}) \log_2(1 - p_{left1})\right) + \frac{N_{right}}{N_p} \left(-p_{right1} \log_2(p_{right1}) - (1 - p_{right1}) \log_2(1 - p_{right1})\right)) $$

where
where
- $p_{j1}$ is the proportion of the positive class in the $j$th child node
- $p_{j2} = 1 - p_{j1}$ is the proportion of the negative class in the $j$th child node

## Gini impurity 
Gini impurity is another criterion that is often used in training decision trees:

$$Gini(p) = \sum_{k=1}^{|\mathcal{Y}|} p_{k} (1 - p_{k}) = \sum_{k=1}^{|\mathcal{Y}|} p_{k} - \sum_{k=1}^{|\mathcal{Y}|} p_{k}^2 = 1 - \sum_{k=1}^{|\mathcal{Y}|} p_{k}^2$$

where $p_{k}$ is the proportion of the $k$th class.

Imformation gain for the Gini impurity is calculated as follows:

$$IG(D_p, f) = Gini(D_p) - \sum_{j=1}^{m} \frac{N_j}{N_p} Gini(D_j)$$

where $f$ is the feature to perform the split, $D_p$ and $D_j$ are the dataset of the parent and $j$th child node, $N_p$ is the total number of samples at the parent node, and $N_j$ is the number of samples in the $j$th child node.

## Classification error
The classification error is another criterion that is often used in training decision trees:

$$E = 1 - \max_k p_{k}$$

where $p_{k}$ is the proportion of the $k$th class.

The information gain ratio is another criterion that is often used in training decision trees:

$$IGR(D_p, f) = \frac{IG(D_p, f)}{H(D_p)}$$

where $f$ is the feature to perform the split, $D_p$ and $D_j$ are the dataset of the parent and $j$th child node, $N_p$ is the total number of samples at the parent node, and $N_j$ is the number of samples in the $j$th child node.

The following code implements the entropy and information gain equations:







# 

## Bayes' Theorem

**Definition**:
- Bayes' Theorem is a mathematical formula that describes the probability of an event based on prior knowledge of conditions that might be related to the event.
- It is named after Thomas Bayes, an 18th-century statistician and theologian.

**Formula**:
$ P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)} $ 

- \( P(A|B) \) is the probability of event A occurring given that B has occurred.
- \( P(B|A) \) is the probability of event B occurring given that A has occurred.
- \( P(A) \) and \( P(B) \) are the probabilities of events A and B occurring independently.

**Application**:
- Bayes' Theorem is widely used in statistics, probability theory, and machine learning, especially in Bayesian statistics and Bayesian inference.

## Naive Bayes

**Definition**:
- Naive Bayes is a classification algorithm based on Bayes' Theorem with the assumption that features are conditionally independent given the class label. This assumption simplifies the computation and leads to the term "naive."

**Types of Naive Bayes**:
- **Gaussian Naive Bayes**: Assumes that the features follow a Gaussian distribution.
- **Multinomial Naive Bayes**: Commonly used for discrete data, such as text data (e.g., document classification).
- **Bernoulli Naive Bayes**: Suitable for binary data (e.g., spam detection).

**Assumption of Independence**:
- The "naive" assumption in Naive Bayes is that features are independent, which might not hold in real-world scenarios. Despite this simplification, Naive Bayes can perform well, especially in text classification tasks.

**Application**:
- Naive Bayes is commonly used in spam filtering, text classification, sentiment analysis, and other tasks where the conditional independence assumption holds reasonably well.

## Example: Text Classification with Naive Bayes

Suppose we want to classify an email as spam (S) or not spam (NS) based on the occurrence of words "free" and "discount."

**Features**:
- \( P(\text{"free"}|S) = 0.8 \)
- \( P(\text{"discount"}|S) = 0.6 \)
- \( P(\text{"free"}|NS) = 0.1 \)
- \( P(\text{"discount"}|NS) = 0.2 \)

**Prior Probabilities**:
- \( P(S) = 0.4 \)
- \( P(NS) = 0.6 \)

**Naive Bayes Calculation**:
$$ P(S|\text{"free", "discount"}) \propto P(S) \cdot P(\text{"free"}|S) \cdot P(\text{"discount"}|S) $$
$$ P(NS|\text{"free", "discount"}) \propto P(NS) \cdot P(\text{"free"}|NS) \cdot P(\text{"discount"}|NS) $$

By comparing the probabilities, we can classify the email as spam or not spam.

These concepts are foundational in probability theory, statistics, and machine learning, providing a basis for making probabilistic inferences and classifications.



## Introduction to Navie Bayes

Navie Bayes is a classification algorithm based on Bayes' theorem. According to Bayes' theorem, the conditional probability of an event A, given another event B, is given by P(A|B) = P(B|A)P(A)/P(B). In the context of classification, we can think of P(A) as the prior probability of A, and P(A|B) as the posterior probability of A. In other words, we can think of the posterior probability as the probability of A given the data B.

## Conditional Probability

Conditional probability is the probability of an event given that another event has occurred. For example, let's say that we have two events A and B. The probability of A given that B has occurred is given by:

$$ P(A|B) = \frac{P(A \cap B)}{P(B)} $$
where P(A ∩ B) is the probability of A and B occurring together.

## Bayes' Rule:

Probability of A given B is equal to the probability of interestion of A and B divided by the probability of B.
$$ P(A|B) = \frac{P(A \cap B)}{P(B)} $$
Similarly, we can write the probability of B given A as:
$$ P(B|A) = \frac{P(A \cap B)}{P(A)} $$

If we rearrange the above equation, we get:
$$ P(A|B) = \frac{P(B|A)P(A)}{P(B)} $$

In the context of classification, we can think of P(A) as the prior probability of A, and P(A|B) as the posterior probability of A. In other words, we can think of the posterior probability as the probability of A given the data B.

## Navie Bayes Classifier

For tweets classification, we can use Navie Bayes classifier to classify tweets into positive and negative tweets. 

product of conditional probability of words in tweet of positive class and negative class:
$$ P(positive|tweet) = P(w_1|positive) * P(w_2|positive) * P(w_3|positive) * ... * P(w_n|positive) $$
$$ P(negative|tweet) = P(w_1|negative) * P(w_2|negative) * P(w_3|negative) * ... * P(w_n|negative) $$
where w1, w2, w3, ... , wn are words in tweet.
Calculate likelihood of tweet being positive and negative:
$$
\frac{P(\text{positive}|\text{tweet})}{P(\text{negative}|\text{tweet})} = 
\begin{cases} >1, & \text{positive} \\ <1, & \text{negative} \end{cases}
$$

## Laplacian Smoothing

We usually compute the probability of a word given a class as follows:

$$
P(w_i|\text{class}) = \frac{\text{freq}(w_i, \text{class})}{N_{\text{class}}} \qquad \text{class} \in \{\text{Positive}, \text{Negative}\} 
$$

However, if a word does not appear in the training, then it automatically gets a probability of 0, to fix this we add smoothing as follows

$$
P(w_i|\text{class}) = \frac{\text{freq}(w_i, \text{class}) + 1}{(N_{\text{class}} + V)}
$$

Note that we added a 1 in the numerator, and since there are $V$ words to normalize, we add $V$ in the denominator.

$N_{\text{class}}$: number of words in class

$V$: number of unique words in vocabulary

## Log Likelihood

We can use log likelihood to avoid underflow. The log likelihood is given by:

$$ \lambda(w) = \log \frac{P(w|\text{pos})}{P(w|\text{neg})} $$

where $P(w|\text{pos})$ and $P(w|\text{neg})$ are computed using Laplacian smoothing.


## train naïve Bayes classifier

1) Get or annotate a dataset with positive and negative tweets

2) Preprocess the tweets:

    Lowercase, Remove punctuation, urls, names , Remove stop words , Stemming , Tokenize sentences
    
3) Compute $\text{freq}(w, \text{class})$:

4) Get $P(w|\text{pos}), P(w|\text{neg})$

5) Get $\lambda(w)$

$$
\lambda(w) = \log \frac{P(w|\text{pos})}{P(w|\text{neg})}
$$

6) Compute $\text{logprior}$

$$
\text{logprior} = \log \frac{D_{\text{pos}}}{D_{\text{neg}}}
$$

where $D_{\text{pos}}$ and $D_{\text{neg}}$ correspond to the number of positive and negative documents respectively.

7) Compute Score 
$$\text{logprior} + \sum_{w \in \text{tweet}} \lambda(w) = 
\begin{cases} >1, & \text{positive} \\ <1, & \text{negative} \end{cases}
$$



#
