---
title: Decision Tree
date: 2023-11-07 00:00:00 +0800
categories: [ML, Decision Tree]
tags: [ML]
math: true
---

# Q&A

<details>
  <summary>
    How is a split done in a decision tree?
  </summary>

The split in a decision tree is done by selecting the feature and threshold that results in the highest information gain.

Steps

- Calculate the Entropy of the Target
- Calculate the Entropy for Each Feature: For each feature, the entropy is calculated for different possible splits. For a continuous feature, potential split thresholds can be midpoints between sorted feature values. For a categorical feature, splits can be binary or multi-way.
- Calculate the Information Gain for Each Feature: Information gain is the reduction in entropy after a split. weighted avg of entropy of each class in each branch.
- Choose the Feature with the Largest Information Gain as the Root Node
- Repeat Above Steps for Each Branch Until the Desired Tree Depth is Achieved: until stopping criteria are met
  - maximum depth
  - minimum samples per leaf
  - no further gain in information it is below threshold
  - number of example in node is below threshold.

This can be achieved through various metrics such as Gini impurity, entropy, or variance reduction, depending on the type of decision tree (classification or regression).

</details>

<details>
  <summary>
    When do you stop in a decision tree?
  </summary>

Stopping criteria for a decision tree include reaching

- a maximum depth,
- having a minimum number of samples per leaf,
- no further gain in information it is below threshold
- number of example in node is below threshold.

</details>

<details>
  <summary>
    Why one hot encoded needed for decision tree?
  </summary>
  
One-hot encoding is not strictly necessary for decision trees, but it can be beneficial depending on the implementation and the nature of the categorical data.

If a categorical feature has three classes (e.g., floppy, pointy, oval), a split on this feature would create three branches (one for each class). One-hot encoding transforms this categorical feature into three binary features:

    •	Is_floppy (1 if floppy, 0 otherwise)
    •	Is_pointy (1 if pointy, 0 otherwise)
    •	Is_oval (1 if oval, 0 otherwise)

This allows the decision tree algorithm to handle the categorical data without any additional modifications, enabling it to make binary splits based on each class separately. One-hot encoding ensures that the decision tree can process categorical features as numerical data, avoiding issues with algorithms that may not natively support categorical inputs or might misinterpret ordinal relationships in label-encoded data.

1. Handling High Cardinality: If a categorical feature has many possible values (high cardinality), one-hot encoding can help manage the complexity. Instead of creating a multi-way split, which can lead to overfitting and high computational cost, one-hot encoding allows the decision tree to handle each category individually as binary splits.
2. Algorithm Compatibility: Some decision tree implementations may not natively support categorical features and require numerical input. One-hot encoding transforms categorical features into a format that these algorithms can process without further modifications.
3. Avoiding Implicit Ordering: When using label encoding (assigning numerical labels to categories), the decision tree might interpret the numerical labels as having an inherent order, which can lead to incorrect splits. One-hot encoding avoids this issue by treating each category as a separate binary feature.
4. Binary Splits: Decision trees typically create binary splits. For categorical data with more than two categories, one-hot encoding allows the tree to split the data into binary decisions, making the tree structure simpler and potentially more interpretable.

</details>

<details>
  <summary>
    How does a decision tree handle continuous data?
  </summary>

Decision trees handle continuous data by creating binary splits based on threshold values. Here’s how the process works:

1. Identify Potential Splits: For a continuous feature, the algorithm identifies potential split points. These are typically chosen as midpoints between each pair of consecutive values in the sorted list of the feature’s unique values.
2. Calculate Information Gain: For each potential split point, the algorithm calculates the information gain (or reduction in impurity) that would result from splitting the data at that point. The impurity can be measured using metrics like Gini impurity or entropy.
3. Choose the Best Split: The split point that maximizes the information gain is selected. This means the algorithm evaluates all potential splits and chooses the one that best separates the data into two groups that are more homogeneous in terms of the target variable.
4. Create Binary Decision: The chosen split creates a binary decision rule. For example, if the chosen split point for a feature X is x , the decision rule will be X \leq x . The data is then divided into two subsets: one where the feature value is less than or equal to x , and one where it is greater than x .
5. Recursively Apply the Process: The splitting process is recursively applied to each subset until stopping criteria are met (e.g., maximum depth, minimum samples per leaf, or no further gain in information).

Example:
Consider a continuous feature “age” with values [25, 30, 35, 40]. Potential split points could be 27.5, 32.5, and 37.5. The algorithm calculates the information gain for each split point and selects the one with the highest gain. If the best split point is 32.5, the decision rule will be “age ≤ 32.5”.

</details>

<details>
  <summary>
    How can decision trees be used for regression?
  </summary>

Decision trees can be used for regression tasks, where the goal is to predict a continuous target variable. Here is how a decision tree regression model works:

1. **Calculate the Variance of the Target**: Instead of using measures like Gini impurity or entropy (which are used for classification), decision tree regression uses the variance (or mean squared error) of the target variable to measure the impurity of a node.

2. **Determine the Best Split**: For each feature and potential split point, the algorithm calculates the reduction in variance that would result from the split. The split that results in the largest reduction in variance is chosen.

3. **Create Decision Rules**: Similar to classification trees, the chosen split creates a decision rule. For continuous features, this is a binary split based on a threshold (e.g., \( X \leq x \)).

4. **Repeat the Process**: The splitting process is recursively applied to each subset of the data, creating branches of the tree, until a stopping criterion is met (e.g., maximum depth, minimum samples per leaf, or no further reduction in variance).

5. **Make Predictions**: Once the tree is built, predictions for new data are made by traversing the tree from the root to a leaf node, following the decision rules at each node. The prediction for a leaf node is typically the mean of the target values of the training samples that reach that leaf.

Example:
Consider a dataset with features "square footage" and "number of bedrooms" and a target variable "house price". The decision tree regression algorithm will:

- Calculate the variance of house prices.
- Evaluate potential split points for "square footage" and "number of bedrooms".
- Choose the split that results in the largest reduction in variance (e.g., "square footage ≤ 1500").
- Split the data into two subsets based on the decision rule.
- Recursively apply the process to each subset until stopping criteria are met.
- Predict house prices for new data by traversing the tree and using the mean price of the leaf nodes.

In summary, decision tree regression models use the same basic principles as classification trees but focus on reducing variance (or mean squared error) to create splits that best predict a continuous target variable.

</details>

<details>
  <summary>
    What is overfitting in decision trees?
  </summary>

Overfitting in decision trees occurs when the model learns not only the underlying pattern but also the noise in the training data. This results in a model that performs well on the training data but poorly on unseen data. Pruning, setting maximum depth, and limiting the minimum number of samples per split are common ways to prevent overfitting.

</details>

<details>
  <summary>
    How do you handle missing values in decision trees?
  </summary>

Missing values can be handled by imputing them with the mean, median, or mode of the feature, or by using algorithms that can handle missing values natively. Some decision tree implementations can split on features with missing values by assigning the split based on the majority or by distributing the samples across possible values.

</details>

<details>
  <summary>
    What is pruning in decision trees?
  </summary>

Pruning is the process of removing parts of the tree that do not provide significant power to classify instances. Pruning can be done preemptively (pre-pruning) by setting constraints like maximum depth or minimum samples per node, or post-pruning by trimming branches from a fully grown tree based on certain criteria.

</details>

<details>
  <summary>
    What is the difference between Gini impurity and entropy?
  </summary>

Gini impurity and entropy are both measures of the impurity or randomness in a dataset. Gini impurity measures the probability of misclassifying a randomly chosen element, whereas entropy measures the amount of uncertainty or disorder. Both metrics are used to determine the best split in a decision tree.

</details>

<details>
  <summary>
    What is the importance of feature selection in decision trees?
  </summary>

Feature selection is crucial in decision trees as irrelevant or redundant features can lead to overfitting and increase computational complexity. Decision trees inherently perform feature selection by choosing the best features for splits, but preprocessing steps like removing low-variance features can further improve model performance.

</details>

<details>
  <summary>
    How do ensemble methods improve decision tree performance?
  </summary>

Ensemble methods like bagging, boosting, and random forests improve decision tree performance by combining multiple trees to reduce variance (bagging, random forests) or bias (boosting). These methods lead to more robust models that generalize better to unseen data.

</details>

<details>
  <summary>
    What are the advantages and disadvantages of decision trees?
  </summary>

**Advantages**:

- Easy to understand and interpret.
- Requires little data preprocessing.
- Can handle both numerical and categorical data.
- Non-parametric and flexible.

**Disadvantages**:

- Prone to overfitting.
- Can create biased trees if some classes dominate.
- Sensitive to small changes in data.
- Can be computationally expensive with large datasets.

</details>

---

# Decision Tree

## Introduction

Decision trees are a type of supervised learning algorithm that can be used for both classification and regression tasks. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

## Decision tree Steps

1. Calculate the entropy of the target.
2. Calculate the entropy of the target for each feature.
3. Calculate the information gain for each feature.
4. Choose the feature with the largest information gain as the root node.
5. Repeat steps 1 to 4 for each branch until you get the desired tree depth.

<div align="center">
  <img src="https://media.licdn.com/dms/image/D4E22AQGdNfTiCXyhyg/feedshare-shrink_800/0/1718541617124?e=1721260800&v=beta&t=Uc1SuotysQKrXjcPzA3YfnRfzzG99TNfKpChJgVbnJk" alt="gd" width="600" height="700" />
</div>
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
$$ IG(D*p, f) = H(D_p) - \sum*{j=1}^{2} \frac{N*j}{N_p} H(D_j) $$
$$ IG(D_p, f) = H(D_p) - (\frac{N*{left}}{N*p} H(D*{left}) + \frac{N*{right}}{N_p} H(D*{right})) $$
$$ IG(D*p, f) = H(D_p) - (\frac{N*{left}}{N*p} \left(-p*{left1} \log*2(p*{left1}) - (1 - p*{left1}) \log_2(1 - p*{left1})\right) + \frac{N*{right}}{N_p} \left(-p*{right1} \log*2(p*{right1}) - (1 - p*{right1}) \log_2(1 - p*{right1})\right)) $$

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

Both decision trees and gradient boosting are machine learning techniques that can be used for making predictions by dividing the input space.
