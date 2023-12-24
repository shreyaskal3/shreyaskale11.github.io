---
title: Statistics
date: 2023-12-02 00:00:00 +0800
categories: [Statistics]
tags: [Statistics]
---

# Basic

## Mean, Median, and Mode

   - **Use Cases:** The mean is often used when dealing with numerical data. It is sensitive to extreme values (outliers), so it may not be the best representation of central tendency when outliers are present.
   - **Example:** When calculating the average test score in a class.

2. **Median:**
   - **Use Cases:** The median is robust to extreme values, making it a better choice when dealing with skewed distributions or datasets with outliers. It is especially useful when the data is not normally distributed.
   - **Example:** When determining the median income in a neighborhood.

3. **Mode:**
   - **Use Cases:** The mode is suitable for categorical data or discrete values. It represents the most frequently occurring value in a dataset.
   - **Example:** When identifying the most common blood type in a population.

**Summary:**
- Use the **mean** when dealing with numerical data and the distribution is roughly symmetric.
- Use the **median** when the data is skewed or has outliers, as it is less affected by extreme values.
- Use the **mode** when dealing with categorical or discrete data to identify the most common category or value.

It's also important to note that in some cases, using multiple measures of central tendency can provide a more comprehensive understanding of the data. For example, reporting both the mean and median can give a sense of the data's typical value while also indicating whether extreme values are influencing the mean.

```python
mean = sum(x) / n

median = (x[n // 2] + x[(n - 1) // 2]) / 2 if n % 2 == 0 else x[n // 2]

count_dict = {}
for num in x:
    count_dict[num] = count_dict.get(num, 0) + 1

mode = [key for key, value in count_dict.items() if value == max(count_dict.values())][0]
```

## Variance and Standard Deviation

```python
mean = sum(x) / n

variance = sum((num - mean) ** 2 for num in x) / n

std_dev = variance ** 0.5
```

## Weighted Mean

```python
def weightedMean(X, W):
    # Write your code here
    wm = sum([X[i]*W[i] for i in range(len(X))])/sum(W)
    print("%.1f" % wm)
```

## Quartiles

```python
def median(arr):
    n = len(arr)
    return (arr[n // 2] + arr[(n - 1) // 2]) / 2 if n % 2 == 0 else arr[n // 2]

def quartiles(arr):
    arr.sort()
    n = len(arr)
    q2 = median(arr)
    q1 = median(arr[:n // 2])
    q3 = median(arr[(n + 1) // 2:])
    return q1, q2, q3
```

## Interquartile Range

```python
def interquartileRange(values, freqs):
    # Print your answer to 1 decimal place within this function
    x = [values[i] for i in range(len(values)) for _ in range(freqs[i])]
    q1, q2, q3 = quartiles(x)
    return q3 - q1
```

## Inadequate Reasoning in Statistical Analysis

#### 1. Correlation vs. Causation
- **Issue:** Assuming causation based on correlation.
- **Example:** Single-sex schools show better results for girls, but multiple factors like socioeconomic status and selectivity may influence the outcome.

#### 2. Misunderstanding Randomness
- **Issue:** Misinterpreting natural randomness, especially in sample size effects.
- **Example:** A smaller hospital with fewer daily births may show more variability in the proportion of boys born due to its smaller sample size.

#### 3. Ecological Fallacy
- **Issue:** Attributing characteristics to individuals based on group characteristics.
- **Example:** Assuming everyone in a census area earns $50,000 p.a. when individual incomes within the group can vary widely.

#### 4. Atomistic Fallacy
- **Issue:** Generalizing characteristics based on an unrepresentative sample.
- **Example:** Drawing conclusions about a population from a small or biased sample that may not accurately represent the whole.

#### 5. Misinterpretation of Visualizations
- **Issue:** Incorrectly interpreting data presented visually.
- **Example:** Misleading interpretation of prostate cancer survival rates due to outdated data, differences in screening practices, and incorrect deduction of survival rates from raw diagnosis and mortality data.

#### 6. Prosecutor's Fallacy
- **Issue:** Assuming guilt based on the probability of evidence, without considering other possibilities.
- **Example:** Claiming a one-in-a-million chance of innocence based on rare evidence, without accounting for the overall population.

#### 7. Overreliance on Past Events
- **Issue:** Assuming past random events influence future outcomes.
- **Example:** Believing that a coin is more likely to land heads after a series of tails, without considering each toss's independent probability.

#### 8. Lack of Consideration for Confounding Variables
- **Issue:** Neglecting third variables that could influence observed correlations.
- **Example:** Linking high fat consumption to breast cancer without considering other factors like genetics, lifestyle, or overall health.

#### 9. Time Dependency in Data
- **Issue:** Drawing conclusions without considering changes over time.
- **Example:** Using outdated data on prostate cancer survival rates without accounting for improvements in treatment and healthcare practices.

#### 10. Lack of Context in Data Interpretation
- **Issue:** Failing to consider contextual factors that impact conclusions.
- **Example:** Interpreting declining teenage pregnancy rates without accounting for cyclical patterns and selective reporting.

In statistical analysis, addressing these issues requires careful consideration of study design, awareness of potential biases, and a cautious approach to drawing causal relationships from observed correlations.



# Correlation in Statistics

Correlation in statistics refers to a measure of similarity between paired datasets. It does not imply causation but may suggest a potential causal relationship. For instance, the correlation between global temperature rise and carbon dioxide emissions in the context of climate change does not necessarily prove causation, as other factors may contribute.

## Independence and Correlation

If two variables measured in pairs show no apparent correlation, they are likely independent. Detection of correlation prompts questioning of independence, leading to the examination of a potential dependent relationship. The terms "independent variable" and "dependent variable" are used in this context, signifying the manner in which co-variation is explored.

## Correlation Coefficient

The degree of correlation between variables is quantified by a correlation coefficient. The most common is the Pearson correlation coefficient, ranging from -1 to +1. A coefficient of 0 indicates no correlation, +1 suggests perfect positive correlation, and -1 denotes perfect negative (inverse) correlation. Pearson's coefficient, while widely used, may produce misleading results in the presence of outliers or non-linear associations. Spearman and Kendall rank correlation coefficients offer more robust alternatives.

## Multivariate Analysis and Partial Correlation

Correlation analysis extends to more than two variables by examining relationships while holding one or more variables constant. Partial correlation analysis explores correlations between specific pairs of variables with other variables controlled.

## Time Series and Autocorrelation

Correlation techniques can be applied to data recorded in series, such as time series or spatial series. Unlike standard correlation, where two variables are analyzed, autocorrelation focuses on a single variable, studying patterns of dependency over time or distance. This is particularly useful in modeling scenarios where the assumption of independence of observations does not hold.

For further details on graphing data pairs, interpretation challenges, and data analysis difficulties, refer to Exploratory Data Analysis in this Handbook, including discussions on Anscombe's Quartet and Scale dependency.


# Pearson Correlation and Correlograms

## Pearson Correlation Coefficient (rxy)

The Pearson or Product Moment correlation coefficient, denoted as rxy, measures the linear association between two paired variables, x and y. It is often computed in data analysis exercises by plotting the variables and fitting a best-fit or regression line. The correlation coefficient is calculated as the ratio of covariance to the product of standard deviations. A positive value indicates a positive correlation, a negative value indicates a negative correlation, and 0 indicates no linear association. The `coefficient of determination (r²)` estimates the proportion of variance explained by the linear relationship.

### Calculation:
Equation represents the Pearson correlation coefficient,

Correlation coefficient is calculated as the ratio of covariance to the product of standard deviations.

$$ cov(x, y) = S_{xy} = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{n-1} $$
Where,
$ \bar{x} $ is the mean of x, $ \bar{y} $ is the mean of y, and n is the number of observations.
$$ s_x = \sqrt{\frac{\sum(x_i - \bar{x})^2}{n-1}} $$
$$
r_{xy} = \frac{cov(x, y)}{s_x \cdot s_y}
$$
$$ r_{xy} = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2 \sum(y_i - \bar{y})^2}} $$
Where,
Equation denotes the coefficient of determination.
$$
r^2
$$

```python
# Function to calculate Pearson correlation coefficient
def pearson_correlation_coefficient(x, y):
    n = len(x)

    # Calculate means
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    # Calculate covariance
    covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))

    # Calculate standard deviations
    std_dev_x = (sum((xi - mean_x) ** 2 for xi in x) / n) ** 0.5
    std_dev_y = (sum((yi - mean_y) ** 2 for yi in y) / n) ** 0.5

    # Calculate Pearson correlation coefficient
    correlation_coefficient = covariance / (n * std_dev_x * std_dev_y)

    return correlation_coefficient
```


## Scale Dependency

`Scale dependency refers to the impact of scale, grouping, and arrangement on data analysis.` It can be challenging to identify using simple statistical methods. For example, a scattergram showing a strong positive correlation between the number of mammal species and forest productivity might reveal a different picture when colored based on grouping parameters. This phenomenon, known as the `Yule-Simpson effect or Simpson's Paradox`, emphasizes the importance of considering scale dependencies in analysis.

## Confidence Intervals and Bootstrapping

`Bootstrapping is a resampling technique used to estimate the distribution of a statistic.` In correlation analysis, it involves generating multiple datasets by random sampling with replacement and computing correlation coefficients. Confidence intervals can be obtained from these bootstrapped distributions. Another method uses `Fisher's transform` on the correlation coefficient to approximate normal distribution for calculating confidence intervals. This ensures confidence limits under the assumption of bivariate normality.

## Correlation Matrix

In analyzing datasets with multiple variables, a correlation matrix can be created, showing pairwise correlation coefficients. Each variable is correlated with every other variable, resulting in a symmetric matrix. Visualization tools like correlograms help in interpreting the relationships between variables. The R package provides functions like `pairs()` for scatterplot matrices and `corrgrams` for enhanced correlation matrix presentations.

## Partial Correlation

Real-world problems involve interactions between correlated variables. Partial correlation coefficients, such as $(r_{yx.z})$, allow examining the relationship between two variables while controlling for a third. Adjustments are made automatically without grouping, providing insights into isolated relationships. The first-order partial correlation formula is extended to higher-order partial correlations.

## Correlograms

Correlograms are diagrams displaying the variation in correlation against an ordered variable, such as time or distance. The R package `corrgrams` offers a visualization tool for correlation matrices, aiding in exploratory data analysis.

# Summary of Rank Correlation

## Spearman's Rank Correlation, ρ

Spearman's rank correlation is a non-parametric measure used to assess the `monotonic relationship between paired observations`. Given a set of paired observations (xi, yi), the data is ranked, and the difference in rankings (di) is calculated. Spearman's rank correlation coefficient (ρ) is computed using the formula:

$$ \rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)} $$
where
- $ d_i = rank(x_i) - rank(y_i) $
- $ n = \text{number of observations} $

If tied values exist, the ranks are adjusted by replacing ties with their average ranks. The statistic ranges from -1 to 1, indicating perfect negative to perfect positive correlation.

The significance of ρ is often assessed using a t-distribution with (n-2) degrees of freedom.

```python
# Function to calculate Spearman's rank correlation coefficient
def spearman_rank_correlation(x, y):
    n = len(x)

    # Sorting x and y, and getting ranks
    x_s = sorted(x)
    y_s = sorted(y)
    x_r = [x_s.index(ele) + 1 for ele in x]
    y_r = [y_s.index(ele) + 1 for ele in y]

    # Calculating Spearman's rank correlation coefficient
    d_sqr = [(x_r[i] - y_r[i])**2 for i in range(n)]
    r = 1 - ((6 * sum(d_sqr)) / (n * (n**2 - 1)))

    return r
```

## Kendall's Rank Correlation, τB

Kendall's rank correlation measures the concordance or discordance between pairs of observations. It counts the number of concordant (Nc) and discordant (Nd) pairs and computes the statistic as:

$$ \tau_B = \frac{N_c - N_d}{\frac{1}{2}n(n-1)} $$

The formula is adjusted for tied rankings. The statistic ranges from -1 to 1, with 1 indicating perfect positive correlation and -1 perfect negative correlation.

Significance is often evaluated by comparing the observed τB to the distribution of τB values under different arrangements. For larger sample sizes (n>10), the distribution approximates a Normal distribution, simplifying significance testing.



# statistical concepts

1. **Descriptive Statistics:**
   - Definition: Descriptive statistics involve methods and measures that describe the data.
   - Key Components:
     - Measures of Central Tendency `(Mean, Median, Mode)`.They help us understand where the “average” or “central” point lies amidst a collection of data points.
     - Measures of Variability `(Range, Variance, Standard Deviation, Quartile Range)`.These provide valuable insights into the degree of variability or uniformity in the data.   

2. **Inferential Statistics:**
   - Definition: Inferential statistics `draw conclusions about a population from a sample`, using techniques like `hypothesis testing and regression analysis` to determine the likelihood of observed patterns occurring by chance and to estimate population parameters.
   - Importance: Enables data scientists to make data-driven decisions and formulate hypotheses about broader contexts.

3. **Probability Distributions:**
   - Definition: Probability distributions provide a structured framework for characterizing the `probabilities of various outcomes in random events`.
   - Key Distributions: `Normal, Binomial, Poisson`.
   - Significance: Essential for statistical analysis, hypothesis testing, and predictive modeling.

4. **Sampling Methods:**
   - Importance: Ensures that the sample is representative of the population in inferential statistics.
   - Common Methods:
        - **Simple Random Sampling:** Each member of the population has an equal chance of being selected for the sample through random processes.

        - **Stratified Sampling:** Population is divided into subgroups (strata), and a random sample is taken from each stratum in proportion to its size.

        - **Systematic Sampling:** Selecting every "kth" element from a population list, using a systematic approach to create the sample.

        - **Cluster Sampling:** Population is divided into clusters, and a random sample of clusters is selected, with all members in selected clusters included.

        - **Convenience Sampling:** Selection of individuals/items based on convenience or availability, often leading to non-representative samples.

        - **Purposive (Judgmental) Sampling:** Researchers deliberately select specific individuals/items based on their expertise or judgment, potentially introducing bias.

        - **Quota Sampling:** The population is divided into subgroups, and individuals are purposively selected from each subgroup to meet predetermined quotas.

        - **Snowball Sampling:** Used in hard-to-reach populations, where participants refer researchers to others, leading to an expanding sample.

5. **Regression Analysis:**
   - Definition: Quantifies the relationship between a `dependent variable and one or more independent variables`.
   - Models: `Linear Regression, Logistic Regression`.
   - Applications: Used in various fields for predicting and understanding relationships.

6. **Hypothesis Testing:**
   - Definition: Assesses claims or `hypotheses about a population using sample data`.
   - Process: Formulate null and alternative hypotheses, use statistical tests to evaluate support for the alternative hypothesis.

7. **Data Visualizations:**
   - Definition: Represents complex data visually for easy comprehension.
   - Importance: Enables the identification of trends, patterns, and outliers, facilitating data analysis and decision-making.

8. **ANOVA (Analysis of Variance):**
   - Definition: Compares means of two or more groups to determine significant differences.
   - Process: Calculates a test statistic and p-value to assess whether observed differences are statistically significant.
   - Applications: Widely used in research to assess the impact of different factors on a dependent variable.

9. **Time Series Analysis:**
   - Definition: Focuses on studying data points collected over time to understand patterns and trends.
   - Techniques: Data visualization, smoothing, forecasting, and modeling.
   - Applications: Used in finance, economics, climate science, and stock market predictions.

10. **Bayesian Statistics:**
    - Definition: Bayesian statistics treats probability as a measure of uncertainty, updating beliefs based on prior information and new evidence.
    - Applications: Particularly useful for complex, uncertain, or small-sample data in fields like machine learning, Bayesian networks, and decision analysis.





