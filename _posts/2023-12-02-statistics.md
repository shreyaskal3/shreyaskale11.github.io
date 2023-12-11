---
title: Statistics
date: 2023-12-02 00:00:00 +0800
categories: [Statistics]
tags: [Statistics]
---

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

The Pearson or Product Moment correlation coefficient, denoted as rxy, measures the linear association between two paired variables, x and y. It is often computed in data analysis exercises by plotting the variables and fitting a best-fit or regression line. The correlation coefficient is calculated as the ratio of covariance to the product of standard deviations. A positive value indicates a positive correlation, a negative value indicates a negative correlation, and 0 indicates no linear association. The coefficient of determination (r²) estimates the proportion of variance explained by the linear relationship.

### Calculation:

$$
r_{xy} = \frac{cov(x, y)}{s_x \cdot s_y}
$$

$$
r^2
$$

The first equation represents the Pearson correlation coefficient, and the second equation denotes the coefficient of determination.

## Scale Dependency

Scale dependency refers to the impact of scale, grouping, and arrangement on data analysis. It can be challenging to identify using simple statistical methods. For example, a scattergram showing a strong positive correlation between the number of mammal species and forest productivity might reveal a different picture when colored based on grouping parameters. This phenomenon, known as the Yule-Simpson effect or Simpson's Paradox, emphasizes the importance of considering scale dependencies in analysis.

## Confidence Intervals and Bootstrapping

Bootstrapping is a resampling technique used to estimate the distribution of a statistic. In correlation analysis, it involves generating multiple datasets by random sampling with replacement and computing correlation coefficients. Confidence intervals can be obtained from these bootstrapped distributions. Another method uses Fisher's transform on the correlation coefficient to approximate normal distribution for calculating confidence intervals. This ensures confidence limits under the assumption of bivariate normality.

## Correlation Matrix

In analyzing datasets with multiple variables, a correlation matrix can be created, showing pairwise correlation coefficients. Each variable is correlated with every other variable, resulting in a symmetric matrix. Visualization tools like correlograms help in interpreting the relationships between variables. The R package provides functions like `pairs()` for scatterplot matrices and `corrgrams` for enhanced correlation matrix presentations.

## Partial Correlation

Real-world problems involve interactions between correlated variables. Partial correlation coefficients, such as \( r_{yx.z} \), allow examining the relationship between two variables while controlling for a third. Adjustments are made automatically without grouping, providing insights into isolated relationships. The first-order partial correlation formula is extended to higher-order partial correlations.

## Correlograms

Correlograms are diagrams displaying the variation in correlation against an ordered variable, such as time or distance. The R package `corrgrams` offers a visualization tool for correlation matrices, aiding in exploratory data analysis.

**References:**
- Crawley M (2007) The R Book, John Wiley & Son, New York
- Efron B, Tibshirani R J (1993) An Introduction to the Bootstrap. Chapman and Hall, New York
- Fisher R A (1915) Frequency distribution of the values of the correlation coefficient in samples from an indefinitely large population. Biometrika, 10(4), 507–521
- Fisher R A (1921) Metron., 1(4), 3-32
- Wikipedia, Resampling: [Link](https://en.wikipedia.org/wiki/Resampling_%28statistics%29#Permutation_tests)


# Summary of Rank Correlation

## Spearman's Rank Correlation, ρ

Spearman's rank correlation is a non-parametric measure used to assess the monotonic relationship between paired observations. Given a set of paired observations (xi, yi), the data is ranked, and the difference in rankings (di) is calculated. Spearman's rank correlation coefficient (ρ) is computed using the formula:

$$ \rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)} $$
If tied values exist, the ranks are adjusted by replacing ties with their average ranks. The statistic ranges from -1 to 1, indicating perfect negative to perfect positive correlation.

The significance of ρ is often assessed using a t-distribution with (n-2) degrees of freedom.

## Kendall's Rank Correlation, τB

Kendall's rank correlation measures the concordance or discordance between pairs of observations. It counts the number of concordant (Nc) and discordant (Nd) pairs and computes the statistic as:

$$ \tau_B = \frac{N_c - N_d}{\frac{1}{2}n(n-1)} $$

The formula is adjusted for tied rankings. The statistic ranges from -1 to 1, with 1 indicating perfect positive correlation and -1 perfect negative correlation.

Significance is often evaluated by comparing the observed τB to the distribution of τB values under different arrangements. For larger sample sizes (n>10), the distribution approximates a Normal distribution, simplifying significance testing.









