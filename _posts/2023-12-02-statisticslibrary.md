---
title: Statistics
date: 2023-12-02 00:00:00 +0800
categories: [StatisticsLibrary]
tags: [Statistics]
math: True
---

# Statistical functions (scipy.stats)

This module contains a large number of probability distributions, summary and frequency statistics, correlation functions and statistical tests, masked statistics, kernel density estimation, quasi-Monte Carlo functionality, and more.

Hypothesis Tests and related functions
SciPy has many functions for performing hypothesis tests that return a test statistic and a p-value, and several of them return confidence intervals and/or other related information.

The headings below are based on common uses of the functions within, but due to the wide variety of statistical procedures, any attempt at coarse-grained categorization will be imperfect. Also, note that tests within the same heading are not interchangeable in general (e.g. many have different distributional assumptions).

# sns

### EDA

#### relationship between two variables

Figure-level interface for drawing relational plots onto a FacetGrid.

This function provides access to several different axes-level functions that show the relationship between two variables with semantic mappings of subsets. The kind parameter selects the underlying axes-level function to use:

scatterplot() (with kind="scatter"; the default)

lineplot() (with kind="line")

```python
sns.kdeplot(data=df.iloc[:,:4])
plt.show()
```

#### categorical plots

Figure-level interface for drawing categorical plots onto a FacetGrid.

This function provides access to several axes-level functions that show the relationship between a numerical and one or more categorical variables using one of several visual representations. The kind parameter selects the underlying axes-level function to use.

Categorical scatterplots:

- stripplot() (with kind="strip"; the default)
- swarmplot() (with kind="swarm")

Categorical distribution plots:

- boxplot() (with kind="box")
- violinplot() (with kind="violin")
- boxenplot() (with kind="boxen")

Categorical estimate plots:

- pointplot() (with kind="point")
- barplot() (with kind="bar")
- countplot() (with kind="count")

```python

sns.catplot(data=df, x="age", y="class")
```

#### Plot univariate or bivariate distributions using kernel density estimation.

A kernel density estimate (KDE) plot is a method for visualizing the distribution of observations in a dataset, analogous to a histogram. KDE represents the data using a continuous probability density curve in one or more dimensions.

The approach is explained further in the user guide.

Relative to a histogram, KDE can produce a plot that is less cluttered and more interpretable, especially when drawing multiple distributions. But it has the potential to introduce distortions if the underlying distribution is bounded or not smooth. Like a histogram, the quality of the representation also depends on the selection of good smoothing parameters.

```python
sns.kdeplot(data=df.iloc[:,:4])
plt.show()
```

#### Plot empirical cumulative distribution functions.

An ECDF represents the proportion or count of observations falling below each unique value in a dataset. Compared to a histogram or density plot, it has the advantage that each observation is visualized directly, meaning that there are no binning or smoothing parameters that need to be adjusted. It also aids direct comparisons between multiple distributions. A downside is that the relationship between the appearance of the plot and the basic properties of the distribution (such as its central tendency, variance, and the presence of any bimodality) may not be as intuitive.

```python
sns.ecdfplot(data=df, x="F19", hue="C",stat="count")
plt.show()
sns.ecdfplot(data=df, x="F19", hue="C", stat="proportion")
```

#### pairwise relationships in a dataset

Subplot grid for plotting pairwise relationships in a dataset.

```python
g = sns.PairGrid(df.iloc[:,19:],hue="C",corner=True)
g.map(sns.scatterplot)

x_vars = ["body_mass_g", "bill_length_mm", "bill_depth_mm", "flipper_length_mm"]
y_vars = ["body_mass_g"]
g = sns.PairGrid(penguins, hue="species", x_vars=x_vars, y_vars=y_vars)
g.map_diag(sns.histplot, color=".3")
g.map_offdiag(sns.scatterplot)
g.add_legend()
```

#### multi-plot grids

https://seaborn.pydata.org/tutorial/axis_grids.html

#### boxplot

<div align="center" >
  <img src="https://media.licdn.com/dms/image/D4E22AQEQXTE4CmiIUQ/feedshare-shrink_800/0/1716584151496?e=1721865600&v=beta&t=NvAuLsanb8ANMmhPR4A4UIflcsqkH8yOYGKv0o1zVCQ" alt="gd" width="500" height="500" />
</div>
### regression-fits

Plot data and a linear regression model fit.
https://seaborn.pydata.org/tutorial/regression.html#estimating-regression-fits

```python
sns.regplot(data = df, x="weight_kg", y="height_cm", ci=95)
sns.lmplot(x="size", y="tip", data=tips, x_estimator=np.mean);
sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips);
sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips,
           markers=["o", "x"], palette="Set1");
sns.lmplot(x="total_bill", y="tip", hue="smoker", col="time", data=tips);
sns.lmplot(x="total_bill", y="tip", hue="smoker",
           col="time", row="sex", data=tips, height=3);
```

Plotting a regression in other contexts

```python
sns.jointplot(x="total_bill", y="tip", data=tips, kind="reg");
sns.pairplot(tips, x_vars=["total_bill", "size"], y_vars=["tip"],
             height=5, aspect=.8, kind="reg");
sns.pairplot(tips, x_vars=["total_bill", "size"], y_vars=["tip"],
             hue="smoker", height=5, aspect=.8, kind="reg");
```

# autogluon

```python
!pip install autogluon.eda
import autogluon.eda.auto as auto
auto.dataset_overview(train_data=train, test_data=test, label='C',sample=None)

auto.target_analysis(train_data=train,test_data=test, label='C',sample=None)

fit_ = auto.quick_fit(train_data=train,test_data=test, label='C', show_feature_importance_barplots=True,sample= None,return_state=True)
print(fit_.model_evaluation.keys())
fit_.model_evaluation['undecided']

auto.partial_dependence_plots(train_data=train, label='C')
auto.covariate_shift_detection(train_data=train, test_data=test, label='C')
```

Partial Dependence Plots

Individual Conditional Expectation (ICE) plots complement Partial Dependence Plots (PDP) by showing the relationship between a feature and the model's output for each individual instance in the dataset. ICE lines (blue) can be overlaid on PDPs (red) to provide a more detailed view of how the model behaves for specific instances. Here are some points on how to interpret PDPs with ICE lines:

- Central tendency: The PDP line represents the average prediction for different values of the feature of interest. Look for the overall trend of the PDP line to understand the average effect of the feature on the model's output.
- Variability: The ICE lines represent the predicted outcomes for individual instances as the feature of interest changes. Examine the spread of ICE lines around the PDP line to understand the variability in predictions for different instances.
- Non-linear relationships: Look for any non-linear patterns in the PDP and ICE lines. This may indicate that the model captures a non-linear relationship between the feature and the model's output.
- Heterogeneity: Check for instances where ICE lines have widely varying slopes, indicating different relationships between the feature and the model's output for individual instances. This may suggest interactions between the feature of interest and other features.
- Outliers: Look for any ICE lines that are very different from the majority of the lines. This may indicate potential outliers or instances that have unique relationships with the feature of interest.
- Confidence intervals: If available, examine the confidence intervals around the PDP line. Wider intervals may indicate a less certain relationship between the feature and the model's output, while narrower intervals suggest a more robust relationship.
- Interactions: By comparing PDPs and ICE plots for different features, you may detect potential interactions between features. If the ICE lines change significantly when comparing two features, this might suggest an interaction effect.

#

Confiedence Interval:

Confidence Interval is a type of estimate computed from the statistics of the observed data which gives a range of values that’s likely to contain a population parameter with a particular level of confidence.
A confidence interval for the mean is a range of values between which the population mean possibly lies.

```python
sns.regplot(data = df, x="weight_kg", y="height_cm", ci=95)
```

<div align="center" >
  <img src="https://www.kaggleusercontent.com/kf/89866716/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..xWMTIOexyELWBUK-bEbf5A.HtRQpSvYoFJdd0BZ7aTL5qAX6UC9G4cdE8Tp_a7iA_mVAzOcjg9Fp2y5lCQvPVUmknXl33WupUFhPmNVqAg_Z4UrWQHRcFB2zz62DrqQIU3NqS8fnxR_Gl9lUztKJMeJ5_vHGb4Qm7P58l5esBF-4h0_xKPpSa56SFSQtTcA69uKiak6K9NmxpvT6MvidBw-th_hnh7uvVsStRBppG_cGwrdlb6QN0yiRQFzY_XrSNBsYuAVuvFCeDbjzARUXScDmTPTDCqP1IMFQYXTABw5WNpNy50loNpTLIylL2QtvK_UliOERpd67103aV-n7HC5RyzDjDiDcXrFTw0UQ_XaSDbPP6NNYWR8m2TAfTGHnKYcHIOZqPgpH8Pufzcvvlf02-pG3WnVlBR8FdiAvlTPSI5sU_HzHnJWnaayWFWo_DOTA1GREst8_uu98AFo7XxvzgbcpxSyTeIj81lmB3mOaw5rCs7-b4kqYZ-31yG27IcNfphOLsYjiuRr0N1uO-8HgL3Joay-o5NwOTX74evlRuOKSqjT-I2ppa6tSE9hWvjgZea0hhkX5AtJAPvy4fAOhBuT5BX8pbS7EXJL0f2gR-ohQX9J8YWqfkAw2SN3tbQpQPuYOhZyuEsiq3yR62exVl-Xjz60byeQj0LIavFv-btdUNwtu8Z-K2VtgRYHvHI.ozrJD-9okuhG8LFBb6ESIA/__results___files/__results___62_1.png" alt="gd" width="400" height="200" />
</div>
