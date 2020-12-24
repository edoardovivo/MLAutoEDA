# AutoEDA

Helper functions to automate Exploratory Data Analysis. The focus is mainly on typical Machine Learning datasets, where there is a target variable (numerical or categorical) and the aim of EDA is to get a sense of the data and whether the variables may be useful for the prediction or not. 

## Quick start

In order to perform automatic EDA on your dataset, you need to import the library, define wich variables are numerical and which are categorical, and which one is the target. The method `summaryEDA`  will compute summaries and charts and output a dictionary of all the data generated. 
```
import AutoEDA.AutoEDA as eda

numerical_vars = ['Age', 'SibSp', 'Fare']
categorical_vars = ['Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked']
target = 'Survived'
summary_dict = eda.summaryEDA(dataframe, numerical_vars, categorical_vars, target, target_type='categorical')
```

The dictionary will contain the summary data for various combination of the variables and the target, as well as the matplotlib figure and axes handles so that they can be rendered and modified afterwards. You can also save the dictionary as a pickle file to avoid computing it multiple times.

## Types of charts

This library produces different types of charts depending on the combination of the types of variables to plot. The library handles univariate, bi-variate (variable vs target) and tri-variate (2 variables vs target) plots.

#### Univariate

##### Numerical

The univariate graphs for numerical variables are a histogram of the values, along with a kde distribution, and a boxplot: 
![Univariate numerical](https://github.com/edoardovivo/AutoEDA/blob/develop/img/univariate_numerical.png)

##### Categorical

The univariate graph for categorical variables is a countplot: 
![Univariate categorical](https://github.com/edoardovivo/AutoEDA/blob/develop/img/univariate_categorical.png)


#### Bi-variate

##### Numerical vs numerical target




##### Numerical vs categorical target
For a numerical variable vs. a categorical target, the library produces a grouped boxplot and a kde plot for each category of the target. The boxplot also features the overall mean and median of the variable, for comparison.

![Numerical vs target(cat)](https://github.com/edoardovivo/AutoEDA/blob/develop/img/numerical_vs_target(cat).png)


##### Categorical vs numerical target


##### Categorical vs categorical target

This produces two plots: a bar plot, grouped by the target, with the percentages of records in each group, and a countplot grouped by the target variable. In the percentage plot, the overall proportions for each level of the target variable are marked with vertical dashed lines, for comparison. For instance, in the following graph it is very easy to see that the survival rate for first class passengers is much higher than the base rate, whereas it is quite lower for 3rd class.
![Categorical vs target(cat)](https://github.com/edoardovivo/AutoEDA/blob/develop/img/categorical_vs_target(cat).png)



#### Tri-variate

##### Two numerical vs numerical target

##### Two numerical vs categorical target

This produces a scatterplot of the two numerical variables, with a hue for each level of the target.
![Categorical vs target(cat)](https://github.com/edoardovivo/AutoEDA/blob/develop/img/two_numerical_vs_target(cat).png)


##### Two categorical vs numerical target

##### Two categorical vs categorical target

This produces two graphs **for each value of the target variable**. On the left, a heatmap with the proportion of records with that target value for each combination of the two categorical variables. On the left, the same heatmap, with the values "shifted" with respect to the overall rate of the target variable. 
For instance, in the first line of the following two graphs, one can see that the percentage of men travelling first class that would not survive is 63%, which represents a 1.5% increase with respect to the overall death rate. **Note of caution: since the percentages are calculated over the target, given one heatmap those do not sum to 1. For the percentages to sum to 1, one needs to look only at the left side graphs, and sum up the corresponding squares of both graphs. For instance, for men in first class the death rate is 0.63 and the survival rate is 0.37, which sums to 1**

![Categorical vs target(cat)](https://github.com/edoardovivo/AutoEDA/blob/develop/img/two_categorical_vs_target(cat).png)


##### One numerical and one categorical vs numerical target

##### One numerical and one categorical vs categorical target

This produces a boxplot like the following one, with the overall mean and median of the numerical variable to provide reference.

![Categorical vs target(cat)](https://github.com/edoardovivo/AutoEDA/blob/develop/img/categorical_numerical_vs_target(cat).png)

















