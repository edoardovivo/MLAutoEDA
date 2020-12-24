# AutoEDA

Helper functions to automate Exploratory Data Analysis. The focus is mainly on typical Machine Learning datasets, where there is a target variable (numerical or categorical) and the aim of EDA is to get a sense of whether the variables may improve the prediction or not. 

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

[Univariate numerical](https://github.com/edoardovivo/AutoEDA/blob/develop/img/univariate_numerical.png)
