#!/usr/bin/env python

# coding: utf-8

# **BIVARIATE**
# 
# * Categorical vs Categorical Target --> Count + pct for each category and target category; Count + pct for each target category; Grouped count plot with hue=target; Grouped bar plot with percentages with hue=target and lines indicating target percentages without grouping
# 
# * Categorical vs Numerical Target --> Descibe for numerical target alone and for each category; Boxplot for each category with reference line with mean(median) for target alone; overlapping distributions for each value of categorical variable; QQ-plot 
# 
# * Numerical vs Categorical Target --> Describe for numerical variable alone and for each target category; Boxplot for each category with reference line with mean (median) for variable alone; overlapping distributions for each value of target variable; QQ-plot
# 
# * Numerical vs Numerical target --> Describe for both variable and target; Correlation values; Scatterplot
# 
# 
# **TRIVARIATE**
# 
# * 2 Categorical vs Categorical target --> Count + pct for each category and target category; Count + pct for each target category; For each value of target, two heatmaps: one with percentages of the corresponding value for each pair of category values, and one with the difference between the first one, and the pct of the target value without grouping.
# 
# * 2 categorical vs numerical target --> Grouped boxplot with x=categorical, y=target, hue=categorical
# 
# * 1 Categorical, 1 Numerical vs Categorical target --> Grouped boxplot with x=Categorical, y=numerical, hue=target
# 
# * 1 Categorical, 1 Numerical vs Numerical target --> Scatterplot with hue=categorical
# 
# * 2 Numerical vs Categorical target --> Scatterplot with hue=target
# 
# * 2 Numerical vs numerical target --> heatmap
# 
# 
# 
# 
# 

'''
Helper functions to do EDA
'''

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
import scipy.stats as stats
import statsmodels.api as sm
from matplotlib.widgets import Slider
from scipy.stats import pearsonr



class MLAutoEDA():
    
    
    def __init__(self, dataframe, numerical_vars, categorical_vars, target, target_type):
        
        """
        Instantiates an MLAutoEDA object.
        

        :param dataframe: The dataset.
        :type dataframe: pandas DataFrame
        :param numerical_vars: List of names of numerical variables.
        :type numerical_vars: list
        :param categorical_vars: List of names of categorical variables.
        :type categorical_vars: list
        :param target: The name of the target variable.
        :type target: string
        :param target_type: The type of the target variable.
        :type target_type: string
        
        """
        
        
        self.dataframe = dataframe
        self.numerical_vars = numerical_vars
        self.categorical_vars = categorical_vars
        self.target = target
        self.target_type = target_type



    def entropy(self, var_cat):
        
        
        """
        Entropy of a categorical variable H(var_cat)
        
        :param var_cat: The name of the variable.
        :type var_cat: string
    
        :return: The entropy of the variable.
        :rtype: double
        """
        
        
        df = self.dataframe
        if ("Unknown" not in df[var_cat].cat.categories):
            df.loc[:, var_cat] = df[var_cat].cat.add_categories("Unknown").fillna("Unknown")
        #df.loc[df[var_cat].isnull(), var_cat] = "Unknown"
        dist = df[var_cat].value_counts(normalize=True, dropna=False)
        entropy = - np.sum( dist*np.log(dist)  )
        return entropy


    def conditional_entropy(self, var_cat1, var_cat2):
        # Conditional Entropy H(X | Y) = H(cat1 | cat2) = - sum ( p(x,y)*log( p(x,y)/p(y) )  )
        
        """
        Conditional entropy of var_cat1 given var_cat2:  H(var_cat1 | var_cat2) = - sum ( p(x,y)*log( p(x,y)/p(y) )  )
        
        :param var_cat1: The name of the first categorical variable.
        :type var_cat1: string
        :param var_cat2: The name of the second categorical variable.
        :type var_cat2: string
    
        :return: The conditional entropy.
        :rtype: double
        """

        df = self.dataframe
        #df.loc[:, var_cat1] = df[var_cat1].cat.add_categories("Unknown").fillna("Unknown")
        if ("Unknown" not in df[var_cat2].cat.categories):
            df.loc[:, var_cat2] = df[var_cat2].cat.add_categories("Unknown").fillna("Unknown")

        # Computes p(x, y)
        joint_dist = pd.crosstab(index=df[var_cat1], columns=df[var_cat2], normalize='all', dropna=False, ).unstack().reset_index().rename(columns={0: 'P(x,y)'})


        #Computes p(y)
        var_cat2_dist = df[var_cat2].value_counts(normalize=True, dropna=False).rename_axis(var_cat2).reset_index(name='P(y)')

        # Joins dataframes
        df_dist = pd.merge(joint_dist, var_cat2_dist, on=var_cat2)

        # If the dist is zero for some values, I do not have to compute the log
        #cond = df_dist["P(y)"] == 0.
        #df_dist = df_dist[~cond]

        #Computes log(p(x,y) / p(y))
        df_dist["LogProb"] = np.log(df_dist["P(x,y)"] / df_dist["P(y)"])
        df_dist.loc[df_dist["LogProb"] == -np.inf, "LogProb"] = 0.

        #Computes the conditional entropy
        conditional_entropy = -np.sum(df_dist["P(x,y)"]*df_dist["LogProb"] )
        return conditional_entropy


    def eta(self, var_cat, var_num):
        
        """
        Correlation ratio between a categorical variable and a numerical one
        
        :param var_cat: The name of the categorical variable.
        :type var_cat: string
        :param var_num: The name of the numerical variable.
        :type var_num: string
    
        :return: The correlation ratio eta.
        :rtype: double
        """
        
        df = self.dataframe
        cat = df[var_cat]
        num = df[var_num]
        n = cat.value_counts()
        yx = df.groupby(var_cat)[var_num].mean()
        y = df[var_num].mean()
        num = np.sum(n*(yx - y)**2)
        den = np.sum((df[var_num] - y)**2)
        eta = num/den

        return np.sqrt(eta)



    def theil_U(self, var_cat1, var_cat2):
        
        """
        Theil's U coefficient between two categorical variables
        
        :param var_cat1: The name of the first categorical variable.
        :type var_cat1: string
        :param var_cat2: The name of the second categorical variable.
        :type var_cat2: string
    
        :return: The Theil's U coefficient.
        :rtype: double
        """
        
        
        df = self.dataframe
        h1 = self.entropy(var_cat1)
        h12 = self.conditional_entropy(var_cat1, var_cat2)
        U = (h1 - h12)/h1
        return U



    #df_corr = pd.DataFrame(index=categorical2, columns=categorical2)
    #for i in categorical2:
    #  for j in categorical2:
    #    print(i, j)
    #    if (i != j):
    #      df_corr.loc[i, j] = thiel_U(variables, i, j)
    #    else:
    #      df_corr.loc[i, j] = 1.
    #df_corr

    def compute_association(self, var_x, type_var_x, var_y, type_var_y):
        
        """
        Compute's association between two variables:
        * If both variables are numerical, returns the absolute value of the Pearson's correlation coefficient.
        * If one variable is numerical and the other is categorical, then it returns the correlation ratio eta.
        * If both variables are categorical, it returns the Theil's U coefficient.
        
        :param var_x: The name of the first variable.
        :type var_x: string
        :param type_var_x: The type of the first variable (can be 'numerical' or 'categorical')
        :type type_var_x: string
        :param var_y: The name of the second variable.
        :type var_y: string
        :param type_var_y: The type of the second variable (can be 'numerical' or 'categorical')
        :type type_var_y: string
    
        :return: The value of the association depending on the type of the two variables.
        :rtype: double
        """
        
        
        df = self.dataframe
        if (type_var_x == 'numerical') & (type_var_y == 'numerical'):
            # Pearson
            cond = df[var_x].isnull() | df[var_y].isnull()
            df_new = df[~cond]
            pears = pearsonr(df_new[var_x], df_new[var_y])
            assoc = np.abs(pears[0])
        elif ((type_var_x == 'numerical') & (type_var_y == 'categorical')) :
            # eta coefficient
            assoc = self.eta(var_y, var_x)
        elif ((type_var_x == 'categorical') & (type_var_y == 'numerical')):
            assoc = self.eta(var_x, var_y)
        elif (type_var_x == 'categorical') & (type_var_y == 'categorical'):
            #Thiel's U coefficient
            assoc = self.theil_U(var_x, var_y)

        return assoc




    def associations(self, cmap='coolwarm'):
        """
        Computes the association matrix between all variables in the dataset.
                
        :param cmap: The colormap to apply to the graph.
        :type cmap: string
    
        :return: A tuple with: 
            * Correlation matrix
            * Unstacked version of the correlation matrix, ordered from higher to lower value
            * A tuple with figure and axis handles
        :rtype: tuple
        """
        #df_corr = nominal.compute_associations(dataframe, theil_u=True, clustering=True,
        #                nan_strategy='drop_samples',
        #                mark_columns=True)
        #if (target_type == 'numerical'):
        #    numerical_vars = [target] + numerical_vars
        #elif (target_type == 'categorical'):
        #    categorical_vars = [target] + categorical_vars

        numerical_vars_dict = {k: 'numerical' for k in self.numerical_vars}
        categorical_vars_dict = {k: 'categorical' for k in self.categorical_vars}
        all_vars = {**numerical_vars_dict, **categorical_vars_dict}
        all_vars_lst = self.numerical_vars + self.categorical_vars

        target_corr = pd.Series(index=[self.target] + all_vars_lst)
        target_corr.loc[self.target] = 1.
        for j in all_vars_lst:
            target_corr.loc[j] = self.compute_association(self.target, self.target_type, j, all_vars[j])


        idx = ["{} ({})".format(self.target, self.target_type[0:3])] +  ["{} ({})".format(c, all_vars[c][0:3]) for c in all_vars_lst]
        df_corr = pd.DataFrame(index=[self.target] + all_vars_lst, columns=[self.target] + all_vars_lst, dtype="float")
        df_corr.loc[self.target, :] = target_corr
        df_corr.loc[:, self.target] = target_corr

        for i in all_vars_lst:
            for j in all_vars_lst:
                if (i != j):
                    df_corr.loc[i, j] = float(self.compute_association(i, all_vars[i], j, all_vars[j]))
                else:
                    df_corr.loc[i, j] = 1.

        df_corr.index = idx
        df_corr.columns = idx
        df_corr_ordered = df_corr.stack().reset_index()
        df_corr_ordered.columns = ["Var_y", "Var_x", "Corr"]
        df_corr_ordered = df_corr_ordered.sort_values("Corr", ascending=False)

        fig, ax = plt.subplots(figsize=(10, 10))
        g = sns.heatmap(df_corr, annot = True, fmt='.2g',
                       vmin=0, vmax=1, center= 0, cmap=cmap, 
                       linewidths=2, linecolor='black', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        #plt.show()

        return (df_corr, df_corr_ordered, (fig, ax))



    def compute_summary_categorical(self, column):
        
        """
        Computes the summary for a categorical variable:
                
        :param column: The name of the variable.
        :type column: string
    
        :return: A dataframe with normalized and unnormalized value_counts
        :rtype: pandas DataFrame
        """
        
        d = {}
        c = self.dataframe[column].value_counts(dropna=False)
        p = self.dataframe[column].value_counts(dropna=False, normalize=True)
        d[column] = c
        d[column + "_pct"] = p
        return pd.DataFrame(d)


    def compute_summary_numerical_vs_categorical(self, column_num, column_cat):
        
        """
        Computes the summary for a numerical vs categorical variable. The summary is an overall `describe` for the numerical variable, along with a `describe` for each value of the target variable.
                
        :param column_num: The name of the numerical variable.
        :type column_num: string
        
        :param column_cat: The name of the categorical variable.
        :type column_cat: string
    
        :return: A dataframe with the overall `describe` for the numerical variable, along with a `describe` for each value of the target variable. 
        :rtype: pandas DataFrame
        """
        
        d1 = self.dataframe[column_num].describe()
        d2 = self.dataframe.groupby(column_cat)[column_num].describe().transpose()
        d2.columns = ["{}_{}:{}".format(column_num, column_cat,c) for c in d2.columns]
        df_summary = pd.concat([d1, d2], axis=1)

        return df_summary


    def compute_summary_categorical_vs_categorical(self, column_cat1, column_cat2, normalize='both'):
        
        """
        Computes the summary for a categorical vs categorical variable. If `normalize=both`, the summary consists of two crosstabs, one with a count and another one with percentages computed over the target. Otherwise, only one of the two crosstabs is returned.
                
        :param column_cat1: The name of the first categorical variable.
        :type column_cat1: string
        
        :param column_cat2: The name of the second categorical variable.
        :type column_cat2: string
        
        :param normalize: Can be 'yes', 'no' or 'both'. 
        :type normalize: string
        
    
        :return: Crosstab dataframes depending on the value of normalize
        :rtype: pandas DataFrame or tuple of pandas DataFrames
        """
        
        if (normalize == 'yes'):
            d2 = pd.crosstab(index=self.dataframe[column_cat1], columns=self.dataframe[column_cat2], margins=True, normalize='index')
            return d2
        elif (normalize == 'no'):
            d1 = pd.crosstab(index=self.dataframe[column_cat1], columns=self.dataframe[column_cat2], margins=True)
            return d1
        elif (normalize == 'both'):
            d1 = pd.crosstab(index=self.dataframe[column_cat1], columns=self.dataframe[column_cat2], margins=True)
            d2 = pd.crosstab(index=self.dataframe[column_cat1], columns=self.dataframe[column_cat2], margins=True, normalize='index')
            return d1, d2
        else:
            return 'Error'


    def compute_summary_numerical_vs_numerical(self, column_num1, column_num2):
        
        """
        Computes the summary for a numerical vs numerical variable. The summary is a combination of the single-variable summaries given by the `describe` function in pandas.
                
        :param column_num1: The name of the first numerical variable.
        :type column_num1: string
        
        :param column_num2: The name of the second numerical variable.
        :type column_num2: string
        
    
        :return: describe dataframe
        :rtype: pandas DataFrame
        """
        
        return self.dataframe[[column_num1, column_num2]].describe()


    def compute_summary_2categorical_vs_categorical(self, column_cat1, column_cat2, column_cat3, normalize='both'):
        
        """
        Computes the summary for two categorical vs another categorical variable. If `normalize=both`, the summaries are two hierarchical crosstabs, where the target is in the columns. The first one is for the counts and the second one is for percentages. Otherwise, only one of the crosstabs is returned.
                
        :param column_cat1: The name of the first categorical variable.
        :type column_cat1: string
        
        :param column_cat2: The name of the second categorical variable.
        :type column_cat2: string
        
        :param column_cat3: The name of the third categorical variable.
        :type column_cat3: string
        
        :param normalize: Can be 'yes', 'no' or 'both'. 
        :type normalize: string
        
    
        :return: Crosstab dataframes depending on the value of normalize
        :rtype: pandas DataFrame or tuple of pandas DataFrames
        """
        
        
        if (normalize == 'yes'):
            d2 = pd.crosstab(index=[self.dataframe[column_cat1], self.dataframe[column_cat2]], columns=self.dataframe[column_cat3], margins=True, normalize='index')
            return d2
        elif (normalize == 'no'):
            d1 = pd.crosstab(index=[self.dataframe[column_cat1], self.dataframe[column_cat2]], columns=self.dataframe[column_cat3], margins=True)
            return d1
        elif (normalize == 'both'):
            d1 = pd.crosstab(index=[self.dataframe[column_cat1], self.dataframe[column_cat2]], columns=self.dataframe[column_cat3], margins=True)
            d2 = pd.crosstab(index=[self.dataframe[column_cat1], self.dataframe[column_cat2]], columns=self.dataframe[column_cat3], margins=True, normalize='index')
            return d1, d2
        else:
            return 'Error'


    def compute_summary_2categorical_vs_numerical(self, column_cat1, column_cat2, column_num):    
        
        """
        Computes the summary for two categorical vs a numerical variable. The summary is a `describe` statement for each combination of the two categorical variables.
                
        :param column_cat1: The name of the first categorical variable.
        :type column_cat1: string
        
        :param column_cat2: The name of the second categorical variable.
        :type column_cat2: string
        
        :param column_num: The name of the numerical variable.
        :type column_num: string
      
    
        :return: Dataframe with a `describe` statement for each combination of the two categorical variables.
        :rtype: pandas DataFrame
        """
        
        
        d1 = self.dataframe[column_num].describe().to_frame(column_num)
        d2 = self.dataframe.groupby([column_cat1, column_cat2])[column_num].describe()
        d2.index = ["{}_{}:{}_{}:{}".format(column_num, column_cat1, x[0], column_cat2, x[1]) for x in d2.index.to_flat_index()]
        return pd.concat([d1, d2.transpose()], axis=1)


    def compute_summary_2numerical_vs_categorical(self, column_num1, column_num2, column_cat):
        
        
        """
        Computes the summary for two numerical vs a categorical variable. The summary is a `describe` statement for each of the numerical variables, along with `describe` statements for each value of the target. 
                
        :param column_num1: The name of the first numerical variable.
        :type column_num1: string
        
        :param column_num2: The name of the second numerical variable.
        :type column_num2: string
        
        :param column_cat: The name of the categorical variable.
        :type column_cat: string
        
    
        :return: Dataframe with a `describe` statement for each of the numerical variables, along with `describe` statements for each value of the categorical variable. 
        :rtype: pandas DataFrame
        """
        
        
        d1 = self.dataframe[[column_num1, column_num2]].describe()
        d2 = self.dataframe.groupby([column_cat])[column_num1].describe().transpose()
        d2.columns = ["{}_{}:{}".format(column_num1, column_cat, x) for x in d2.columns ]
        d3 = self.dataframe.groupby([column_cat])[column_num2].describe().transpose()
        d3.columns = ["{}_{}:{}".format(column_num2, column_cat, x) for x in d3.columns ]
        #d2.columns = d2.columns.droplevel(0)
        return pd.concat([d1, d2, d3], axis=1)


    def compute_summary_2numerical_vs_numerical(self, column_num1, column_num2, column_num3):
        
        """
        Computes the summary for two numerical vs a numerical variable. The summary is a `describe` statement for each of the variables.
                
        :param column_num1: The name of the first numerical variable.
        :type column_num1: string
        
        :param column_num2: The name of the second numerical variable.
        :type column_num2: string
        
        :param column_num3: The name of the third numerical variable.
        :type column_num3: string
        
    
        :return: Dataframe with `describe` statement for each of the variables.
        :rtype: pandas DataFrame
        """
        
        
        return self.dataframe[[column_num1, column_num2, column_num3]].describe()



    def categorical_univariate(self, column, palette, show_plot=True):
        
        """
        Computes the summary and the corresponding plot for a categorical variable
                
        :param column: The name of the categorical variable.
        :type column: string
        
        :param palette: The color palette for the plot.
        :type palette: string
        
        :param show_plot: Whether or not to show the plot.
        :type show_plot: bool
        
    
        :return: tuple with figure and axis handles, and the summary dataframe
        :rtype: tuple
        """

        df_summary = self.compute_summary_categorical(column)

        fig, axs = plt.subplots(figsize=(15,8))

        y = column
        #axs[0].table(cellText=df_cnts.values,
        #      rowLabels=df_cnts.index,
        #      colLabels=df_cnts.columns, loc='center')
        #axs[0].axis('off')
        cntplot = sns.countplot(y=y, data=self.dataframe,
                      palette=palette);

        if (show_plot):
            plt.show();

        return (fig, axs), df_summary


    def numerical_univariate(self, column, palette, show_plot=True):
        
        """
        Computes the summary and the corresponding plot for a numerical variable
                
        :param column: The name of the numerical variable.
        :type column: string
        
        :param palette: The color palette for the plot.
        :type palette: string
        
        :param show_plot: Whether or not to show the plot.
        :type show_plot: bool
        
    
        :return: tuple with figure and axis handles, and the summary dataframe
        :rtype: tuple
        """
        
        
        df_summary = self.dataframe[column].describe()

        fig, axs = plt.subplots(1, 2,figsize=(10,8))

        y = column
        #axs[0].table(cellText=df_cnts.values,
        #      rowLabels=df_cnts.index,
        #      colLabels=df_cnts.columns, loc='center')
        #axs[0].axis('off')
        #sns.countplot(y=y, data=dataframe,
        #              palette=palette, order=order, ax=axs)

        hst = sns.histplot(data=self.dataframe, x=column, palette=palette, ax=axs[0],  kde=True);
        boxplot = sns.boxplot(data=self.dataframe, y=column, palette=palette, ax=axs[1]);

        if (show_plot):
            plt.show();

        return (fig, axs), df_summary




    def categorical_vs_categorical(self, column_cat1, column_cat2, palette, show_plot=True):
        
        """
        Computes the summary and the corresponding plot for categorical vs categorical variables. This produces two plots: a bar plot, grouped by the target, with the percentages of records in each group, and a countplot grouped by the target variable. In the percentage plot, the overall proportions for each level of the target variable are marked with vertical dashed lines, for comparison
                
        :param column_cat1: The name of the first categorical variable.
        :type column_cat1: string
        
        :param column_cat2: The name of the second categorical variable.
        :type column_cat2: string
        
        :param palette: The color palette for the plot.
        :type palette: string
        
        :param show_plot: Whether or not to show the plot.
        :type show_plot: bool
        
    
        :return: tuple with figure and axis handles, and the summary dataframe
        :rtype: tuple
        """
        
        
        
        #plt.table(cellText=dcsummary.values,colWidths = [0.25]*len(dc.columns),
        #      rowLabels=dcsummary.index,
        #      colLabels=dcsummary.columns,
        #      cellLoc = 'center', rowLoc = 'center',
        #      loc='top')

        df_summary = self.compute_summary_categorical_vs_categorical(column_cat1, column_cat2, normalize='both')
        df_summary_new = df_summary[1].unstack().drop('All', level=column_cat1).reset_index().rename(columns={0: '%'})
        df_summary_new[[column_cat1, column_cat2]] = df_summary_new[[column_cat1, column_cat2]].astype("category")

        # Counts and percentages by column and target
        #c = dataframe.groupby([column_cat1, column_cat2]).size()
        #p = c.groupby(level=0).apply(lambda x: x / float(x.sum()))
        #df_cnts = pd.concat([c,p], axis=1, keys=['counts', '%'])

        # Counts and percentages by target
        #c1 = dataframe[column_cat2].value_counts(dropna=False, normalize=False)
        #p1 = dataframe[column_cat2].value_counts(dropna=False, normalize=True)
        #df_cnts1 = pd.concat([c1,p1], axis=1, keys=['counts', '% ' + column_cat2])

        # Joint counts and percentages
        #df_cnts_def = df_cnts.join(df_cnts1.rename_axis(column_cat2), lsuffix='', rsuffix='_' + column_cat2)
        #df_cnts_def.update(df_cnts_def[['%', '% ' + column_cat2]].applymap('{:,.2f}'.format))


        y = column_cat1
        hue = column_cat2

        fig, ax = plt.subplots(1, 2, figsize=(15,10))


        # Building the table
        #table = ax1.table(cellText=df_cnts_def.values,
        #      rowLabels=df_cnts_def.index,
        #      colLabels=df_cnts_def.columns, loc='center')
        #table.auto_set_font_size(False)
        #table.set_fontsize(14)
        #table.scale(1.1, 1.1)
        #ax1.axis('off')
        #fig.suptitle("Variables summary: {} vs {}".format(column_cat1, column_cat2), fontsize=20)

        #print(df_cnts.reset_index())
        # Building percentages plot
        barplt = sns.barplot(x='%', y=y, hue=hue ,data=df_summary_new,
                   palette=palette, ax=ax[0])
        ax[0].set_title("Percentages")
        # Making the vertical lines
        leg = barplt.get_legend()
        colors = [x.get_facecolor() for x in leg.legendHandles]
        for color, x in zip(colors, df_summary[1].loc['All']):
            ax[0].axvline(x, color=color, linestyle="--", linewidth=2.)


        # Building values plot
        sns.countplot(y=y, hue=hue, data=self.dataframe,
                      palette=palette, ax=ax[1])
        ax[1].set_title("Values")
        plt.tight_layout()
        
        if (show_plot):
            plt.show()

        return (fig, ax), df_summary


    def categorical_vs_numerical(self, column_num, column_cat, palette, show_plot=True):
        
        """
        Computes the summary and the corresponding plot for categorical vs numerical variables. TThis produces two plots: a boxplot for the distribution of the target for each value of the categorical variable, and a kde plot showing the distributions. The boxplot also features the overall mean and median of the target, for comparison.
                
        :param column_num: The name of the numerical variable.
        :type column_num: string
        
        :param column_cat: The name of the categorical variable.
        :type column_cat: string
        
        :param palette: The color palette for the plot.
        :type palette: string
        
        :param show_plot: Whether or not to show the plot.
        :type show_plot: bool
        
    
        :return: tuple with figure and axis handles, and the summary dataframe
        :rtype: tuple
        """
        
        #plt.table(cellText=dcsummary.values,colWidths = [0.25]*len(dc.columns),
        #      rowLabels=dcsummary.index,
        #      colLabels=dcsummary.columns,
        #      cellLoc = 'center', rowLoc = 'center',
        #      loc='top')

        df_summary = self.compute_summary_numerical_vs_categorical(column_num, column_cat)

        # Describe dataframe
        d1 = self.dataframe[column_num].describe()
        d2 = self.dataframe.groupby(column_cat)[column_num].describe().transpose()
        d2.columns = ["{}_{}_{}".format(column_num, column_cat,c) for c in d2.columns]
        df_describe_def = pd.concat([d1, d2], axis=1)

        fig, ax = plt.subplots(1, 2, figsize=(15,10))


        #print(df_cnts.reset_index())
        # Building box plot
        bxplt = sns.boxplot(x=column_cat, y=column_num, hue=None ,data=self.dataframe,
                   palette=palette,ax=ax[0])
        ax[0].set_title("BoxPlot by category")
        # Making the mean and median lines
        mn = self.dataframe[column_num].mean()
        med = self.dataframe[column_num].median()
        ax[0].axhline(mn, color="red", label="Ungrouped Mean")
        ax[0].axhline(med, color="black", label="Ungrouped Median")
        ax[0].legend()

        # Building distributions plot
        sns.kdeplot(data=self.dataframe, x=column_num, hue=column_cat,  palette=palette, ax=ax[1])
        #sns.countplot(y=y, hue=hue, data=dataframe,
        #              palette=palette, order=order, ax=ax3)
        ax[1].set_title("Distributions by category")


        plt.tight_layout()
        
        if (show_plot):
            plt.show()

        return (fig, ax), df_summary




    def numerical_vs_numerical(self, column_num1, column_num2, palette, show_plot=True):
        
        
        
        """
        Computes the summary and the corresponding plot for numerical vs numerical variables. In this case, a scatterplot is produced.
                
        :param column_num1: The name of the first numerical variable.
        :type column_num1: string
        
        :param column_num2: The name of the second numerical variable.
        :type column_num2: string
        
        :param palette: The color palette for the plot.
        :type palette: string
        
        :param show_plot: Whether or not to show the plot.
        :type show_plot: bool
        
    
        :return: tuple with figure and axis handles, and the summary dataframe
        :rtype: tuple
        """
        
        
        # Numerical vs Numerical target --> Describe for both variable and target; Scatterplot
        df_summary = self.dataframe[[column_num1, column_num2]].describe()
        #corr = dataframe[[column_num1, column_num2]].corr()

        fig, ax = plt.subplots(1, 1, figsize=(15,10))

        sns.scatterplot(data=self.dataframe, x=column_num1, y=column_num2,  palette=palette, ax=ax)
        
        if (show_plot):
            plt.show()

        return (fig, ax), df_summary



    def two_categorical_vs_categorical(self, column_cat1, column_cat2, column_cat3, palette, show_plot=True):
        
        
        """
        Computes the summary and the corresponding plot for two categorical vs categorical variables. This produces two graphs for each value of the target variable. On the left, a heatmap with the proportion of records with that target value for each combination of the two categorical variables. On the left, the same heatmap, with the values "shifted" with respect to the overall rate of the target variable. 
                
        :param column_cat1: The name of the first categorical variable.
        :type column_cat1: string
        
        :param column_cat2: The name of the second categorical variable.
        :type column_cat2: string
        
        :param column_cat3: The name of the third categorical variable.
        :type column_cat3: string
        
        :param palette: The color palette for the plot.
        :type palette: string
        
        :param show_plot: Whether or not to show the plot.
        :type show_plot: bool
        
    
        :return: tuple with figure and axis handles, and the summary dataframe
        :rtype: tuple
        """
        
        
        
        '''
        2 Categorical vs Categorical target --> Count + pct for each category and target category; Count + pct for each target category; For each value of target (column_cat3), two heatmaps: one with percentages of the corresponding value for each pair of category values, and one with the difference between the first one, and the pct of the target value without grouping (baseline).
        '''
        
        df_summary = self.compute_summary_2categorical_vs_categorical(column_cat1, column_cat2, column_cat3)

        target_values = self.dataframe[column_cat3].unique().tolist()
        n_target_values = len(target_values)

        x = column_cat1
        y = column_cat2
        hue = column_cat3

        fig, ax = plt.subplots(n_target_values,2, figsize=(15,10))



        #Building heatmaps
        for i in range(0, n_target_values):
            target_value = target_values[i]
            ax2 = ax[i, 0]
            ax3 = ax[i, 1]

            df_heatmap = df_summary[1][target_value].drop("All", level=0).reset_index().pivot(column_cat1, column_cat2, target_value).astype("float")

            val = df_summary[1].loc["All", target_value].values[0]
            df_heatmap2 = df_heatmap -val


            sns.heatmap(df_heatmap, annot=True, ax=ax2)
            ax2.set_title("Percentage of {} = {}".format(column_cat3, target_value))
            sns.heatmap(df_heatmap2, annot=True, ax=ax3)
            ax3.set_title("Percentage increase of {} = {} wrt baseline".format(column_cat3, target_value))

        if (show_plot):
            plt.show()

        return (fig, ax), df_summary



    def two_categorical_vs_numerical(self, column_cat1, column_cat2, column_num, palette, show_plot=True):
        
        """
        Computes the summary and the corresponding plot for two categorical vs numerical variables. This produces a grouped boxplot for the distribution of the numerical variable. 
                
        :param column_cat1: The name of the first categorical variable.
        :type column_cat1: string
        
        :param column_cat2: The name of the second categorical variable.
        :type column_cat2: string
        
        :param column_num: The name of the numerical variable.
        :type column_num: string
        
        :param palette: The color palette for the plot.
        :type palette: string
        
        :param show_plot: Whether or not to show the plot.
        :type show_plot: bool
        
    
        :return: tuple with figure and axis handles, and the summary dataframe
        :rtype: tuple
        """
        
        
        
        '''
        2 categorical vs numerical target --> Grouped boxplot with x=categorical, y=target, hue=categorical
        '''

        df_summary = self.compute_summary_2categorical_vs_numerical(column_cat1, column_cat2, column_num)


        fig, ax = plt.subplots(1,1, figsize=(15,10))

        x=column_cat1
        y=column_num
        hue=column_cat2

        # Building box plot
        bxplt = sns.boxplot(x=x, y=y, hue=hue ,data=self.dataframe,
                   palette=palette, ax=ax)
        ax.set_title("BoxPlot by category")
        # Making the mean and median lines
        mn = self.dataframe[column_num].mean()
        med = self.dataframe[column_num].median()
        ax.axhline(mn, color="red", label="Ungrouped Mean")
        ax.axhline(med, color="black", label="Ungrouped Median")
        ax.legend()
        
        if (show_plot):
            plt.show()

        return (fig, ax), df_summary


    def two_numerical_vs_categorical(self, column_num1, column_num2, column_cat, palette, show_plot=True):
        
        
        """
        Computes the summary and the corresponding plot for two numerical vs categorical variables. This produces a scatterplot of the two numerical variables, with a hue for each level of the target.
                
        :param column_num1: The name of the first numerical variable.
        :type column_num1: string
        
        :param column_num2: The name of the second numerical variable.
        :type column_num2: string
        
        :param column_cat The name of the categorical variable.
        :type column_cat: string
        
        :param palette: The color palette for the plot.
        :type palette: string
        
        :param show_plot: Whether or not to show the plot.
        :type show_plot: bool
        
    
        :return: tuple with figure and axis handles, and the summary dataframe
        :rtype: tuple
        """
        
        
        
        '''
        2 Numerical vs Categorical target --> Scatterplot with hue=target
        '''

        df_summary = self.compute_summary_2numerical_vs_categorical(column_num1, column_num2, column_cat)

        fig, ax = plt.subplots(1, 1, figsize=(15,10))

        sns.scatterplot(data=self.dataframe, x=column_num1, y=column_num2,  hue=column_cat, palette=palette, ax=ax)

        if (show_plot):
            plt.show()

        return (fig, ax), df_summary


    def two_numerical_vs_numerical(self, column_num1, column_num2, column_num3, palette, show_plot=True):
        
        """
        Computes the summary and the corresponding plot for two numerical vs categorical variables. This produces a hexbin plot where the value of the target variable is represented as a color gradient.
                
        :param column_num1: The name of the first numerical variable.
        :type column_num1: string
        
        :param column_num2: The name of the second numerical variable.
        :type column_num2: string
        
        :param column_num3 The name of the third numerical variable.
        :type column_num3: string
        
        :param palette: The color palette for the plot.
        :type palette: string
        
        :param show_plot: Whether or not to show the plot.
        :type show_plot: bool
        
    
        :return: tuple with figure and axis handles, and the summary dataframe
        :rtype: tuple
        """
        
        
        '''
        2 Numerical vs numerical target --> hexbin
        '''

        df_summary = self.compute_summary_2numerical_vs_numerical(column_num1, column_num2, column_num3)


        fig, ax = plt.subplots(1, 1, figsize=(15,10))
        fig.suptitle("Variables summary: {} and {} vs {}".format(column_num1, column_num2, column_num3), fontsize=20)


        hb = ax.hexbin(self.dataframe[column_num1], self.dataframe[column_num2], C=self.dataframe[column_num3], gridsize=20, cmap=palette)
        ax.set_xlabel(column_num1)
        ax.set_ylabel(column_num2)
        ax.set_title("HexBin Plot")
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label(column_num3)
        
        
        if (show_plot):
            plt.show()

        return (fig, ax), df_summary



    def summaryEDA(self, show_plot=False, palette=None):
        
        
        """
        Computes univariate, bivariate and trivariate summaries and charts and stores them in a dictionary.
        
         
        :param show_plot: Whether or not to show the plots.
        :type show_plot: bool
        
        :param palette: The color palette to use in the plots.
        :type palette: string
        
    
        :return: Dictionary with all the summaries, figure and axes handles
        :rtype: dictionary
        
        
        """
        
        
        
        summary_dict = {}

        #Associations
        summary_dict['associations'] = {}
        df_corr, df_corr_ordered, (fig,ax) =  self.associations()
        summary_dict['associations']['df_corr'] = df_corr
        summary_dict['associations']['df_corr_ordered'] = df_corr_ordered
        summary_dict['associations']['figure'] = fig
        summary_dict['associations']['axes'] = ax



        # Univariate plots and summaries
        for num_var in self.numerical_vars:
            (fig, axs), df_summary = self.numerical_univariate(num_var, palette, show_plot)
            summary_dict[num_var] = {}
            summary_dict[num_var]['figure'] = fig
            summary_dict[num_var]['axes'] = axs
            summary_dict[num_var]['summary'] = df_summary


        for cat_var in self.categorical_vars:
            (fig, axs), df_summary = self.categorical_univariate(cat_var, palette, show_plot)
            summary_dict[cat_var] = {}
            summary_dict[cat_var]['figure'] = fig
            summary_dict[cat_var]['axes'] = axs
            summary_dict[cat_var]['summary'] = df_summary

        # Numerical vs target
        for num_var in self.numerical_vars:
            if (self.target_type == 'numerical'):
                (fig, axs), df_summary = self.numerical_vs_numerical(num_var, self.target, palette, show_plot)
            elif (self.target_type == 'categorical'):
                (fig, axs), df_summary = self.categorical_vs_numerical( num_var, self.target, palette, show_plot)
            k = (num_var, self.target)
            summary_dict[k] = {}
            summary_dict[k]['figure'] = fig
            summary_dict[k]['axes'] = axs
            summary_dict[k]['summary'] = df_summary

        # Categorical vs target
        for cat_var in self.categorical_vars:
            if (self.target_type == 'numerical'):
                (fig, axs), df_summary = self.categorical_vs_numerical(self.target, cat_var, palette, show_plot)
            elif (self.target_type == 'categorical'):
                (fig, axs), df_summary = self.categorical_vs_categorical(cat_var, self.target, palette, show_plot)
            k = (cat_var, self.target)
            summary_dict[k] = {}
            summary_dict[k]['figure'] = fig
            summary_dict[k]['axes'] = axs
            summary_dict[k]['summary'] = df_summary

        # 2 Numerical vs target
        for i, num_var_i in enumerate(self.numerical_vars):
            for j, num_var_j in enumerate(self.numerical_vars):
                if (i < j):
                    if (self.target_type == 'numerical'):
                        (fig, axs), df_summary = self.two_numerical_vs_numerical(num_var_i, num_var_j, self.target, palette, show_plot)
                    elif (self.target_type == 'categorical'):
                        (fig, axs), df_summary = self.two_numerical_vs_categorical( num_var_i, num_var_j, self.target, palette, show_plot)
                    k = (num_var_i, num_var_j, self.target)
                    summary_dict[k] = {}
                    summary_dict[k]['figure'] = fig
                    summary_dict[k]['axes'] = axs
                    summary_dict[k]['summary'] = df_summary


        # 2 Categorical vs target
        for i, cat_var_i in enumerate(self.categorical_vars):
            for j, cat_var_j in enumerate(self.categorical_vars):
                if (i < j):
                    if (self.target_type == 'numerical'):
                        (fig, axs), df_summary = self.two_categorical_vs_numerical(cat_var_i, cat_var_j, self.target, palette, show_plot)
                    elif (self.target_type == 'categorical'):
                        (fig, axs), df_summary = self.two_categorical_vs_categorical(cat_var_i, cat_var_j, self.target, palette, show_plot)
                    k = (cat_var_i, cat_var_j, self.target)
                    summary_dict[k] = {}
                    summary_dict[k]['figure'] = fig
                    summary_dict[k]['axes'] = axs
                    summary_dict[k]['summary'] = df_summary

        # 1 Numerical and 1 categorical vs target
        for i, num_var_i in enumerate(self.numerical_vars):
            for j, cat_var_j in enumerate(self.categorical_vars):
                if (self.target_type == 'numerical'):
                    (fig, axs), df_summary = self.two_numerical_vs_categorical(num_var_i, self.target, cat_var_j, palette, show_plot)
                elif (self.target_type == 'categorical'):
                    (fig, axs), df_summary = self.two_categorical_vs_numerical(cat_var_j, self.target, num_var_i, palette, show_plot)
                k = (num_var_i, cat_var_j, self.target)
                summary_dict[k] = {}
                summary_dict[k]['figure'] = fig
                summary_dict[k]['axes'] = axs
                summary_dict[k]['summary'] = df_summary



        return summary_dict
    
   


if __name__ == "__main__":
    pass










