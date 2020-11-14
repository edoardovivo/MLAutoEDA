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


def prova_prova():
    print("prova")



def categorical_univariate(dataframe, column, palette, ax, order):
    #plt.table(cellText=dcsummary.values,colWidths = [0.25]*len(dc.columns),
    #      rowLabels=dcsummary.index,
    #      colLabels=dcsummary.columns,
    #      cellLoc = 'center', rowLoc = 'center',
    #      loc='top')
    
    # Counts and percentages
    c = dataframe[column].value_counts(dropna=False)
    p = dataframe[column].value_counts(dropna=False, normalize=True)
    df_cnts = pd.concat([c,p], axis=1, keys=['counts', '%'])
    
    fig, axs = plt.subplots(1, 2, figsize=(15,8))
    
    y = column
    axs[0].table(cellText=df_cnts.values,
          rowLabels=df_cnts.index,
          colLabels=df_cnts.columns, loc='center')
    axs[0].axis('off')
    sns.countplot(y=y, data=dataframe,
                  palette=palette, order=order, ax=axs[1])
    plt.show()



def categorical_vs_categorical(dataframe, column, target, palette, ax, order):
    #plt.table(cellText=dcsummary.values,colWidths = [0.25]*len(dc.columns),
    #      rowLabels=dcsummary.index,
    #      colLabels=dcsummary.columns,
    #      cellLoc = 'center', rowLoc = 'center',
    #      loc='top')
    
    # Counts and percentages by column and target
    c = dataframe.groupby([column, target]).size()
    p = c.groupby(level=0).apply(lambda x: x / float(x.sum()))
    df_cnts = pd.concat([c,p], axis=1, keys=['counts', '%'])
    
    # Counts and percentages by target
    c1 = dataframe[target].value_counts(dropna=False, normalize=False)
    p1 = dataframe[target].value_counts(dropna=False, normalize=True)
    df_cnts1 = pd.concat([c1,p1], axis=1, keys=['counts', '% ' + target])
    
    # Joint counts and percentages
    df_cnts_def = df_cnts.join(df_cnts1.rename_axis(target), lsuffix='', rsuffix='_' + target)
    df_cnts_def.update(df_cnts_def[['%', '% ' + target]].applymap('{:,.2f}'.format))


    y = column
    hue = target
    
    fig = plt.figure(constrained_layout=True, figsize=(15,10))

    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Building the table
    table = ax1.table(cellText=df_cnts_def.values,
          rowLabels=df_cnts_def.index,
          colLabels=df_cnts_def.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.1, 1.1)
    ax1.axis('off')
    fig.suptitle("Variables summary: {} vs {}".format(column, target), fontsize=20)
    
    #print(df_cnts.reset_index())
    # Building percentages plot
    barplt = sns.barplot(x='%', y=y, hue=hue ,data=df_cnts.reset_index(),
               palette=palette, order=order, ax=ax2)
    ax2.set_title("Percentages")
    # Making the vertical lines
    leg = barplt.get_legend()
    colors = [x.get_facecolor() for x in leg.legendHandles]
    for color, x in zip(colors, df_cnts1['% ' + target].values):
        ax2.axvline(x, color=color, linestyle="--", linewidth=2.)
    
    
    # Building values plot
    sns.countplot(y=y, hue=hue, data=dataframe,
                  palette=palette, order=order, ax=ax3)
    ax3.set_title("Values")
    plt.tight_layout()
    
    plt.show()


def categorical_vs_numerical(dataframe, column_num, column_cat, palette, ax, order):
    #plt.table(cellText=dcsummary.values,colWidths = [0.25]*len(dc.columns),
    #      rowLabels=dcsummary.index,
    #      colLabels=dcsummary.columns,
    #      cellLoc = 'center', rowLoc = 'center',
    #      loc='top')
    
    # Describe dataframe
    d1 = dataframe[column_num].describe()
    d2 = dataframe.groupby(column_cat)[column_num].describe().transpose()
    d2.columns = ["{}_{}_{}".format(column_num, column_cat,c) for c in d2.columns]
    df_describe_def = pd.concat([d1, d2], axis=1)

    fig = plt.figure(constrained_layout=True, figsize=(15,10))
    fig.suptitle("Variables summary: {} vs {}".format(column_cat, column_num), fontsize=20)

    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Building the table
    table = ax1.table(cellText=df_describe_def.values,
          rowLabels=df_describe_def.index,
          colLabels=df_describe_def.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.1, 1.1)
    ax1.axis('off')
    
    
    #print(df_cnts.reset_index())
    # Building box plot
    bxplt = sns.boxplot(x=column_cat, y=column_num, hue=None ,data=dataframe,
               palette=palette, order=order, ax=ax2)
    ax2.set_title("BoxPlot by category")
    # Making the mean and median lines
    mn = dataframe[column_num].mean()
    med = dataframe[column_num].median()
    ax2.axhline(mn, color="red", label="Ungrouped Mean")
    ax2.axhline(med, color="black", label="Ungrouped Median")
    ax2.legend()
    
    # Building distributions plot
    sns.kdeplot(data=dataframe, x=column_num, hue=column_cat,  palette=palette, ax=ax3)
    #sns.countplot(y=y, hue=hue, data=dataframe,
    #              palette=palette, order=order, ax=ax3)
    ax3.set_title("Distributions by category")
    

    plt.tight_layout()
    
    plt.show()
    

    
def numerical_vs_numerical(dataframe, column_num1, column_num2, palette, ax, order):
    # Numerical vs Numerical target --> Describe for both variable and target; Correlation values; Scatterplot
    d1 = dataframe[[column_num1, column_num2]].describe()
    corr = dataframe[[column_num1, column_num2]].corr()
    
    fig = plt.figure(constrained_layout=True, figsize=(15,10))
    fig.suptitle("Variables summary: {} vs {}".format(column_num1, column_num2), fontsize=20)
    
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    
    
    # Building the table
    table = ax1.table(cellText=d1.values,
          rowLabels=d1.index,
          colLabels=d1.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.1, 1.1)
    ax1.set_title("Summary Table")
    ax1.axis('off')
    
    table2 = ax2.table(cellText=corr.values,
          rowLabels=corr.index,
          colLabels=corr.columns, loc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(14)
    table2.scale(1.1, 1.1)
    ax2.set_title("Pearson correlation")
    ax2.axis('off')
    
    
    sns.scatterplot(data=dataframe, x=column_num1, y=column_num2,  palette=palette, ax=ax3)
    
    
    plt.show()
    
    
    




def two_categorical_vs_categorical(dataframe, column_x, column_y, target, palette, ax, order):
    
    target_values = dataframe[target].unique().tolist()
    n_target_values = len(target_values)
    
    # Counts and percentages by column and target
    c = dataframe.groupby([column_x, column_y, target]).size()
    p = c.groupby(level=[0,1]).apply(lambda x: x / float(x.sum()))
    df_cnts = pd.concat([c,p], axis=1, keys=['counts', '%'])
    
    # Counts and percentages by target
    c1 = dataframe[target].value_counts(dropna=False, normalize=False)
    p1 = dataframe[target].value_counts(dropna=False, normalize=True)
    df_cnts1 = pd.concat([c1,p1], axis=1, keys=['counts', '% ' + target])
    
    # Joint counts and percentages
    df_cnts_def = df_cnts.join(df_cnts1.rename_axis(target), lsuffix='', rsuffix='_' + target)
    df_cnts_def.update(df_cnts_def[['%', '% ' + target]].applymap('{:,.2f}'.format))

    x = column_x
    y = column_y
    hue = target
    
    fig = plt.figure(constrained_layout=True, figsize=(15,10))

    gs = GridSpec(n_target_values + 1, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    axs = []
    for i in range(1, n_target_values+1):
        axs.append([fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1])])


    
    # Building the table
    table = ax1.table(cellText=df_cnts_def.values,
          rowLabels=df_cnts_def.index,
          colLabels=df_cnts_def.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.1, 1.1)
    ax1.axis('off')
    fig.suptitle("Variables summary: {} - {} vs {} ".format(column_x, column_y, target), fontsize=20)
    
    #Building heatmaps
    for i in range(0, n_target_values):
        target_value = target_values[i]
        ax2 = axs[i][0]
        ax3 = axs[i][1]
        df_heatmap_aux = df_cnts_def.reset_index()
        df_heatmap = df_heatmap_aux[df_heatmap_aux[target] == target_value].pivot(column_y, column_x, "%").astype("float")

        val = df_heatmap_aux[df_heatmap_aux[target] == target_value]["% "+ target].astype("float").unique().tolist()[0]
        df_heatmap2 = df_heatmap -val

        sns.heatmap(df_heatmap, annot=True, ax=ax2)
        ax2.set_title("Percentage of {} = {}".format(target, target_value))
        sns.heatmap(df_heatmap2, annot=True, ax=ax3)
        ax3.set_title("Percentage increase of {} = {} wrt baseline".format(target, target_value))
    
    
    return df_cnts_def





    
    

#def categorical_summarized(dataframe, x=None, y=None, hue=None, palette='Set1', ax=None, order=None):
def categorical_summarized(dataframe, variables=None, target=None, palette='Set1', ax=None, order=None):
    '''
    Helper function that gives a quick summary of a given column of categorical data

    Arguments
    =========
    dataframe: pandas dataframe
    x: str. horizontal axis to plot the labels of categorical data, y would be the count
    y: str. vertical axis to plot the labels of categorical data, x would be the count
    hue: str. if you want to compare it another variable (usually the target variable)
    palette: array-like. Colour of the plot

    Returns
    =======
    Quick Stats of the data and also the count plot
    '''
    
    '''
    Possible cases:
    
    - variables is None --> ERROR
    - variables is string and target is None --> Univariate plot
    - variables is list of one and target is None --> Univariate plot
    - variables is list of two and target is None --> Bi-variate plot
    - variables is list of three and target is None --> Tri-variate plot
    - variable is a list of more than 3 and target is None --> ERROR
    - variables is string and target is a string --> Bi-variate plot
    - variables is list of one and target is a string --> Bi-variate plot
    - variables is list of two and target is a string --> Tri-variate plot
    - variables is list of three and target a string --> ERROR
    - variable is a list of more than 3 and target is a string --> ERROR
    '''
    
    
    if variables is None:
        return;
    elif (isinstance(variables, str)) and (target is None):
        # Univariate plot
        categorical_univariate(dataframe, column = variables, palette=palette, ax=ax, order=order)
    elif (type(variables) is list) and (len(variables) == 1) and (target is None):
        # Univariate plot
        categorical_univariate(dataframe, column = variables[0], palette=palette, ax=ax, order=order)
    elif (type(variables) is list) and (len(variables) == 2) and (target is None):
        # Bi-variate plot
        categorical_bivariate(dataframe, column=variables[0], target=variables[1], 
                              palette=palette, ax=ax, order=order)
    elif (type(variables) is list) and (len(variables) == 3) and (target is None):
        # Tri-variate plot
        categorical_trivariate(dataframe, column_x=variables[0], column_y=variables[1], 
                               target=variables[2], 
                              palette=palette, ax=ax, order=order)
    elif (type(variables) is list) and (len(variables) > 3):
        # ERROR
        return;
    elif (isinstance(variables, str)) and (isinstance(target, str)):
        #Bi-variate plot
        categorical_bivariate(dataframe, column=variables, target=target, palette=palette, ax=ax, order=order)
    elif (type(variables) is list) and (len(variables) == 1) and (isinstance(target, str)):
        #Bi-variate plot
        categorical_bivariate(dataframe, column=variables[0], target=target, palette=palette, ax=ax, order=order)
    elif (type(variables) is list) and (len(variables) == 2) and (isinstance(target, str)):
        # Tri-variate plot
        categorical_trivariate(dataframe, column_x=variables[0], column_y=variables[1], 
                               target=target, 
                              palette=palette, ax=ax, order=order)
    elif (type(variables) is list) and (len(variables) >= 3) and (isinstance(target, str)):
        # ERROR
        return;
    else:
        # ERROR
        return;

        
    
    #if x == None:
    #    column_interested = y
    #else:
    #    column_interested = x
    #series = dataframe[column_interested]
    #cnts = series.value_counts()
    #
    ##print(series.describe())
    ##print('mode: ', series.mode())
    ##if verbose:
    ##    print('='*80)
   ##     print(series.value_counts())
#
    #sns.countplot(x=x, y=y, hue=hue, data=dataframe,
    #              palette=palette, order=order, ax=ax)
    #plt.show()




def quantitative_summarized(dataframe, x=None, y=None, hue=None, palette='Set1', ax=None, order=None, verbose=True, swarm=False):
    '''
    Helper function that gives a quick summary of quantattive data

    Arguments
    =========
    dataframe: pandas dataframe
    x: str. horizontal axis to plot the labels of categorical data (usually the target variable)
    y: str. vertical axis to plot the quantitative data
    hue: str. if you want to compare it another categorical variable (usually the target variable if x is another variable)
    palette: array-like. Colour of the plot
    swarm: if swarm is set to True, a swarm plot would be overlayed

    Returns
    =======
    Quick Stats of the data and also the box plot of the distribution
    '''
    series = dataframe[y]
    print(series.describe())
    print('mode: ', series.mode())
    if verbose:
        print('='*80)
        print(series.value_counts())

    sns.boxplot(x=x, y=y, hue=hue, data=dataframe,
                palette=palette, order=order, ax=ax)

    if swarm:
        sns.swarmplot(x=x, y=y, hue=hue, data=dataframe,
                      palette=palette, order=order, ax=ax)

    plt.show()




if __name__ == "__main__":
    pass









