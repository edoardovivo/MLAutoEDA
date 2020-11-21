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
from dython import nominal


def associations(dataframe, cmap='coolwarm'):
    '''
    Uses dython to compute associations
    - Numerical vs numerical: pearson's correlation coefficient
    - Categorical vs Categorical: Thiels U coefficient
    - Categorical vs Numerical: Correlation ratio (eta)
    '''
    df_corr = nominal.compute_associations(dataframe, theil_u=True, clustering=True,
                    nan_strategy='drop_samples',
                    mark_columns=True)
    df_corr_ordered = df_corr.stack().reset_index()
    df_corr_ordered.columns = ["Var_y", "Var_x", "Corr"]
    df_corr_ordered = df_corr_ordered.sort_values("Corr", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 10))
    g = sns.heatmap(df_corr, annot = True, fmt='.2g',
                   vmin=-1, vmax=1, center= 0, cmap=cmap, 
                   linewidths=2, linecolor='black', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    
    plt.show()
    
    return (df_corr, df_corr_ordered, (fig, ax)) 


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



def categorical_vs_categorical(dataframe, column_cat1, column_cat2, palette, ax, order):
    #plt.table(cellText=dcsummary.values,colWidths = [0.25]*len(dc.columns),
    #      rowLabels=dcsummary.index,
    #      colLabels=dcsummary.columns,
    #      cellLoc = 'center', rowLoc = 'center',
    #      loc='top')
    
    # Counts and percentages by column and target
    c = dataframe.groupby([column_cat1, column_cat2]).size()
    p = c.groupby(level=0).apply(lambda x: x / float(x.sum()))
    df_cnts = pd.concat([c,p], axis=1, keys=['counts', '%'])
    
    # Counts and percentages by target
    c1 = dataframe[column_cat2].value_counts(dropna=False, normalize=False)
    p1 = dataframe[column_cat2].value_counts(dropna=False, normalize=True)
    df_cnts1 = pd.concat([c1,p1], axis=1, keys=['counts', '% ' + column_cat2])
    
    # Joint counts and percentages
    df_cnts_def = df_cnts.join(df_cnts1.rename_axis(column_cat2), lsuffix='', rsuffix='_' + column_cat2)
    df_cnts_def.update(df_cnts_def[['%', '% ' + column_cat2]].applymap('{:,.2f}'.format))


    y = column_cat1
    hue = column_cat2
    
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
    fig.suptitle("Variables summary: {} vs {}".format(column_cat1, column_cat2), fontsize=20)
    
    #print(df_cnts.reset_index())
    # Building percentages plot
    barplt = sns.barplot(x='%', y=y, hue=hue ,data=df_cnts.reset_index(),
               palette=palette, order=order, ax=ax2)
    ax2.set_title("Percentages")
    # Making the vertical lines
    leg = barplt.get_legend()
    colors = [x.get_facecolor() for x in leg.legendHandles]
    for color, x in zip(colors, df_cnts1['% ' + column_cat2].values):
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
    
    

def two_categorical_vs_categorical(dataframe, column_cat1, column_cat2, column_cat3, palette, ax, order):
    '''
    2 Categorical vs Categorical target --> Count + pct for each category and target category; Count + pct for each target category; For each value of target (column_cat3), two heatmaps: one with percentages of the corresponding value for each pair of category values, and one with the difference between the first one, and the pct of the target value without grouping (baseline).
    '''
     
    
    target_values = dataframe[column_cat3].unique().tolist()
    n_target_values = len(target_values)
    
    # Counts and percentages by column and target
    c = dataframe.groupby([column_cat1, column_cat2, column_cat3]).size()
    p = c.groupby(level=[0,1]).apply(lambda x: x / float(x.sum()))
    df_cnts = pd.concat([c,p], axis=1, keys=['counts', '%'])
    
    # Counts and percentages by target
    c1 = dataframe[column_cat3].value_counts(dropna=False, normalize=False)
    p1 = dataframe[column_cat3].value_counts(dropna=False, normalize=True)
    df_cnts1 = pd.concat([c1,p1], axis=1, keys=['counts', '% ' + column_cat3])
    
    # Joint counts and percentages
    df_cnts_def = df_cnts.join(df_cnts1.rename_axis(column_cat3), lsuffix='', rsuffix='_' + column_cat3)
    df_cnts_def.update(df_cnts_def[['%', '% ' + column_cat3]].applymap('{:,.2f}'.format))

    x = column_cat1
    y = column_cat2
    hue = column_cat3
    
    fig = plt.figure(constrained_layout=True, figsize=(15,10))
    fig.suptitle("Variables summary: {} - {} vs {} ".format(column_cat1, column_cat2, column_cat3), fontsize=20)

    gs = GridSpec(2, 2, figure=fig)
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
    
    
    #Building heatmaps
    for i in range(0, n_target_values):
        target_value = target_values[i]
        ax2 = axs[i][0]
        ax3 = axs[i][1]
        df_heatmap_aux = df_cnts_def.reset_index()
        df_heatmap = df_heatmap_aux[df_heatmap_aux[column_cat3] == target_value].pivot(column_cat1, column_cat2, "%").astype("float")

        val = df_heatmap_aux[df_heatmap_aux[column_cat3] == target_value]["% "+ column_cat3].astype("float").unique().tolist()[0]
        df_heatmap2 = df_heatmap -val

        sns.heatmap(df_heatmap, annot=True, ax=ax2)
        ax2.set_title("Percentage of {} = {}".format(column_cat3, target_value))
        sns.heatmap(df_heatmap2, annot=True, ax=ax3)
        ax3.set_title("Percentage increase of {} = {} wrt baseline".format(column_cat3, target_value))
    
    
    plt.show()



def two_categorical_vs_numerical(dataframe, column_cat1, column_cat2, column_num, palette, ax, order):
    '''
    2 categorical vs numerical target --> Grouped boxplot with x=categorical, y=target, hue=categorical
    '''
    d1 = dataframe[column_num].describe()
    d2 = dataframe.groupby(column_cat1)[column_num].describe().transpose()
    d3 = dataframe.groupby(column_cat2)[column_num].describe().transpose()
    d4 = dataframe.groupby([column_cat1, column_cat2])[column_num].describe().transpose()
    d2.columns = ["{}_{}_{}".format(column_num, column_cat1,c) for c in d2.columns]
    d3.columns = ["{}_{}_{}".format(column_num, column_cat2,c) for c in d3.columns]
    #["{}_{}{}_{}".format(column_num, levelvalues_name,c, ) for c in levelvalues_categories]
    names = d4.columns.names
    cats = list(d4.columns)
    combined = [list(zip(names,a)) for a in cats]
    combined2 = ["{}_{}:{}_{}:{}".format(column_num,x[0][0], x[0][1], x[1][0], x[1][1]) for x in combined]
    d4.columns = combined2
    df_describe_def = pd.concat([d1, d2, d3, d4], axis=1)
    df_describe_def = df_describe_def.applymap('{:,.2f}'.format)
    
    
    fig = plt.figure(constrained_layout=True, figsize=(15,10))
    fig.suptitle("Variables summary: {} - {} vs {} ".format(column_cat1, column_cat2, column_num), fontsize=20)

    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])

    x = column_cat1
    y = column_num
    hue = column_cat2
    
    
    
    df_cnts_def = pd.DataFrame()
    # Building the table
    table = ax1.table(cellText=df_describe_def.values,
          rowLabels=df_describe_def.index,
          colLabels=df_describe_def.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.1, 1.1)
    ax1.axis('off')
    #spos = Slider(ax1, 'Pos', 0.1, 90.0)

    #def update(val):
    #    pos = spos.val
    #    ax1.axis([pos,pos+10,-1,1])
    #    fig.canvas.draw_idle()

    #spos.on_changed(update)
    
    # Building box plot
    bxplt = sns.boxplot(x=x, y=y, hue=hue ,data=dataframe,
               palette=palette, order=order, ax=ax2)
    ax2.set_title("BoxPlot by category")
    # Making the mean and median lines
    mn = dataframe[column_num].mean()
    med = dataframe[column_num].median()
    ax2.axhline(mn, color="red", label="Ungrouped Mean")
    ax2.axhline(med, color="black", label="Ungrouped Median")
    ax2.legend()
    
    plt.show()


def two_numerical_vs_categorical(dataframe, column_num1, column_num2, column_cat, palette, ax, order):
    '''
    2 Numerical vs Categorical target --> Scatterplot with hue=target
    '''
    
    d1 = dataframe[[column_num1, column_num2]].describe()
    d2 = dataframe.groupby(column_cat)[[column_num1, column_num2]].describe().transpose().unstack(level=0)
    name = d2.columns.names[0]
    new_cols = d2.columns.map(lambda x: "{}_{}:{}".format(x[1], name, x[0]) )
    d2.columns = new_cols
    df_describe_def = pd.concat([d1, d2], axis=1).applymap('{:,.2f}'.format)
    
    fig = plt.figure(constrained_layout=True, figsize=(15,10))
    fig.suptitle("Variables summary: {} and {} vs {}".format(column_num1, column_num2, column_cat), fontsize=20)
    
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    #ax3 = fig.add_subplot(gs[1, 1])
    
    
    # Building the table
    table = ax1.table(cellText=df_describe_def.values,
          rowLabels=df_describe_def.index,
          colLabels=df_describe_def.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.1, 1.1)
    ax1.set_title("Summary Table")
    ax1.axis('off')
    
  
    
    sns.scatterplot(data=dataframe, x=column_num1, y=column_num2,  hue=column_cat, palette=palette, ax=ax2)
    
    
    plt.show()
    

def two_numerical_vs_numerical(dataframe, column_num1, column_num2, column_num3, palette, ax, order):
    '''
    2 Numerical vs numerical target --> heatmap
    '''
    d1 = dataframe[[column_num1, column_num2, column_num3]].describe().applymap('{:,.2f}'.format)
    corr = dataframe[[column_num1, column_num2, column_num3]].corr().applymap('{:,.2f}'.format)
    
    fig = plt.figure(constrained_layout=True, figsize=(15,10))
    fig.suptitle("Variables summary: {} and {} vs {}".format(column_num1, column_num2, column_num3), fontsize=20)
    
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
    
    #df_heatmap = dataframe[[column_num1, column_num2, column_num3]].pivot(column_num1, column_num2, column_num3)
    
    #sns.scatterplot(data=df_heatmap, x=column_num1, y=column_num2,  palette=palette, ax=ax3)
    
    hb = ax3.hexbin(dataframe[column_num1], dataframe[column_num2], C=dataframe[column_num3], gridsize=20, cmap=palette)
    ax3.set_xlabel(column_num1)
    ax3.set_ylabel(column_num2)
    ax3.set_title("HexBin Plot")
    cb = fig.colorbar(hb, ax=ax3)
    cb.set_label(column_num3)
    
    plt.show()
    
    
    
    

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










