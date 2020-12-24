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


def compute_summary_categorical(dataframe, column):
    d = {}
    c = dataframe[column].value_counts(dropna=False)
    p = dataframe[column].value_counts(dropna=False, normalize=True)
    d[column] = c
    d[column + "_pct"] = p
    return pd.DataFrame(d)


def compute_summary_numerical_vs_categorical(dataframe, column_num, column_cat):
    d1 = dataframe[column_num].describe()
    d2 = dataframe.groupby(column_cat)[column_num].describe().transpose()
    d2.columns = ["{}_{}:{}".format(column_num, column_cat,c) for c in d2.columns]
    df_summary = pd.concat([d1, d2], axis=1)
    
    return df_summary


def compute_summary_categorical_vs_categorical(dataframe, column_cat1, column_cat2, normalize='both'):
    
    if (normalize == 'yes'):
        d2 = pd.crosstab(index=dataframe[column_cat1], columns=dataframe[column_cat2], margins=True, normalize='index')
        return d2
    elif (normalize == 'no'):
        d1 = pd.crosstab(index=dataframe[column_cat1], columns=dataframe[column_cat2], margins=True)
        return d1
    elif (normalize == 'both'):
        d1 = pd.crosstab(index=dataframe[column_cat1], columns=dataframe[column_cat2], margins=True)
        d2 = pd.crosstab(index=dataframe[column_cat1], columns=dataframe[column_cat2], margins=True, normalize='index')
        return d1, d2
    else:
        return 'Error'
    

def compute_summary_numerical_vs_numerical(dataframe, column_num1, column_num2):
    return dataframe[[column_num1, column_num2]].describe()
 

def compute_summary_2categorical_vs_categorical(dataframe, column_cat1, column_cat2, column_cat3, normalize='both'):
    
    if (normalize == 'yes'):
        d2 = pd.crosstab(index=[dataframe[column_cat1], dataframe[column_cat2]], columns=dataframe[column_cat3], margins=True, normalize='index')
        return d2
    elif (normalize == 'no'):
        d1 = pd.crosstab(index=[dataframe[column_cat1], dataframe[column_cat2]], columns=dataframe[column_cat3], margins=True)
        return d1
    elif (normalize == 'both'):
        d1 = pd.crosstab(index=[dataframe[column_cat1], dataframe[column_cat2]], columns=dataframe[column_cat3], margins=True)
        d2 = pd.crosstab(index=[dataframe[column_cat1], dataframe[column_cat2]], columns=dataframe[column_cat3], margins=True, normalize='index')
        return d1, d2
    else:
        return 'Error'

    
def compute_summary_2categorical_vs_numerical(dataframe, column_cat1, column_cat2, column_num):    
    
    d1 = dataframe[column_num].describe().to_frame(column_num)
    d2 = dataframe.groupby([column_cat1, column_cat2])[column_num].describe()
    d2.index = ["{}_{}:{}_{}:{}".format(column_num, column_cat1, x[0], column_cat2, x[1]) for x in d2.index.to_flat_index()]
    return pd.concat([d1, d2.transpose()], axis=1)


def compute_summary_2numerical_vs_categorical(dataframe, column_num1, column_num2, column_cat):
    d1 = dataframe[[column_num1, column_num2]].describe()
    d2 = dataframe.groupby([column_cat])[column_num1].describe().transpose()
    d2.columns = ["{}_{}:{}".format(column_num1, column_cat, x) for x in d2.columns ]
    d3 = dataframe.groupby([column_cat])[column_num2].describe().transpose()
    d3.columns = ["{}_{}:{}".format(column_num2, column_cat, x) for x in d3.columns ]
    #d2.columns = d2.columns.droplevel(0)
    return pd.concat([d1, d2, d3], axis=1)


def compute_summary_2numerical_vs_numerical(dataframe, column_num1, column_num2, column_num3):
    return dataframe[[column_num1, column_num2, column_num3]].describe()



def categorical_univariate(dataframe, column, palette, show_plot=True):
    
    df_summary = compute_summary_categorical(dataframe, column)
    
    fig, axs = plt.subplots(figsize=(15,8))
    
    y = column
    #axs[0].table(cellText=df_cnts.values,
    #      rowLabels=df_cnts.index,
    #      colLabels=df_cnts.columns, loc='center')
    #axs[0].axis('off')
    cntplot = sns.countplot(y=y, data=dataframe,
                  palette=palette);
    
    if (show_plot):
        plt.show();
    
    return (fig, axs), df_summary


def numerical_univariate(dataframe, column, palette, show_plot=True):
    
    df_summary = dataframe[column].describe()
    
    fig, axs = plt.subplots(1, 2,figsize=(10,8))
    
    y = column
    #axs[0].table(cellText=df_cnts.values,
    #      rowLabels=df_cnts.index,
    #      colLabels=df_cnts.columns, loc='center')
    #axs[0].axis('off')
    #sns.countplot(y=y, data=dataframe,
    #              palette=palette, order=order, ax=axs)
    
    hst = sns.histplot(data=dataframe, x=column, palette=palette, ax=axs[0],  kde=True);
    boxplot = sns.boxplot(data=dataframe, y=column, palette=palette, ax=axs[1]);
    
    if (show_plot):
        plt.show();
    
    return (fig, axs), df_summary




def categorical_vs_categorical(dataframe, column_cat1, column_cat2, palette):
    #plt.table(cellText=dcsummary.values,colWidths = [0.25]*len(dc.columns),
    #      rowLabels=dcsummary.index,
    #      colLabels=dcsummary.columns,
    #      cellLoc = 'center', rowLoc = 'center',
    #      loc='top')
    
    df_summary = compute_summary_categorical_vs_categorical(dataframe, column_cat1, column_cat2, normalize='both')
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
    sns.countplot(y=y, hue=hue, data=dataframe,
                  palette=palette, ax=ax[1])
    ax[1].set_title("Values")
    plt.tight_layout()
    
    plt.show()
    
    return (fig, ax), df_summary


def categorical_vs_numerical(dataframe, column_num, column_cat, palette):
    #plt.table(cellText=dcsummary.values,colWidths = [0.25]*len(dc.columns),
    #      rowLabels=dcsummary.index,
    #      colLabels=dcsummary.columns,
    #      cellLoc = 'center', rowLoc = 'center',
    #      loc='top')
    
    df_summary = compute_summary_numerical_vs_categorical(dataframe, column_num, column_cat)
    
    # Describe dataframe
    d1 = dataframe[column_num].describe()
    d2 = dataframe.groupby(column_cat)[column_num].describe().transpose()
    d2.columns = ["{}_{}_{}".format(column_num, column_cat,c) for c in d2.columns]
    df_describe_def = pd.concat([d1, d2], axis=1)

    fig, ax = plt.subplots(1, 2, figsize=(15,10))
    
    
    #print(df_cnts.reset_index())
    # Building box plot
    bxplt = sns.boxplot(x=column_cat, y=column_num, hue=None ,data=dataframe,
               palette=palette,ax=ax[0])
    ax[0].set_title("BoxPlot by category")
    # Making the mean and median lines
    mn = dataframe[column_num].mean()
    med = dataframe[column_num].median()
    ax[0].axhline(mn, color="red", label="Ungrouped Mean")
    ax[0].axhline(med, color="black", label="Ungrouped Median")
    ax[0].legend()
    
    # Building distributions plot
    sns.kdeplot(data=dataframe, x=column_num, hue=column_cat,  palette=palette, ax=ax[1])
    #sns.countplot(y=y, hue=hue, data=dataframe,
    #              palette=palette, order=order, ax=ax3)
    ax[1].set_title("Distributions by category")
    

    plt.tight_layout()
    
    plt.show()
    
    return (fig, ax), df_summary
    
    

    
def numerical_vs_numerical(dataframe, column_num1, column_num2, palette):
    # Numerical vs Numerical target --> Describe for both variable and target; Scatterplot
    df_summary = dataframe[[column_num1, column_num2]].describe()
    #corr = dataframe[[column_num1, column_num2]].corr()
    
    fig, ax = plt.subplots(1, 1, figsize=(15,10))
    
    sns.scatterplot(data=dataframe, x=column_num1, y=column_num2,  palette=palette, ax=ax)
    
    
    plt.show()
    
    return (fig, ax), df_summary
    
    

def two_categorical_vs_categorical(dataframe, column_cat1, column_cat2, column_cat3, palette):
    '''
    2 Categorical vs Categorical target --> Count + pct for each category and target category; Count + pct for each target category; For each value of target (column_cat3), two heatmaps: one with percentages of the corresponding value for each pair of category values, and one with the difference between the first one, and the pct of the target value without grouping (baseline).
    '''
    df_summary = compute_summary_2categorical_vs_categorical(dataframe, column_cat1, column_cat2, column_cat3)
    
    target_values = dataframe[column_cat3].unique().tolist()
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
    
    
    plt.show()
    
    return (fig, ax), df_summary



def two_categorical_vs_numerical(dataframe, column_cat1, column_cat2, column_num, palette):
    '''
    2 categorical vs numerical target --> Grouped boxplot with x=categorical, y=target, hue=categorical
    '''
    
    df_summary = compute_summary_2categorical_vs_numerical(dataframe, column_cat1, column_cat2, column_num)
    
    
    fig, ax = plt.subplots(1,1, figsize=(15,10))
    
    x=column_cat1
    y=column_num
    hue=column_cat2
    
    # Building box plot
    bxplt = sns.boxplot(x=x, y=y, hue=hue ,data=dataframe,
               palette=palette, ax=ax)
    ax.set_title("BoxPlot by category")
    # Making the mean and median lines
    mn = dataframe[column_num].mean()
    med = dataframe[column_num].median()
    ax.axhline(mn, color="red", label="Ungrouped Mean")
    ax.axhline(med, color="black", label="Ungrouped Median")
    ax.legend()
    
    plt.show()
    
    return (fig, ax), df_summary


def two_numerical_vs_categorical(dataframe, column_num1, column_num2, column_cat, palette):
    '''
    2 Numerical vs Categorical target --> Scatterplot with hue=target
    '''
    
    df_summary = compute_summary_2numerical_vs_categorical(dataframe, column_num1, column_num2, column_cat)
    
    fig, ax = plt.subplots(1, 1, figsize=(15,10))
    
    sns.scatterplot(data=dataframe, x=column_num1, y=column_num2,  hue=column_cat, palette=palette, ax=ax)
    
    
    plt.show()
    
    return (fig, ax), df_summary
    

def two_numerical_vs_numerical(dataframe, column_num1, column_num2, column_num3, palette):
    '''
    2 Numerical vs numerical target --> hexbin
    '''
    
    df_summary = compute_summary_2numerical_vs_numerical(dataframe, column_num1, column_num2, column_num3)
    
      
    fig, ax = plt.subplots(1, 1, figsize=(15,10))
    fig.suptitle("Variables summary: {} and {} vs {}".format(column_num1, column_num2, column_num3), fontsize=20)
    
   
    hb = ax.hexbin(dataframe[column_num1], dataframe[column_num2], C=dataframe[column_num3], gridsize=20, cmap=palette)
    ax.set_xlabel(column_num1)
    ax.set_ylabel(column_num2)
    ax.set_title("HexBin Plot")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label(column_num3)
    
    plt.show()
    
    return (fig, ax), df_summary
    


def summaryEDA(dataframe, numerical_vars, categorical_vars, target, target_type='numerical', show_plot=False, palette=None):
    '''
    Computes univariate, bivariate and trivariate summaries and charts and stores them in a dictionary
    '''
    summary_dict = {}
    
    
    # Univariate plots and summaries
    for num_var in numerical_vars:
        (fig, axs), df_summary = numerical_univariate(dataframe, num_var, palette, show_plot)
        summary_dict[num_var] = {}
        summary_dict[num_var]['figure'] = fig
        summary_dict[num_var]['axes'] = axs
        summary_dict[num_var]['summary'] = df_summary
        
    
    for cat_var in categorical_vars:
        (fig, axs), df_summary = categorical_univariate(dataframe, cat_var, palette, show_plot)
        summary_dict[num_var] = {}
        summary_dict[num_var]['figure'] = fig
        summary_dict[num_var]['axes'] = axs
        summary_dict[num_var]['summary'] = df_summary
        
    # Numerical vs target
    for num_var in numerical_vars:
        if (target_type == 'numerical'):
            (fig, axs), df_summary = numerical_vs_numerical(dataframe, num_var, target, palette)
        elif (target_type == 'categorical'):
            (fig, axs), df_summary = categorical_vs_numerical(dataframe, num_var, target, palette)
        k = (num_var, target)
        summary_dict[k] = {}
        summary_dict[k]['figure'] = fig
        summary_dict[k]['axes'] = axs
        summary_dict[k]['summary'] = df_summary
    
    # Categorical vs target
    for cat_var in categorical_vars:
        if (target_type == 'numerical'):
            (fig, axs), df_summary = categorical_vs_numerical(dataframe, target, cat_var, palette)
        elif (target_type == 'categorical'):
            (fig, axs), df_summary = categorical_vs_categorical(dataframe, cat_var, target, palette)
        k = (cat_var, target)
        summary_dict[k] = {}
        summary_dict[k]['figure'] = fig
        summary_dict[k]['axes'] = axs
        summary_dict[k]['summary'] = df_summary
    
    # 2 Numerical vs target
    for i, num_var_i in enumerate(numerical_vars):
        for j, num_var_j in enumerate(numerical_vars):
            if (i < j):
                if (target_type == 'numerical'):
                    (fig, axs), df_summary = two_numerical_vs_numerical(dataframe, num_var_i, num_var_j, target, palette)
                elif (target_type == 'categorical'):
                    (fig, axs), df_summary = two_numerical_vs_categorical(dataframe, num_var_i, num_var_j, target, palette)
                k = (num_var_i, num_var_j, target)
                summary_dict[k] = {}
                summary_dict[k]['figure'] = fig
                summary_dict[k]['axes'] = axs
                summary_dict[k]['summary'] = df_summary
    
    
    # 2 Categorical vs target
    for i, cat_var_i in enumerate(categorical_vars):
        for j, cat_var_j in enumerate(categorical_vars):
            if (i < j):
                if (target_type == 'numerical'):
                    (fig, axs), df_summary = two_categorical_vs_numerical(dataframe, cat_var_i, cat_var_j, target, palette)
                elif (target_type == 'categorical'):
                    (fig, axs), df_summary = two_categorical_vs_categorical(dataframe, cat_var_i, cat_var_j, target, palette)
                k = (cat_var_i, cat_var_j, target)
                summary_dict[k] = {}
                summary_dict[k]['figure'] = fig
                summary_dict[k]['axes'] = axs
                summary_dict[k]['summary'] = df_summary
    
    # 1 Numerical and 1 categorical vs target
    for i, num_var_i in enumerate(numerical_vars):
        for j, cat_var_j in enumerate(categorical_vars):
            if (target_type == 'numerical'):
                (fig, axs), df_summary = two_numerical_vs_categorical(dataframe, num_var_i, target, cat_var_j, palette)
            elif (target_type == 'categorical'):
                (fig, axs), df_summary = two_categorical_vs_numerical(dataframe, cat_var_j, target, num_var_i, palette)
            k = (num_var_i, cat_var_j, target)
            summary_dict[k] = {}
            summary_dict[k]['figure'] = fig
            summary_dict[k]['axes'] = axs
            summary_dict[k]['summary'] = df_summary
    
    
    
    return summary_dict
    
   


if __name__ == "__main__":
    pass










