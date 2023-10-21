import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statistical_tests import proportionDiffTest, twoSampleZTest

alpha = 0.05

def summary_for_category(df, category):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # Visualization
    sns.countplot(data=df, x=category, hue=None, ax=ax[0])
    sns.countplot(data=df, x=category, hue='subscription', ax=ax[1])
    ax[0].tick_params(axis='x', rotation=90)
    ax[1].tick_params(axis='x', rotation=90)

    plt.show()

    # Subscription rate by category
    sub_rates = {}
    for occupation in df[category].unique():
        total = len(df[df[category] == occupation])
        subscribed = len(df[(df[category] == occupation) & (df['subscription'] == 'yes')])
        rate = subscribed / total
        
        sub_rates[occupation] = rate

    # Sort by values
    sub_rates_sorted = sorted(sub_rates, key=sub_rates.get)
    for category_item in sub_rates_sorted:
        print(f'Subcription rate for {category}={category_item} : {sub_rates[category_item]:.2f}')

    # Check if any group has a significantly different subscription rate
    p0 = len(df[df['subscription'] == 'yes']) / len(df)
    print(f'Overall subscription rate : {p0:.2f}')
    for category_item in sub_rates_sorted:
        p1 = sub_rates[category_item]
        n1 = len(df[df[category] == category_item])
        
        others = df[df[category] != category_item]
        others_subscribed = len(others[others['subscription'] == 'yes'])
        p2 = others_subscribed / len(others)
        n2 = len(others)
        
        Z, pvalue = proportionDiffTest(p0, p1, p2, n1, n2)
        print('--------------------------------------------------------------------')
        print(f'For {category_item}')
        print(f'  - n1 = {n1}, n2 = {n2}, p1 = {p1:.2f}, p2 = {p2:.2f}')
        print(f'  - Z-statistic for proportion = {Z:.4f}, p-value = {pvalue:.4f}')
        
def summary_for_numerical(df, category):
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    df[category].hist(ax=axes[0])
    sns.boxplot(data=df, y=category, x='subscription', ax=axes[1])
    sns.pointplot(data=df, y=category, x='subscription', ax=axes[2])
    plt.show()
    
    # Difference in the two group
    # Is there a significant difference in age between the two groups
    X1 = df[df['subscription'] == 'yes'][category].values
    X2 = df[df['subscription'] == 'no'][category].values

    # Check if mu_subscribed \ne mu_nosubscribe
    Z, pvalue = twoSampleZTest(X1, X2)
    print(f'Z-statistic = {Z}, pvalue = {pvalue}')

    # Conclude (2-tailed test)
    if(pvalue < alpha/2):
        print('There is a significant difference')
    else:
        print('There is no significant difference')
