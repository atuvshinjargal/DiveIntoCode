import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import missingno as msno
from pathlib import Path  


# Understanding the content of the competition
'''
Home Credit dataset

1. What kind of company is Home Credit?
    Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience.
2. What do you expect in this competition?
    Prediction their clients' repayment abilities and unlock the full potential of their data.
3. What are the benefits companies can get by predicting that?
    Doing so will ensure that clients capable of repayment are not rejected 
    and that loans are given with a principal, maturity, 
    and repayment calendar that will empower their clients to be successful.
'''

# Understanding the overview of data

application_train = pd.read_csv("home-credit-default-risk/application_train.csv")
print("============ head() ===============")
print(application_train.head())
print("============ describe() ===============")
print(application_train.describe())
print("============ info() ===============")
print(application_train.info)
print('============ column names ==============')
print('total feature types: {}'.format(application_train.shape[0]), list(application_train.columns.values))


print("============ missing value information ===============")
print("total data length is {}".format(len(application_train)))

def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_info = missing_data(application_train)
print(missing_info.head(10))
filepath = Path('out.csv')  
missing_info.to_csv(filepath,index = False)

print("============ the percentage of classes ===============")

temp = application_train["TARGET"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
plt.title('The percentage of classes')
sns.barplot(x = 'labels', y="values", data=df)
#locs, labels = plt.xticks()
plt.show()

print('From the barplot of target, this dataset is unbalanced dataset')

# Project setting
print('1. Dataset is an unbalanced \
       2. Almost half of feature has missing values about 50 percent \
        3. Majority of the clients have Secondary / secondary special education, followed by clients with Higher education. \
            In terms of percentage of not repayment of loan, lower secondarty has the highest percent of not repayment (10%), with academic the lowest.\
        4. Most of applicants for loans are income from Working, followed by Commercial associate, Pensioner and State servant.\
            The applicants with the type of income Maternity leave have almost (40%) ratio of not returning loans, followed by Unemployed (37%). \
        5. Most of applicants for loans are income from Working, and the type of income Maternity leave and Unemployed have almost (77%) ratio of not returning loans')


# Data search
# 1. Unbalanced by target
# 2. too many missing values
j=0
for i in range (len(missing_info)):
    if missing_info.iloc[i,1] > 50:
        j = j + 1

#tempo = [['above 50%', j], ['below 50%', len(missing_info)-j]] 
df = pd.DataFrame({'labels': ['above 50 percent', 'below 50%'],
                   'features': [j, len(missing_info)-j]
                  })
plt.title('Missing value')
sns.barplot(x = 'labels', y="features", data=df)
#locs, labels = plt.xticks()
plt.show()

def plot_stats(feature,label_rotation=False,horizontal_layout=True):
    temp = application_train[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index,'Number of contracts': temp.values})

    # Calculate the percentage of target=1 per category value
    cat_perc = application_train[[feature, 'TARGET']].groupby([feature],as_index=False).mean()
    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)
    
    if(horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12,14))
    sns.set_color_codes("pastel")
    s = sns.barplot(ax=ax1, x = feature, y="Number of contracts",data=df1)
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    
    s = sns.barplot(ax=ax2, x = feature, y='TARGET', order=cat_perc[feature], data=cat_perc)
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.ylabel('Percent of target with value 1 [%]', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.show()
# 3. some stats
plot_stats('CODE_GENDER')
plot_stats('NAME_EDUCATION_TYPE')
plot_stats('OCCUPATION_TYPE')
plot_stats('NAME_INCOME_TYPE',False,False)

# Plot distribution of one feature
def plot_distribution(feature,color):
    plt.figure(figsize=(10,6))
    plt.title("Distribution of %s" % feature)
    sns.distplot(application_train[feature].dropna(),color=color, kde=True,bins=100)
    plt.show()   

plot_distribution('AMT_ANNUITY', 'red')



