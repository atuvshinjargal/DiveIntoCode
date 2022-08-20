import numpy as np 
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt 
import seaborn as sns

data = load_iris()

############## Problem 1 ################
# Data acquisition

a = pd.DataFrame(data = data.data, columns = data.feature_names)
b = pd.DataFrame(data = data.target, columns = ['species'])
print(a.head())
print(b.head())

# Data combination

df = pd.concat([a,b], axis=1)
print(df.head())

# Confirmation of data

print(df.info())
print(df.describe())
print('fourth sample: {}'.format(df.loc[4]))
print('Count samples for each label: {}'.format(df.value_counts('sepal width (cm)')))
count = np.count_nonzero(df.isnull())
print('There is {} missing values.'.format(count))

# Investigation of the iris dataset itself

# get columns 1st method using loc
sepal_width = df.loc[:, 'sepal width (cm)']
# get columns 2nd method using iloc
sepal_width1 = df.iloc[:, 1]

sub_data = df.iloc[50:100,:]
print(sub_data)
sub_data_petal_length = sub_data.loc[:,'petal length (cm)']
print(sub_data_petal_length)
matched_df = df[df["petal width (cm)"] == 0.2]
print(matched_df)

# Creating a diagram

a = df.groupby('species').sum()

species_count = df['species'].value_counts(normalize=True)*100
print("count value", species_count)
species_count1 = df.groupby('species').size()
print("count value", species_count1.shape)

#print("count", species_count.iloc[:,0])
plt.pie(species_count, labels=['0 class', '1 class', '2 class'], autopct='%0.1f%%')
plt.legend(title = 'Classes', loc = 'center left', bbox_to_anchor = (1,0,0.5,1))
plt.title("Pie chart")
plt.show()
def plot_box(df, name):
    plt.boxplot((df))
    plt.title(name)
    plt.xticks([1,2,3,4],["sepal_length", "sepal_width", "petal_length", "petal_width"])
    plt.ylabel('cm')
    plt.show()

def plot_violin(df, name):
    plt.violinplot((df))
    plt.title(name)
    plt.xticks([1,2,3,4],["sepal_length", "sepal_width", "petal_length", "petal_width"])
    plt.ylabel('cm')
    plt.show()

df_0 = df[df["species"] == 0].iloc[:,0:4]
plot_box(df_0, "0th")
df_1 = df[df["species"] == 1].iloc[:,0:4]
plot_box(df_1, "1st")
df_2 = df[df["species"] == 2].iloc[:,0:4]
plot_box(df_2, "2nd")

plot_violin(df_0, '0th')
plot_violin(df_1, '1st')
plot_violin(df_2, '2nd')

# Confirmation of the relationship between features

g = sns.pairplot(df, vars=["sepal width (cm)", "sepal length (cm)"], hue="species", kind='scatter')
plt.show()

g = sns.pairplot(df, vars=["petal width (cm)", "petal length (cm)"], hue="species", kind='scatter')
plt.show()

g = sns.pairplot(df, vars=["sepal width (cm)", "petal width (cm)"], hue="species", kind='scatter')
plt.show()

g = sns.pairplot(df, vars=["sepal length (cm)", "petal length (cm)"], hue="species", kind='scatter')
plt.show()

g = sns.pairplot(df, hue = 'species', kind='scatter')
plt.show()

corr = df.corr(method = 'pearson')
print(corr)

h = sns.heatmap(corr)
plt.title('Correlation')
plt.show()

print('from the matrix and figures, I want to state few results: \
    1. 1 and 2 classes are similar with sepal length and sepal width \
    2. first class iris is very different from 1 and 2 class with all features.')