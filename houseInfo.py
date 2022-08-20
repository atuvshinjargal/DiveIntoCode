from xml.etree.ElementInclude import include
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns

# Acquisition of data set
data = pd.read_csv('train.csv')
print(data.head())


# Investigation of the data set itself
'''
LotFrontage: Linear feet of street connected to property
LotArea: Lot size in square feet
Street: Type of road access 
SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
MSSubClass: The building class
MSZoning: The general zoning classification
'''
print(data.dtypes)
print(data.select_dtypes(include=object))
print(data.describe())

# Dealing with missing values

missing_val = msno.bar(data)
plt.show()

missing_dat = data.isnull().sum()
counted = data.count()
missing_ratio = counted/len(data)

ratio = {'Total':counted, 'missing_ratio' : missing_ratio}
print(pd.DataFrame(ratio))


# delete features
for col in data.columns:
    if data[col].isnull().sum() > 5:
        data.drop(col, axis = 1, inplace = True)

print("The shape of cleaned data is {}".format(data.shape))
# delete samples

data.dropna(axis=0,  inplace = True)
print("Samples of cleaned data is {}".format(len(data)))


# Terminology survey
'''
Kurtosis is a measure of the combined weight of a distribution's tails relative to the center of the distribution.
 When a set of approximately normal data is graphed via a histogram, 
 it shows a bell peak and most data within three standard deviations (plus or minus) of the mean. 

'''

kurtosis = data.kurtosis(axis=0)
print('kurtosis: {}'.format(kurtosis))

'''
Skewness is a measure of the asymmetry of a distribution. 
A distribution is asymmetrical when its left and right side are not mirror images.
A distribution can have right (or positive), left (or negative), or zero skewness. 
'''

skewness = data.skew(axis = 0)
print('skewness of data: {}'.format(skewness))

# Confirmation of distribution
kurt_lotarea = data['SalePrice'].kurtosis(axis=0)
skew_lotarea = data['SalePrice'].skew(axis=0)


sns.displot(data=data['SalePrice'],  kde = True)
sns.displot(data=data['SalePrice'], log_scale= True, kde = True)
plt.text(50000,100, 'kurtosis = '+str(kurt_lotarea)  , fontsize=10)
plt.text(50000,90, ' skewness = '+ str(skew_lotarea) , fontsize=10)
plt.show()


#  Check the correlation coefficient

corr = data.corr(method = 'pearson')
print(corr)

h = sns.heatmap(corr)
plt.title('Correlation')
plt.show()


# highly correlated with Sales Price 
T_corr=corr.index[abs(corr['SalePrice'])>0.51]
g = sns.heatmap(data[T_corr].corr(),annot=True)
plt.show()

print('Strong correlations are: GrLivArea_Log and TotRmsAbvGrd; GarageCars and GarageArea; TotalBsmtSF and 1stFlrSF')

print('3 important variables are: 1. OverallQual: Overall material and finish quality, 2. GrLivArea: Above grade (ground) living area square feet and GarageCars: 3. Size of garage in car capacity')
