#!/usr/bin/env python
# coding: utf-8

# In[75]:


import pandas as pd #this is for loadingdataset
import warnings #to get rid of warnings in python
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt


# In[3]:


df=pd.read_csv('C:/Users/Student/Desktop/My Project/HousePricePrediction.xlsx - Sheet1.csv')


# In[4]:


df.shape


# In[5]:


df.head


# In[6]:


df.drop_duplicates(inplace=True)


# In[7]:


df.shape#if it gives you same rows and column,which means no duplication


# In[11]:


#Delete unwanted column
df.drop(columns=['Id'],inplace=True)#inplace=true-Delete or make changes permanently


# In[12]:


df.isna().sum() #check the null values


# In[14]:


#as we got to predict salePrice,therefore as a target value,it is impossible for it to contain null values because it will afffect the perfomance.
# to replace null values , you have several techniques.
#1.First you can go mean for that column and i will do that using simple impute from sk learn.
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy='mean')
#fit the imputer
imputer.fit(df[['SalePrice']]) # we use double square brackets for it not to consider it as a datsframe


# In[16]:


imputer.statistics_# it will show the mean for that specific column


# In[17]:


df['SalePrice']=imputer.transform(df[['SalePrice']]) #transform the imputer into the Array column for SalePrice


# In[18]:


df.isna().sum()# check the count of null values once again 


# In[19]:


# Replace other column with null values using zeros(0's) because they are not important
df=df.fillna(0)


# In[20]:


df.isna().sum()# you must check check if there is still null values


# In[21]:


df.describe() #You will able to see the statistics of data-set for every column


# In[22]:


#You can conclude from describtion that LotArea column contains outliers because ther eis a hudge different from  75% of data-set and the maximum value.
# To analyse  the outliers , you must use boxplot
import matplotlib.pyplot as plt
import seaborn as sns


# In[23]:


sns.set_style('darkgrid')


# In[24]:


sns.boxplot(df,y='LotArea')


# In[25]:


#You can handle outliers using different method
#1.IQR method that is inter Quatal range
#2. Square method
#3. standard deviation method
#4. Percentile method

#i will use IQR method as i think is apppropriate .
# for performing numerical array , you need to import numerical python known as numpy

import numpy as np
Q1 = np.percentile(df['LotArea'],25,interpolation='midpoint')
Q3 = np.percentile(df['LotArea'],75,interpolation='midpoint')
#As we know that the QUARTILE one contains the 25% of the data-set and QUARTILE three contains 75% of the data-set
#interpolation='midpoint': The method used for interpolation when the percentile falls between two data points. The 'midpoint' interpolation method takes the average of the two values closest to the desired percentile.
IQR = Q3 - Q1
# Interquartile Range (IQR): The difference between Q3 and Q1 represents the spread of the middle 50% of the data. It’s often used to detect outliers.


# In[26]:


#set the threshold,which is threshold lower bond and threshhold upper bonf.
#Outliers are typically defined as values that are below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR.
lowerBound = Q1 - 1.5 * IQR#Everything lower than this bond is considered as outlier
upperBound = Q1 + 1.5 * IQR#Everything above upperBond is considered as outliers


# In[27]:


#We create new data frame which fit the instances,lowerbond and upperbond
df = df[(df.LotArea < upperBound) & (df.LotArea > lowerBound)]


# In[28]:


#check the numer of rows and columns now
df.shape


# In[29]:


#check the data types
df.info


# In[30]:


df


# In[31]:


df.info()# check the data types


# In[34]:


df.info()# check the if the set got  categorial values or not because you cannt fit categorial values in a machine learning, you have to converst them to numerical values.


# In[33]:


df.MSZoning.unique()


# In[35]:


cat_cols = df.select_dtypes('object').columns.tolist()


# In[36]:


cat_cols#check categorial values


# In[41]:


#to convert categorial values to numerical values ,,i need to do pre-processing
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output= False, handle_unknown='ignore')


encoder.fit(df[cat_cols])#include the list of all categorical column from the data frame and i have perfomed the encoder operation which convert them to numerical values as OneHotEncoder formalize them to zeroes and ones 


#This imports the OneHotEncoder class from the sklearn.preprocessing module.
#OneHotEncoder is used to convert categorical data into a numerical format that machine learning algorithms can understand. It creates a binary (0 or 1) column for each possible category/level of the feature.
#sparse_output=False:
#1.By default, OneHotEncoder returns a sparse matrix (a memory-efficient way to store large matrices). Setting sparse=False makes the encoder return a dense array instead of a sparse matrix.
#2.Sparse matrices are useful when the data is large and contains many zeros, but dense arrays are easier to work with when the data size is smaller.
#handle_unknown='ignore':
#1.This argument specifies how to handle categories that appear in the test data but are not present in the training data.
#2.'ignore' means that if an unknown category is encountered during transformation, it will be encoded as a row of all zeros instead of throwing an error.


# In[42]:


#how OneHotEncoder convert to zeros and ones
#1. if the category is swimming, draw, and music,it will create spreadsheet which assign one to the chosen hoppy and assign zeros to other two unchoosen hobby
#To create column automatically
encoded_cols = encoder.get_feature_names_out(cat_cols)
#1.This method retrieves the names of the features after transformation by the OneHotEncoder. Specifically, these names correspond to the new columns created during the encoding process.
#2. cat_cols is the list of categorical columns that you passed to the encoder, and the resulting names represent the possible categories in each column.
encoded_cols


# In[43]:


df[encoded_cols] = encoder.transform(df[cat_cols])
#encoder.transform(df[cat_cols]):
#1..transform() is used to apply the one-hot encoding to the data.
#2. df[cat_cols] refers to the columns in the DataFrame df that contain categorical data, which you have previously selected (and cat_cols is a list of these column names).
#3. The transform() method converts the categorical data into one-hot encoded format. For example, if cat_cols is the Color column and has values like ['Red', 'Green', 'Blue'], this method will convert it into multiple columns with binary values (0 or 1), each representing one of the categories.
#df[encoded_cols]
#1.encoded_cols is a list of the new column names generated after the one-hot encoding (e.g., Color_Red, Color_Green, Color_Blue).
#2.This line assigns the one-hot encoded data (result from .transform()) to the columns in df named encoded_cols.


# In[44]:


#We can now delete catecorigal column 
df.drop(columns=cat_cols, inplace=True)


# In[49]:


#I now have to devide the data-set into input feature and target variable from all the display columns
#First thing i check columns
df.columns


# In[50]:


df


# #remeber that SalePrice is what we want to predict,however all columns are input feature,so we need to devide Input featurewith the target variable.
# 
# X = df.drop(columns = 'SalePrice')
# y = df['SalePrice']

# In[51]:


#remeber that SalePrice is what we want to predict,however all columns are input feature,so we need to devide Input featurewith the target variable.
X = df.drop(columns = 'SalePrice')#contains all input features
y = df['SalePrice']#contains only salePrice


# In[52]:


#all input values got different scale,however i need to bring all these columns together,into one standard scale.
# We need minmax library from sk learn,and this process is called standardization and normalization
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X)
MinMaxScaler()


# In[53]:


#Now i must store it by transforming it.
X[:] = scaler.transform(X)# i use x[:] to dect it for storing it in the form of Array
X


# In[54]:


#Deviding data into the training and testing set(the purpose is to test the module)
#you need to use the train test split function from the sk learn model selection library
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#1. from sklearn.model_selection import train_test_split:
#This imports the train_test_split function from the model_selection module of the scikit-learn library.
#The train_test_split function is used to split the dataset into training and testing sets.

#X:
# This represents the input features (i.e., independent variables) of your dataset. It could be a pandas DataFrame or a numpy array that contains the features you want to use to train the model.
# In supervised machine learning, X is often a DataFrame or array with all the feature columns, excluding the target variable.

#y:
#This is the target variable (dependent variable) in your dataset. It contains the values you're trying to predict, such as house prices in a regression problem or labels in a classification task.

#test_size=0.2:
#This specifies the proportion of the data to be used as the test set.
#test_size=0.2 means 20% of the data will be used for testing, and 80% will be used for training.
#The value can range from 0 to 1, where test_size=0.3 would use 30% of the data for testing, leaving 70% for training.

#random_state=42:
#This ensures the results are reproducible.
#Setting a random_state (seed) allows the splitting of data to be consistent across different runs. In this case, the number 42 is an arbitrary seed, and any number can be used. If you omit this, the split will be different each time you run the code.


# In[55]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape#The train_test_split method ensures that your data is split into training and testing sets, and .shape helps to verify the number of samples in each set.


# In[71]:


#Importing linear regression model fromsk learn
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)
LinearRegression()


# In[57]:


y_pred = model.predict(X_test)


# In[58]:


y_test[:5]# print original SalePrice values


# In[59]:


y_pred[:5]# print predicted values for salePrice


# In[60]:


#Comparing values,you can see that our model is over fitting ,we can make predictions and evaluate the performance.

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred)


# In[83]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

#Mean Squared error(MSE)
#Measures the average squared differences between predicted and actual values.
#Smaller MSE indicates better model performance.
#Sensitive to outliers because it squares the differences, giving larger errors more weight..

#R-squared (R²):
#Measures how well the model explains the variance in the target variable.
#Closer to 1 means better performance.
#Negative R² indicates a poor model.
#R² = 1: Perfect model — the predictions match the actual values exactly.

#R² = 0: The model does no better than simply predicting the mean of the target variable.

#Negative R²: The model performs worse than simply predicting the mean of the target variable (e.g., when the model’s predictions are far off from actual values).


# In[63]:


#it is clear that the the model does not fit,however i can perform regulation to make model fit(lasso regulation(L1),ridge regulation(L2))
#lasso regulation(is a linear model that uses L1 regularization to penalize the absolute magnitude of the regression coefficients. It is often used for feature selection in regression tasks, as it can shrink some of the coefficients to zero, effectively removing the corresponding features.)

from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=50, max_iter=100, tol = 0.1)

lasso_reg.fit(X_train, y_train)
#1.alpha=50:
#This is the regularization strength for the Lasso model. The larger the value of alpha, the stronger the regularization.
#Lasso regression adds a penalty to the linear regression cost function based on the sum of the absolute values of the coefficients (the L1 norm). Increasing alpha increases the regularization strength, which can shrink the coefficients toward zero.
#Effect of alpha:
#If alpha is very high, the model will heavily penalize large coefficients, which could result in a simpler model with many coefficients set to zero.
#If alpha is too low, the model may perform similarly to ordinary least squares regression without regularization, and overfitting could occur.
#2. max_iter=100:
#This defines the maximum number of iterations the optimization algorithm will perform to converge to the solution.
#Default is usually 1000, but you can set it lower to reduce computation time if necessary, or increase it if you think the algorithm might require more iterations to converge.
#3tol=0.1:
#This specifies the tolerance for the optimization algorithm's stopping criteria. The optimization will stop when the change in the cost function (or loss function) is smaller than the tolerance value tol.
#Lower tolerance values (e.g., 0.01) require more precise convergence, while higher tolerance values can speed up the process but may result in a less accurate model.


# In[69]:


y_pred = lasso_reg.predict(X_test)
y_pred[:5]


# In[80]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print(f"R-squared: {r2}")


# In[67]:


mean_absolute_error(y_test,y_pred)


# In[70]:


#checking the ridge regulation
from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=50, max_iter=100, tol = 0.1)

ridge_reg.fit(X_train, y_train)


# In[72]:


ridge_pred = ridge_reg.predict(X_test)


# In[73]:


mean_absolute_error(y_test, ridge_pred)


# In[ ]:


#in conclusion it is clear that lass regression model,is the one which fit ,looking at MSE for all model.

