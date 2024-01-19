#!/usr/bin/env python
# coding: utf-8

# <img src="https://rhyme.com/assets/img/logo-dark.png" align="center"> 
# 
# <h2 align="center">Simple Linear Regression</h2>

# Linear Regression is a useful tool for predicting a quantitative response.

# We have an input vector $X^T = (X_1, X_2,...,X_p)$, and want to predict a real-valued output $Y$. The linear regression model has the form

# <h4 align="center"> $f(x) = \beta_0 + \sum_{j=1}^p X_j \beta_j$. </h4>

# The linear model either assumes that the regression function $E(Y|X)$ is linear, or that the linear model is a reasonable approximation.Here the $\beta_j$'s are unknown parameters or coefficients, and the variables $X_j$ can come from different sources. No matter the source of $X_j$, the model is linear in the parameters.

# ### Task 2: Loading the Data and Importing Libraries
# ---

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')


# The adverstiting dataset captures sales revenue generated with respect to advertisement spends across multiple channles like radio, tv and newspaper. [Source](http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv)

# In[2]:


advert = pd.read_csv('I:\\Coursera\\Guided Projects\\Predict Sales Revenue with Simple Linear Regression on\\Advertising.csv')
advert.head()


# In[3]:


advert.info()


# ### Task 3: Remove the index column

# In[4]:


advert.columns


# In[5]:


advert.drop(['Unnamed: 0'], axis = 1, inplace = True)
advert.head()


#  

# ### Task 4: Exploratory Data Analysis

# In[6]:


import seaborn as sns
sns.histplot(advert.sales)


# In[7]:


sns.histplot(advert.newspaper)


# In[8]:


sns.histplot(advert.radio)


# In[9]:


sns.histplot(advert.TV)


# In[ ]:





#  

#  

# ### Task 5: Exploring Relationships between Predictors and Response

# In[10]:


sns.pairplot(advert, x_vars=['TV', 'radio', 'newspaper'], y_vars='sales', height=7,
            aspect=0.7, kind='reg')


# In[11]:


advert.TV.corr(advert.sales)


# In[12]:


advert.corr()


# In[13]:


sns.heatmap(advert.corr(), annot=True)


# ### Task 6: Creating the Simple Linear Regression Model

# General linear regression model:
# $y=\beta_{0}+\beta_{1}x_{1}+\beta_{2}x_{2}+...+\beta_{n}x_{n}$
# 
# - $y$  is the response
# - $\beta_{0}$ is the intercept
# - $\beta_{1}$ is the coefficient for  x1  (the first feature)
# - $\beta_{n}$ is the coefficient for  xn  (the nth feature)
# 
# In our case: $y=\beta_{0}+\beta_{1}×TV+\beta_{2}×Radio+\beta_{3}×Newspaper$
# 
# The $\beta$ values are called the **model coefficients*:
# 
# - These values are "learned" during the model fitting step using the "least squares" criterion
# - The fitted model is then used to make predictions

# In[14]:


X = advert[['TV']]
X.head()


# In[15]:


print(type(X))
print(X.shape)


# In[16]:


y = advert.sales
print(type(y))
print(y.shape)


# In[17]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# In[18]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[20]:


from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)


# In[ ]:





#  

#  

# ### Task 7: Interpreting Model Coefficients

# In[21]:


print(linreg.intercept_)
print(linreg.coef_)


# In[ ]:





# ### Task 8: Making Predictions with our Model

# In[22]:


y_pred = linreg.predict(X_test)
y_pred[:5]


#  

# ### Task 9: Model Evaluation Metrics

# In[23]:


true = [100, 50, 30, 20]
pred = [90, 50, 50, 30]


# **Mean Absolute Error** (MAE) is the mean of the absolute value of the errors:;
# $$ \frac{1}{n} \sum_{i=1}^{n} \left |y_i - \hat{y}_i \right |$$

# In[24]:


print((10+0+20+10)/4)

from sklearn import metrics
print(metrics.mean_absolute_error(true, pred))


# **Mean Squared Error** (MSE) is the mean of the squared errors:
# $$\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

# In[25]:


print((10**2+0**2+20**2+10**2)/4)
print(metrics.mean_squared_error(true, pred))


# **Root Mean Squared Error** (RMSE) is the square root of the mean of the squared errors:
# $$\sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

# In[26]:


print(np.sqrt((10**2+0**2+20**2+10**2)/4))

print(np.sqrt(metrics.mean_squared_error(true, pred)))


# In[27]:


print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:




