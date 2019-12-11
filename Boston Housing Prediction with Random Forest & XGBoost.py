#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_boston


# In[57]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graphviz


# In[3]:


boston = load_boston()


# In[4]:


print(boston.keys())


# In[5]:


data = pd.DataFrame(boston.data)


# In[6]:


#Asssign names to columns
data.columns = boston.feature_names


# In[7]:


data.head()


# In[8]:


#Add field for reponse variable
data['PRICE'] = boston.target


# In[9]:


data.head()


# In[93]:


data.describe()


# In[95]:


data['PRICE'].plot(kind = 'hist')


# In[27]:


#Import XGBoost 
import xgboost as xgb


# In[17]:


from sklearn.metrics import mean_squared_error


# In[18]:


#Separate features from response field
X, y = data.iloc[:,:-1], data.iloc[:,-1]


# In[19]:


#Create DMatrix for Cross Validation
data_dmatrix = xgboost.DMatrix(data=X, label=y)


# In[22]:


#Partition dataset
from sklearn.model_selection import train_test_split


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state= 123)


# In[76]:


from sklearn.ensemble import RandomForestRegressor 


# In[80]:


rf = RandomForestRegressor(n_estimators = 400, min_samples_leaf=0.12, random_state=123)


# In[81]:


rf.fit(X_train, y_train)


# In[84]:


rf_pred = rf.predict(X_test)
rf_pred2 = rf.predict(X_train)


# In[86]:


rmse_test = mean_squared_error(y_test, rf_pred)**(1/2)
rmse_train = mean_squared_error(y_train, rf_pred2)**(1/2)


# In[88]:


print("Test set RMSE: %f " % (rmse_test))
print("Train set RMSE: %f " % (rmse_train))


# In[96]:


rf_params = {'max_depth': [3,4,5,6],'min_samples_leaf':[0.04,0.06,0.08], 'max_features':[0.2,0.4,0.6,0.8]}


# In[98]:


grid_rf = GridSearchCV(estimator = rf,
                     param_grid = rf_params,
                     cv=10,
                     scoring='neg_mean_squared_error',
                     verbose=1)


# In[99]:


grid_rf.fit(X_train, y_train)


# In[101]:


print('Best parameters found for rf: ', grid_rf.best_params_)
print('Lowerst RMSE found for rf: ', np.sqrt(np.sqrt(np.abs(grid_rf.best_score_))))


# In[104]:


#Test this model on unseen data. First, list attributes
list(X.columns)


# In[154]:


#Take a look at the data to help formulate an example observation
data.describe()


# In[144]:


#Create new 'scenario' array 
new = np.array([0.04, 10, 20, 0.01, 0.4, 5, 40, 3, 10, 380, 20, 300, 9])


# In[145]:


#Check new observation 
print(new)


# In[146]:


#Make sure that the array has 13 features and 1 observation
new.shape


# In[150]:


new_obs_final = new.reshape(1,-1)


# In[151]:


new_obs_final.shape


# In[152]:


#Predict price for new values
grid_rf.predict(new_obs_final)


# In[92]:


#Most important features for model variance
pd.Series(rf.feature_importances_, index = X.columns).sort_values(ascending = False)


# In[29]:


#Let's attempt an XGBoost Regressor to see if we can get a better performance
xg_reg = xgb.XGBRegressor(objective = 'reg:linear', colsample_bytree= 0.3, learning_rate = 0.1,
                            max_depth = 5, alpha = 10, n_estimators = 10)


# In[30]:


xg_reg.fit(X_train, y_train)


# In[38]:


preds = xg_reg.predict(X_test)
preds2 = xg_reg.predict(X_train)


# In[39]:


rmse = np.sqrt(mean_squared_error(y_test, preds))
rmse2 = np.sqrt(mean_squared_error(y_train, preds2))
print("Attmept 1 Test RMSE: %f" % (rmse))
print("Attempt 1 Train RMSE: %f" % (rmse2))


# In[40]:


#Add cross validation to yield better performance. First, create params dictionary
params = {'objective':'reg:linear', 'colsample_bytree':0.3, 'learning_rate':0.1, 'maz_depth':5,
         'alpha':10}


# In[41]:


#Now cross validate
cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold = 3, num_boost_round=50, early_stopping_rounds=10,
                   metrics="rmse", as_pandas=True,seed=123)


# In[56]:


#Review output of results
cv_results.head()


# In[43]:


#Review cross validated results
print((cv_results['test-rmse-mean']).tail(1))
print((cv_results['train-rmse-mean']).tail(1))


# In[64]:


#Although better, let's add grid search to find best parameters. First, import module
from sklearn.model_selection import GridSearchCV


# In[68]:


#Second, define parameters
gbm_param_grid = {'learning rate': [0.01, 0.1, 0.5, 0.9], 'n_estimators': [200], 'subsample':[0.3,0.5,0.9]}


# In[69]:


#Third, initialize model and run cross validation on initialized estimator
gbm = xgb.XGBRegressor()
grid_mse = GridSearchCV(estimator=gbm, param_grid=gbm_param_grid, scoring='neg_mean_squared_error', cv=4, verbose=1)


# In[70]:


#Fourth, fit cross validated model
grid_mse.fit(X, y)


# In[73]:


#Print best performing parameters and their RMSE scores
print('Best parameters found: ', grid_mse.best_params_)
print('Lowerst RMSE found: ', np.sqrt(np.sqrt(np.abs(grid_mse.best_score_))))


# In[52]:


#Feature importance vizualized
xgb.plot_importance(xg_reg)

