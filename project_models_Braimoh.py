# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 22:26:38 2020

@author: OLAWALE
"""
"""
Project Summary

In this project, we were able to present visual relationships between renewables 
production and electricity market price, and make viable deductions and connections. 
Although the solar production gave the lowest correlation, it was understandable 
due to the severe seasonality of solar energy and its availability all year round. 
Alongside, the renewables production parameters, the models still took the other 
provided parameters into consideration for the prediction, although, some parameters 
were taken of more importance than others.

All the models used were able to make predictions that followed the pattern of 
the real prices for the first six months of 2020. It could be noted that the XGBoost 
Regressor model was mostly the closest one to the real prices. This becomes further 
evident from the evaluation of the models with the error metrics whereby the XGBoost 
Regressor had the lowest MAE and RMSE among all the models and the highest 
R-Squared score. The Randomforest model appeared to be the worst performing of the 
three models, while it had the highest MAE and RMSE, it also presented a negative 
R-Squared score which indicated a bad model fit.

All in all, we were able to establish that predictions on electricity market price 
can be made to some extent of accuracy just by taking in information of renewable 
energy technologies. From our results, we can conclude that the XGBoost Regressor 
model is a very good model for forecasting electricity market price.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%%

final = pd.read_csv('data/finaldata.csv', index_col=0, parse_dates=True)
final_2020 = pd.read_csv('data/finaldata2020.csv', index_col=0, parse_dates=True)

#%%
# =============================================================================
# Training Models
# =============================================================================

#%%
Features = ['Demand(MWh)', 'Wind prod(MWh)', 'solar_prod(MWh)', 'Hour_', 
            'isWeekday_Saturday', 'isWeekday_Sunday', 'isWeekday_Weekday',
            'Time of day_Day Time', 'Time of day_Night Time']

#%%
# =============================================================================
# Splitting the dataset to train and test models
# =============================================================================
#The 2018 and 2019 data is used as the train model
X_train = final[Features].copy()
y_train = final['Price(€/MWh)'].copy()

#The 2020 data is used as the test model
X_test = final_2020[Features].copy()
y_test = final_2020['Price(€/MWh)'].copy()


#%%
# =============================================================================
# Assessing Accuracy and Error Metrics
# =============================================================================
from sklearn import metrics

def evaluate(model, X_test=X_test, predictions = None):
    try:
        predictions = model.predict(X_test)    
    except:
        pass
    print('Model Performance')
    print ('Mean Absolute Error (MAE): {:0.4f}'.format(metrics.mean_absolute_error(y_test, predictions)))
    print ('Root Mean Squared Error (RMSE): {:0.4f}'.format(np.sqrt(metrics.mean_squared_error(y_test, predictions))))
    print ('R-squared Score: {:0.4f}'.format(metrics.r2_score(y_test, predictions)))
   
#%%
# =============================================================================
# Random Forest
# =============================================================================
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)

rfr_preds = rfr.predict(X_test)

#%%
print('\nFor Random Forest Regressor')
evaluate(rfr)

#%%
# =============================================================================
# Extreme Gradient Boost Regressor (XGBoost)
# =============================================================================
import xgboost as xgb

xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', n_estimators = 1000)

xg_reg.fit(X_train, y_train,
           eval_set=[(X_train, y_train), (X_test, y_test)],
           early_stopping_rounds=50,
           verbose=False)

xgb_preds = xg_reg.predict(X_test)

#%%
print('\nFor Extreme Gradient Boost Regressor')
evaluate(xg_reg)

xgb.plot_importance(xg_reg, title='Plot 10: Extreme Gradient Boost Regressor- Feature Importance ')
plt.savefig('plots/Plot 10 XGBoost Feature Importance.jpg')
#%%
# =============================================================================
# SARIMAX 
# =============================================================================
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(y_train,
                exog=X_train,  
                order = (5, 1, 0),
                ) 
  
result = model.fit() 
result.summary() 

#%%
start = len(X_train) 
end = len(X_train) + len(X_test) - 1
  
# Predictions for six months against the test set 
ar_preds = result.predict(start=start, end=end,
                              exog=X_test, typ = 'levels').rename("SARIMAX") 

#%%
print('\nFor SARIMAX')
evaluate(result, predictions=ar_preds)
#%%
# =============================================================================
# Plotting the model results
# =============================================================================
rfr_df = pd.DataFrame(rfr_preds, index=y_test.index, columns=['randomforest'])
xgb_df = pd.DataFrame(xgb_preds, index=y_test.index, columns=['xgboost'])

all_preds = pd.concat([y_test, rfr_df, xgb_df, ar_preds], axis=1)

all_preds.plot()
plt.title('Plot 11: Plot of Prediction Models with Real Price - Hourly')
plt.ylabel('Price(€/MWh)')
plt.savefig('plots/Plot 11 Plot of Prediction Models with Real Price Hourly.jpg')
#%%
# For better viewing, the hourly price values are averaged to daily resolution

fig, ax = plt.subplots(figsize=(14,6))
plt.plot(y_test.resample('D', label=None).mean(), label='Real Price')
plt.plot(rfr_df.resample('D', label=None).mean(), label='Random Forest')
plt.plot(xgb_df.resample('D', label=None).mean(),  label='XGBoost')
plt.plot(ar_preds.resample('D', label=None).mean(),  label='SARIMAX')

ax.set_xticklabels(list(y_test.index.month_name().unique()) + ['July'])
ax.set_title('Plot 12: Plot of Prediction Models with Real Price - Averaged Daily')
ax.legend()
ax.set_ylabel('Price(€/MWh)')
plt.savefig('plots/Plot 12 Plot of Prediction Models with Real Price Averaged Daily.jpg')
