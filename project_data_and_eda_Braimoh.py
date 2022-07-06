# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 16:54:26 2020

@author: OLAWALE
"""
"""
Project Introduction

Going by the merit-order scheme in Germany’s uniform pricing power market, 
the amount of electricity delivered by renewable sources at different points 
in time has an influence on the electricity prices. Thus, the impact of 
renewable energy sources on electricity market prices needs to be acknowledged. 
Consequently, in this project, we investigate and analyze the relationship between 
renewable electricity generation; solar and wind in particular, and electricity 
market price. The investigation involves some exploratory data analysis and 
correlation analysis, furthermore, some prediction models are developed to predict 
the electricity market price. The performances of the models are analyzed and we 
are able to conclude on a suitable model for forecasting the electricity market price.
    
Data Preparation

Here, the resolution of the data used was in hourly time steps, as such, all 
retrieved data used had to be in the same format. The demand data for 2018 and 
2019 were only available in quarter-hourly resolution and had to be adjusted to 
the hourly form like the other parameters. While, for 2020, the demand and renewable 
generation data were also only available in quarter-hourly resolution, they were 
also adjusted to hourly form.

The time of day parameter was divided with an assumption of 6 am to 6 pm for day time 
and 7 pm to 5 am for night time. While the other variables such as the hour of the day 
and day of the week were included accordingly.

For the exploratory data analysis, the data was also grouped into monthly and exact 
daily categories in order to help with the investigation of the relationships of the 
parameters with the electricity market price.

Model Development

For the forecasting of the electricity market price, three models were taken into 
consideration for development. The models are: Randomforest model, XGBoost, and 
Auto-regression Integrated Moving Average (ARIMA). The execution of the models 
involved training them with the 2018 and 2019 data and then making predictions for 
the first six months of 2020.

Data Sources
● https://www.energy-charts.de/power_inst.htm
● https://www.renewables.ninja/
● https://data.open-power-system-data.org/time_series/
● https://smard.de

"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%%
df = pd.read_csv('data/demand_data2018_2019.csv', sep=';')
pv_cf = pd.read_csv('data/pv_cf.csv', skiprows=2)
wind_cf = pd.read_csv('data/wind_cf.csv', skiprows=2)
capacity = pd.read_csv('data/re_capacity.csv', index_col=0)
price = pd.read_csv('data/Day-ahead_prices_2018-2019.csv', sep=';')

demand_2020 = pd.read_csv('data/Forecasted_demand_2020.csv', sep=';', parse_dates={'DateTime' : [0, 1]}, thousands=',')
price_2020 = pd.read_csv('data/Day-ahead_prices_2020.csv', sep=';', parse_dates={'DateTime' : [0, 1]})
generation_2020 = pd.read_csv('data/Actual_generation_2020.csv', sep=';', thousands=',', parse_dates={'DateTime' : [0, 1]})

#%%

demand_2020['Total[MWh]'] = demand_2020['Total[MWh]'].apply(lambda x: '0' if x == '-' else x)

demand_2020['Total[MWh]'] = demand_2020['Total[MWh]'].apply(lambda x: ''.join(x.split(','))).astype(float)

demand_2020 = demand_2020.resample('H', on='DateTime').sum()

# Demand is missing for this
# 2020-03-29 02:00:00         NaN
# The missing demand values were replaced by the averaged values of that day
demand_2020.loc['2020-03-29 02:00:00'] = demand_2020.loc['2020-03-29',:].mean()

# Demand also missing for 2020-01-31 (Weekday)
# In order to have a complete data,
# their values were replaced with 2020-01-30 (the previous day)
jan_30 = demand_2020[(demand_2020.index >= '2020-01-30') & (demand_2020.index < '2020-01-31')]
demand_2020.loc[demand_2020['Total[MWh]']==0,'Total[MWh]'] = jan_30['Total[MWh]'].values

#%%

generation_2020 = generation_2020.resample('H', on='DateTime').sum()
generation_2020['Wind prod(MWh)'] = generation_2020['Wind offshore[MWh]'] + generation_2020['Wind onshore[MWh]']

#%%
final_2020 = pd.concat([demand_2020.rename(columns = {'Total[MWh]': 'Demand(MWh)'}),
                        generation_2020['Wind prod(MWh)']], axis=1)
#%%
final_2020['solar_prod(MWh)']  = generation_2020['Photovoltaics[MWh]']
final_2020['Price(€/MWh)'] = price_2020.set_index('DateTime')['Germany/Luxembourg[€/MWh]']

#%%
# Price and Wind data is missing for 2020-03-29 02:00:00
# They were replaced by the averaged values of that day
final_2020.loc['2020-03-29 02:00:00','Price(€/MWh)'] = final_2020.loc['2020-03-29',:].mean()['Price(€/MWh)']
final_2020.loc['2020-03-29 02:00:00','Wind prod(MWh)'] = final_2020.loc['2020-03-29',:].mean()['Wind prod(MWh)']

#%% 
# =============================================================================
# Cleaning up Demand Data
# =============================================================================

df['DT'] = df['Date'] + ' ' + df['Time of day']

df.drop(['Date', 'Time of day'],axis=1, inplace=True)

df['DateTime'] = pd.to_datetime(df['DT'])

# On 24/09/2018, 20/10/2019 Total demand was recorded as '-', This line replaces those with 0.
df['Total[MWh]'] = df['Total[MWh]'].apply(lambda x: '0' if x == '-' else x)

df['Total[MWh]'] = df['Total[MWh]'].apply(lambda x: ''.join(x.split(','))).astype(float)

df = df.resample('H', on='DateTime').sum()

# Demand is missing for these
# 2018-03-25 02:00:00         NaN
# 2019-03-31 02:00:00         NaN
# This accounts for the 17518 non-null values in the Total column.
# %%
# =============================================================================
# Manually fixing NaN
# =============================================================================
# The missing demand values were replaced by the averaged values of that day
df.loc['2018-03-25 02:00:00'] = df.loc['2018-03-25',:].mean()
df.loc['2019-03-31 02:00:00'] = df.loc['2019-03-31',:].mean()

# Demand was also missing for the whole of 2018-09-25 (weekday - Monday) and 2019-10-21 (Sunday)
# In order to have a complete data,
# their values were replaced with 2018-09-26 (the next day) and 2019-10-13 (the previous Sunday) respectively
sep_25 = df[(df.index >= '2018-09-25') & (df.index < '2018-09-26')]
df.loc[(df.index >= '2018-09-24') & (df.index < '2018-09-25'),'Total[MWh]']= sep_25['Total[MWh]'].values

oct_13 = df[(df.index >= '2019-10-13') & (df.index < '2019-10-14')]
df.loc[(df.index >= '2019-10-20') & (df.index < '2019-10-21'),'Total[MWh]']= oct_13['Total[MWh]'].values

#%%
# This is required because the resample() method automatically sets DateTime as index
df.reset_index(inplace=True)

#%%
# =============================================================================
# Wind and PV data
# =============================================================================
wind_cf['DateTime'] = pd.to_datetime(wind_cf['time'])

wind_cf_18 = wind_cf[wind_cf['DateTime']>='2018-01-01'].copy()
wind_cf_18.drop('time', inplace=True, axis=1)

wind_cf_18['wind_onshore[MWh]'] = wind_cf_18['onshore'] * wind_cf_18['DateTime'].apply(lambda x: capacity['wind_onshore'].loc[x.year])
wind_cf_18['wind_offshore[MWh]'] = wind_cf_18['offshore'] * wind_cf_18['DateTime'].apply(lambda x: capacity['wind_offshore'].loc[x.year])

wind_cf_18['Wind prod[MWh]'] = wind_cf_18['wind_offshore[MWh]'] + wind_cf_18['wind_onshore[MWh]']

# This resets the index and starts counting at 0 again. This makes sure that 
# all indexes start and end the same
wind_cf_18.reset_index(drop=True, inplace=True)

#%%
pv_cf['DateTime'] = pd.to_datetime(pv_cf['time'])

pv_18 = pv_cf.loc[pv_cf['DateTime']>='2018-01-01'].copy()
pv_18.drop('time', inplace=True, axis=1)

pv_18['solar_prod[MWh]'] = pv_18['national'] * pv_18['DateTime'].apply(lambda x: capacity['solar'].loc[x.year])
pv_18.reset_index(drop=True, inplace=True)

#%%
# =============================================================================
# Price data
# =============================================================================

price['DateTime'] = price['Date'] + ' ' + price['Time of day']

price['DateTime'] = pd.to_datetime(price['Date'])

price.drop('Time of day', inplace=True, axis=1)

# =============================================================================
# Combining all into a single data frame
# =============================================================================
#%%
price['Price[€/MWh]'] = (price['Germany/Luxembourg[€/MWh]'].apply(lambda x: '0' if x == '-' else x).astype(float)
                    + price['Germany/Austria/Luxembourg[€/MWh]'].apply(lambda x: '0' if x == '-' else x).astype(float))

final = pd.concat([df, wind_cf_18['Wind prod[MWh]'], pv_18['solar_prod[MWh]'], price['Price[€/MWh]']], axis=1)

final.rename(columns = {'Total[MWh]': 'Demand[MWh]'}, inplace=True)

# %%
# =============================================================================
# Column creation
# =============================================================================
def day_of_week(datetime):
    if datetime.weekday() < 5:
        return 'Weekday'
    elif datetime.weekday() == 5:
        return 'Saturday'
    else: 
        return 'Sunday'

   
#%%
def create_columns(df):
    df['isWeekday'] = df['DateTime'].apply(day_of_week)
    
    df['Time of day'] = df['DateTime'].apply(lambda x: 'Day Time' if x.hour >= 6 and x.hour <= 18 else 'Night Time')
    
    df['Hour'] = df['DateTime'].apply(lambda x: float(str(x.hour)))
    
    df['Hour_'] = df['DateTime'].apply(lambda x: float(str(x.hour)[:2]))
    
    df['Day'] = df['DateTime'].apply(lambda x: x.day_name())
    
    # month_name is an inbuilt method of the pandas Timestamp object.
    df['Month'] = df['DateTime'].apply(lambda x: x.month_name())
    
#%%
create_columns(final)

#%%
final_2020.reset_index(inplace=True)
create_columns(final_2020)

#%%
# =============================================================================
# Exploratory Data Analysis   
# =============================================================================
final_month_resample = final.set_index('DateTime').resample('M', label=None).sum()

final_monthly_2018 = final[final['DateTime']<'2019'].groupby('Month', sort=False).mean()
final_monthly_2019 = final[final['DateTime']>'2018'].groupby('Month', sort=False).mean()

#%%
# =============================================================================
# Using Renewable Production
# =============================================================================
#%%

MonthHour = final.groupby(by=['Hour','Month'], sort=False).mean()['solar_prod[MWh]'].unstack()

plt.figure(figsize = (18,9))
plt.title('Plot 1: Heatmap of Average Solar Production [MWh] - Month and Hour')
sns.heatmap(MonthHour, cmap='coolwarm', annot=True, fmt= ".2f")
plt.savefig('plots/Plot 1 Heatmap of Average Solar Production - Month and Hour.jpg')
#%%

MonthHour_w = final.groupby(by=['Hour','Month'], sort=False).mean()['Wind prod[MWh]'].unstack()

plt.figure(figsize = (18,9))
plt.title('Plot 2: Heatmap of Average Wind Production [MWh] - Month and Hour')
sns.heatmap(MonthHour_w, cmap='coolwarm', annot=True, fmt= ".2f")
plt.savefig('plots/Plot 2 Heatmap of Average Wind Production - Month and Hour.jpg')
#%%
# =============================================================================
# Using Price
# =============================================================================
TimeDay = final.groupby(by=['isWeekday','Time of day'], sort=False).mean()['Price[€/MWh]'].unstack()

plt.figure(figsize = (10,5))
plt.title('Plot 3: Heatmap of Average Price [€/MWh] - Weekday and Time of Day')
sns.heatmap(TimeDay, annot=True, fmt = ".2f")
plt.savefig('plots/Plot 3 Heatmap of Average Price - Weekday and Time of Day.jpg')
#%%
final['Hour'] = final['DateTime'].apply(lambda x: x.time())

Hour_Day = final.groupby(by=['Hour', 'Day'], sort=False).mean()['Price[€/MWh]'].unstack()

plt.figure(figsize = (11,7))
plt.title('Plot 4: Heatmap of Average Price [€/MWh] - Hour and Day')
sns.heatmap(Hour_Day, cmap='coolwarm')
plt.savefig('plots/Plot 4 Heatmap of Average Price - Hour and Day.jpg')
#%%
# month_name is an inbuilt method of the pandas Timestamp object.
final['Month'] = final['DateTime'].apply(lambda x: x.month_name())
MonthHour_p = final.groupby(by=['Hour','Month'], sort=False).mean()['Price[€/MWh]'].unstack()

plt.figure(figsize = (18,9))
plt.title('Plot 5: Heatmap of Average Price [€/MWh] - Month and Hour')
sns.heatmap(MonthHour_p, annot=True, fmt= ".2f")
plt.savefig('plots/Plot 5 Heatmap of Average Price - Month and Hour.jpg')
#%%
# =============================================================================
# Using Demand
# =============================================================================
TimeDay_d = final.groupby(by=['isWeekday','Time of day'], sort=False).mean()['Demand[MWh]'].unstack()

plt.figure(figsize = (10,5))
plt.title('Plot 6: Heatmap of Average Demand - Weekday and Time of Day')
sns.heatmap(TimeDay_d, annot=True, fmt = ".2f")
plt.savefig('plots/Plot 6 Heatmap of Average Demand [MWh] - Weekday and Time of Day.jpg')
#%%
Hour_Day_d = final.groupby(by=['Hour', 'Day'], sort=False).mean()['Demand[MWh]'].unstack()

plt.figure(figsize = (11,7))
plt.title('Plot 7: Heatmap of Average Demand [MWh] - Hour and Day')
sns.heatmap(Hour_Day_d)
plt.savefig('plots/Plot 7 Heatmap of Average Demand - Hour and Day.jpg')
#%%

MonthHour_d = final.groupby(by=['Hour','Month'], sort=False).mean()['Demand[MWh]'].unstack()

plt.figure(figsize = (18,9))
plt.title('Plot 8: Heatmap of Average Demand [MWh] - Month and Hour')
sns.heatmap(MonthHour_d, annot=True, fmt= ".2f")
plt.savefig('plots/Plot 8 Heatmap of Average Demand - Month and Hour.jpg')
#%%
# =============================================================================
# Preparing data for the model
# =============================================================================
#%%
# This removes [] from column names as forbidden by XGBoost
def xgb_rename(x):    
    for var in x.columns:
        if '[' in var or ']' in var:
            one = '('.join(var.split('['))
            two = ')'.join(one.split(']'))
            x.rename(columns = {var: two}, inplace=True)

xgb_rename(final)


def data_prep(df):    
    df.set_index('DateTime', inplace=True)
    df.drop(['Hour','Month', 'Day'], axis=1, inplace=True)
  
data_prep(final)
data_prep(final_2020)  

final = pd.get_dummies(final, columns=['isWeekday','Time of day']) 
final_2020 = pd.get_dummies(final_2020, columns=['isWeekday','Time of day'])

#%%
# =============================================================================
# Correlation Analysis
# =============================================================================
matrix = final.corr()
price_corr = pd.DataFrame(matrix['Price(€/MWh)'])
print('Correlation of Parameters with Price')
print(price_corr)
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_title('Plot 9: Correlation of Parameters with Price') 
sns.heatmap(price_corr, vmax=1, square=True, cmap="BuPu")
plt.savefig('plots/Plot 9 Correlation of Parameters with Price.jpg', bbox_inches='tight')

# =============================================================================
# Exporting the preparing data to csv
# to be used in the project_models.py file
# =============================================================================
#final.to_csv('data/finaldata.csv')
#final_2020.to_csv('data/finaldata2020.csv')