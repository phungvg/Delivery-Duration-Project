""" 
Delivery Prediction Model
-- Info:
- Dataset: historical_data.csv from DoorDash in early 2015
- Time features: UTC
- Model will predict the total delivery duration seconds
- Workflow is 
+ Firstly, customers place order, there will be estimated_order_place_duration
+ Secondly, restaurant receives order, which is retaurant preparation time
+ Thirdly, restaurant prepares meal, which is estimated store to consumer driving duration
+ Lastly, meal is delivered to consumer
"""
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
from datetime import datetime
##--------------------------------------------------------------------------------------------------------------------------------------------------
"""Data Processing"""
##--------------------------------------------------------------------------------------------------------------------------------------------------
## Read data
old_data = pd.read_csv('/Users/panda/Documents/Work/Strata Scratch/Doordash/datasets/historical_data.csv')
# print(old_data.head())
# print(old_data.info())

old_data['created_at'] = pd.to_datetime(old_data['created_at'])
old_data['actual_delivery_time'] = pd.to_datetime(old_data['actual_delivery_time'])

##Feature creation
##Target = actual delivery time - order created time in seconds
old_data["actual_total_delivery_duration"] = (old_data['actual_delivery_time'] - old_data['created_at']).dt.total_seconds()

##Total number of available dashers within a certain area will change from time to time
##Percentage of Dasher available at the time of order creation 
old_data['busy_dasher_ratio'] = old_data['total_busy_dashers']/old_data['total_onshift_dashers']

## First,second,third in the workflow
old_data['estimated_non_prep_duration'] = old_data['estimated_store_to_consumer_driving_duration'] + old_data['estimated_order_place_duration']

##check ids to encode or not
old_data['market_id'].nunique()
old_data['store_id'].nunique()
old_data['order_protocol'].nunique()
# print(check)
# print(check1)
# print(check2)

order_protocol_dummies = pd.get_dummies(old_data.order_protocol)
order_protocol_dummies = order_protocol_dummies.add_prefix('order_protocol_')
order_protocol_dummies.head()
# print(order_protocol_head)

market_id_dummies = pd.get_dummies(old_data.market_id)
market_id_dummies = market_id_dummies.add_prefix('market_id')
market_id_dummies.head()
# print(market_id_head)

##--------------------------------------------------------------------------------------------------------------------------------------------------
"""Reference Dictionary -- maps each store_id to the most frequent cuisine_category they have"""
##--------------------------------------------------------------------------------------------------------------------------------------------------
def fill(store_id):
    """Return primary store category from the dictionary"""
    # if store_id in store_id_and_category:
    try:
        return store_id_and_category[store_id].value[0]
    except:
        return np.nan
##Filling Null values
old_data['nan_free_store_primary_category'] = old_data.store_id.apply(fill)

##One hot encode the colunm
store_primary_category_dummies = pd.get_dummies(old_data.nan_free_store_primary_category)
store_primary_category_dummies = store_primary_category_dummies.add_prefix('category_')
store_primary_category_dummies.head()

##Removing org column with these one hot encoded values
train_df = old_data.drop(columns = ['created_at','market_id','store_id','store_primary_category','actual_delivery_time',
'nan_free_store_primary_category'])
# print(train_df.head())

##Concatenate all column together
train_df = pd.concat([train_df, order_protocol_dummies,market_id_dummies, store_primary_category_dummies], axis =1)

##Convert to float for the future use
# Convert timedelta to seconds before casting
train_df = train_df.astype("float32")
train_df.head()
# print(train_df.describe())
# print(train_df['busy_dasher_ratio'].describe())

##Max:inf as busy_dasher_ratio, if a #/0 = infinity

##Check infinity values
np.where(np.any(~np.isfinite(train_df), axis=0) == True)
##Replace infinity values with nan to drop all nans
train_df.replace([np.inf, -np.inf], np.nan, inplace =True)
##Drop all nans
train_df.dropna(inplace=True)
print(train_df.shape)