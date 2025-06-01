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
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
##--------------------------------------------------------------------------------------------------------------------------------------------------
"""Data Processing"""
##--------------------------------------------------------------------------------------------------------------------------------------------------
#Set randome seed to have reproducible results
np.random.seed(42)
## Read data
old_data = pd.read_csv('/Users/panda/Documents/Work/Side_Learn_Projects/Side/Doordash/datasets/historical_data.csv')
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

order_protocol_dummies = pd.get_dummies(old_data.order_protocol).add_prefix('order_protocol_')
order_protocol_dummies = order_protocol_dummies.add_prefix('order_protocol_')
order_protocol_dummies.head()
# print(order_protocol_head)

market_id_dummies = pd.get_dummies(old_data.market_id).add_prefix('market_id_')
market_id_dummies = market_id_dummies.add_prefix('market_id')
market_id_dummies.head()
# print(market_id_head)

##--------------------------------------------------------------------------------------------------------------------------------------------------
"""Reference Dictionary -- maps each store_id to the most frequent cuisine_category they have"""
##--------------------------------------------------------------------------------------------------------------------------------------------------
# Fill NaNs in store_primary_category by most common per store_id
store_id_unique = old_data["store_id"].unique().tolist()
store_id_and_category = {
    store_id: old_data[old_data.store_id == store_id].store_primary_category.mode()
    for store_id in store_id_unique
}

def fill(store_id):
    """Return primary store category from the dictionary"""
    # if store_id in store_id_and_category:
    try:
        return store_id_and_category[store_id].values[0]
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
'nan_free_store_primary_category','order_protocol'])
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
# print(train_df.shape)

##--------------------------------------------------------------------------------------------------------------------------------------------------
"""There are 100 cols in the final dataset, so maybe have redundant features are not useful bc of repeating another feature
or having a 0 STD
or collinearity
"""

"""Correlation matrix -- dimension of 100x100 for better visualization, only 1 triangle"""
##--------------------------------------------------------------------------------------------------------------------------------------------------
 ##Generate a mask for the upper triangle
corr = train_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

##Plot
f,ax = plt.subplots(figsize=(11,9))

## Generate a custom diverging colormap
cmap = sns.diverging_palette(230,20, as_cmap= True)

sns. heatmap(corr, mask=mask, cmap=cmap, vmax = 0.6, center = 0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5})

##Show plot (correlation heat map)
# plt.savefig('Correlation heat map.png', dpi=300,bbox_inches='tight')
# plt.show()

##Check STD=0
# print(train_df['category_vietnamese'].describe())
# print(train_df['category_japanese'].describe())
# print(train_df['category_indonesian'].describe()) #This has SRD=0, drop this

def get_redundant_pairs(df):
    """Get diagonal and lower triangular pairs of correlation matrix"""
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    """Sort correlations in the descending order and return n highest results"""
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

# print("Top Absolute Correlations")
# print(get_top_abs_correlations(train_df, 20))

##Drop created at, market_id, store_id, order_protocol, actual_delivery_time, nan_free_store_primary_category
train_df = old_data.drop(columns = ['created_at','market_id','store_id','actual_delivery_time',
'nan_free_store_primary_category', 'order_protocol','store_primary_category'])

##Not concat market id
train_df = pd.concat([train_df, order_protocol_dummies, store_primary_category_dummies], axis =1)

##Drop highly correlated features
train_df = train_df.drop(columns = ["total_onshift_dashers","total_busy_dashers","category_indonesian","estimated_non_prep_duration"])

# Make sure all columns are numeric before conversion
# print(train_df.select_dtypes(include=['object']).columns)

##Align dtype over dataset
train_df = train_df.astype("float32")
# print(train_df.head())

##Replace inf values with Nan to drop all nans
train_df.replace([np.inf, -np.inf], np.nan, inplace =True)
##Drop all nans
train_df.dropna(inplace=True)
# print(train_df.head())

##Save the cleaned dataset
# train_df.to_csv('/Users/panda/Documents/Work/Side_Learn_Projects/Side/Doordash/datasets/cleaned_data.csv', index=False)

# print(train_df.shape) 
# (177070, 90)

# print('Top Absolute Correlations')
# print(get_top_abs_correlations(train_df, 20))

# drop created_at, market_id, store_id, store_primary_category, actual_delivery_time, order_protocol
train_df = old_data.drop(columns = ["created_at", "market_id", "store_id", "store_primary_category", "actual_delivery_time", 
                                        "nan_free_store_primary_category", "order_protocol"])
# don't concat order_protocol_dummies
train_df = pd.concat([train_df, store_primary_category_dummies], axis=1)
train_df = train_df.drop(columns=["total_onshift_dashers", "total_busy_dashers",
                                  "category_indonesian", 
                                  "estimated_non_prep_duration"])
# align dtype over dataset
train_df = train_df.astype("float32")
# replace inf values with nan to drop all nans
train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
train_df.dropna(inplace=True)
# print(train_df.head())

##New features
train_df['percent_distinct_item_of_total'] = train_df['num_distinct_items'] / train_df['total_items']
train_df['avg_price_per_item'] = train_df['subtotal'] / train_df['total_items']
train_df.drop(columns=['num_distinct_items', 'subtotal'], inplace=True)
# print('Top Absolute Correlations')
# print(get_top_abs_correlations(train_df, 20))

train_df["price_range_of_items"] = train_df["max_item_price"] - train_df["min_item_price"]
train_df.drop(columns=["max_item_price", "min_item_price"], inplace=True)
print("Top Absolute Correlations")
print(get_top_abs_correlations(train_df, 20))
print(train_df.shape)