# -*- coding: utf-8 -*-
import datetime

import pandas as pd
from sklearn import linear_model

import numpy as np
import math

SECS_IN_DAY = 24*60*60


orders_path = 'data/debug/train_set/debug_c10_p100_d50_int0_pmod_cond_orders.csv'
group_size = 5

dataset = pd.read_csv(orders_path)
print dataset.head()
client_cnt = dataset['clientId'].nunique()
product_cnt = dataset['productId'].nunique()

print 'Clients:', client_cnt, 'Products', product_cnt


df = dataset

df['clientId'] = df['clientId'].astype(int)
df['productId'] = df['productId'].astype(int)
df['day_group'] = df['datetime'] / (SECS_IN_DAY * group_size)
df['day_group'] = df['day_group'].astype(int)
df['ordered'] = 1
df = df.drop('datetime', axis=1)

print df.head()

df = df.groupby(['clientId','productId']).sum()


print df['ordered'].loc[33,:]