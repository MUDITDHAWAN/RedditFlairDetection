

import csv

import pandas as pd

import pymongo
### connect with my clusster
from pymongo import MongoClient

### mongodb uri for connection
client = MongoClient("mongodb://MuditAdmin:Admin%40123@cluster0-shard-00-00-krhzl.mongodb.net:27017,cluster0-shard-00-01-krhzl.mongodb.net:27017,cluster0-shard-00-02-krhzl.mongodb.net:27017/test?ssl=true&replicaSet=Cluster0-shard-0&authSource=admin&retryWrites=true&w=majority")

### database creation
db = client.Flair_reddit_data

### collection creation
collection = db.data

### read file to be uploaded
data_read = pd.read_csv('datafinal.csv')

### uploaded the file using insert_many()
collection.insert_many(data_read.to_dict('records'))
