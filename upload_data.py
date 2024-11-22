from pymongo import MongoClient
import pandas as pd
import json
import urllib.parse

# Import urllib to use quote_plus
username = "farooq"
password = urllib.parse.quote_plus("Farooq@2003")  # Encodes the special character '@'

# Construct the URL
url = f"mongodb+srv://{username}:{password}@cluster0.sxhri.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(url)

#create Database name and collection name
DATABASE_NAME = "Sensor_data"
COLLECTION_NAME = "waterfault"

df =pd.read_csv("C:\Users\Farooq\OneDrive\Desktop\onedrive\Desktop\sensor_project\notebooks\wafer_23012020_041211.csv")

df=df.drop("Unnamed: 0", axis=1)

json_record= list(json.loads(df.T.to_json()).values())

client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)

