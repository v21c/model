
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from bson import ObjectId
import json

uri = "mongodb+srv://admin:admin1234@cluster0.0mfe7qt.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client = MongoClient(uri, server_api=ServerApi('1'))

db = client['intheview']
# 약간 수동 데이터 넣기
user_collection = db['users']

# user_collection.update_one(
#     {"_id":ObjectId("66a3324a1cc4ca962e4c9afe")},
#     { "$set" :{"sessions":[]}}
# )
# user_collection.insert_one(user_data)

#삭제
# user_collection.delete_one({"_id":ObjectId('')})


session_collection = db['sessions']


# session_collection.insert_one(session_data)

session_collection.delete_one({"_id":ObjectId('66a34e2d4a6b365d03eb3087')})