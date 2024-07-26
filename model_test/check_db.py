from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from bson import ObjectId
from openai_api_key import URI
import json

uri=URI

client = MongoClient(uri, server_api=ServerApi('1'))
db = client['intheview']

# ObjectId를 문자열로 변환하는 함수
def convert_objectid(data):
    if isinstance(data, list):
        return [convert_objectid(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_objectid(value) for key, value in data.items()}
    elif isinstance(data, ObjectId):
        return str(data)
    else:
        return data

# 컬렉션 목록 가져오기
collections = db.list_collection_names()

# 각 컬렉션의 내용을 보기 좋게 출력
for collection_name in collections:
    collection = db[collection_name]
    documents = collection.find()
    
    print(f"\nCollection: {collection_name}\n" + "-"*60)
    
    for document in documents:
        # ObjectId를 문자열로 변환
        document = convert_objectid(document)
        # JSON 형식으로 변환하고 들여쓰기
        document_str = json.dumps(document, indent=4, ensure_ascii=False)
        print(document_str)
        print("-"*60)