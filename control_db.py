from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from bson import ObjectId
from openai_api_key import URI
import json

uri = URI

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

client = MongoClient(uri, server_api=ServerApi('1'))

db = client['intheview']



while True:
    case_num = input("just check : 1\ndelete : 2\n: ")
    if case_num == "1":
        # 컬렉션 목록 가져오기
        collections = db.list_collection_names()

        for collection_name in collections:
            collection = db[collection_name]
            documents = collection.find()

            print(f"\nCollection: {collection_name}\n" + "-"*60)
            for document in documents:
                document = convert_objectid(document)
                document_str = json.dumps(document, indent=4, ensure_ascii=False)
                print(document_str)
                print("-"*60)
    elif case_num == "2":
        coll = input("which collection : ")
        collection = db[coll]

        while True:
            collections = db.list_collection_names()

            for collection_name in collections:
                collection = db[collection_name]
                documents = collection.find()

            print(f"\nCollection: {collection_name}\n" + "-"*60)
            for document in documents:
                document = convert_objectid(document)
                document_str = json.dumps(document, indent=4, ensure_ascii=False)
                print(document_str)
                print("-"*60)
            oid = input("Object Id to remove : ")
            if oid == 1:
                break
            if coll == "sessions":
                db['users'].update_many(
                    {},
                    {"$pull": {"sessions": ObjectId(oid)}}
                )
            collection.delete_one({"_id": ObjectId(oid)})
            # 컬렉션 목록 가져오기
    elif case_num=="0":
        break
    else:
        print("Invalid input. Please enter 1 or 2.")
