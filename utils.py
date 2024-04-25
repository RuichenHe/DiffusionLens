import requests

from pymongo import MongoClient

def connect_db(collectionName):
    uri = "mongodb+srv://kyle666666:19941114hHaimeng!@gallery-info.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000"
    client = MongoClient(uri)

    db = client['gallery']
    collection = db[collectionName]
    return collection

def random_pick(collectionName, pick_size = 10):
    collection = connect_db(collectionName)
    random_documents = collection.aggregate([
        {"$sample": {"size": pick_size}}
    ])
    return random_documents

def set_text_index(collectionName, fieldName):
    uri = "mongodb+srv://kyle666666:19941114hHaimeng!@gallery-info.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000"
    client = MongoClient(uri)

    db = client['gallery']
    collection = db[collectionName]
    collection.create_index([(fieldName, 'text')])

def delete_collection(collectionName):
    uri = "mongodb+srv://kyle666666:19941114hHaimeng!@gallery-info.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000"
    client = MongoClient(uri)

    db = client['gallery']
    collection = db[collectionName]

    # collection.drop_index("$**_text")
    # collection.create_index([('name', 'text')])

    collection.drop()

def upload_json(document):
    uri = "mongodb+srv://kyle666666:19941114hHaimeng!@gallery-info.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000"
    client = MongoClient(uri)

    db = client['gallery']
    collection = db['test']

    # collection.drop_index("$**_text")
    # collection.create_index([('name', 'text')])

    collection.insert_one(document)

def find_json(query):
    uri = "mongodb+srv://kyle666666:19941114hHaimeng!@gallery-info.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000"
    client = MongoClient(uri)
    db = client['gallery']
    collection = db['test']
    indexes = collection.list_indexes()
    for index in indexes:
        print(index)
    results = collection.find({ '$text': { '$search': query } })
    count = collection.count_documents({ '$text': { '$search': query } })
    print(f"Number of documents matching the query: {count}")
    return results

def upload_image(file_path = None):
    url = 'http://localhost:7071/api/getuploadurl'
    # Data to send (example data)
    data = {
    }

    # Send a POST request with JSON data
    response = requests.post(url, json=data)
    upload_url = None
    # Check if the request was successful
    if response.status_code == 200:
        print('Success!')
        data = response.json()  # Parse JSON data from response
        upload_url = data['uploadURL']  # Access the 'uploadURL' item
    else:
        print('Failed to post data.', response.status_code)
        return

    # Form the file and POST using the uploadURL

    public_url = None
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f)}
        # Optionally add additional data or parameters if needed
        data = {
        }
        # Post the request
        response = requests.post(upload_url, files=files)

        if response.status_code == 200:
            data = response.json()  # Parse JSON data from response
            public_url = data['result']['variants'][0]  # Access the 'uploadURL' item
            print('Upload successful:', public_url)
        else:
            print('Failed to upload:', response.text)

    return public_url
