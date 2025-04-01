import pymongo
from pymongo.errors import ConnectionFailure
import os
from dotenv import load_dotenv
import json
import datetime

# Load environment variables
load_dotenv()

# MongoDB connection string - defaulting to MongoDB Atlas
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "nba_strategy_app")

# Print MongoDB connection status
print(f"MongoDB URI configured: {bool(os.getenv('MONGO_URI'))}")

def get_client():
    """Get MongoDB client connection"""
    try:
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # Check if the connection is valid
        client.admin.command('ping')
        print("✅ Successfully connected to MongoDB Atlas")
        return client
    except ConnectionFailure:
        print("⚠️ MongoDB server not available. Using fallback data storage.")
        return None
    except Exception as e:
        print(f"❌ Error connecting to MongoDB: {e}")
        return None

def get_db():
    """Get database connection"""
    client = get_client()
    if client:
        return client[DB_NAME]
    return None

def close_connection(client):
    """Close MongoDB connection"""
    if client:
        client.close()

def get_players_collection():
    """Get players collection"""
    db = get_db()
    if db is not None:
        return db["players"]
    return None

def get_teams_collection():
    """Get teams collection"""
    db = get_db()
    if db is not None:
        return db["teams"]
    return None

def get_game_logs_collection():
    """Get game logs collection"""
    db = get_db()
    if db is not None:
        return db["game_logs"]
    return None

def get_predictions_collection():
    """Get predictions collection"""
    db = get_db()
    if db is not None:
        return db["predictions"]
    return None

def get_simulations_collection():
    """Get simulations collection"""
    db = get_db()
    if db is not None:
        return db["simulations"]
    return None

def get_cached_data_collection():
    """Get cached data collection"""
    db = get_db()
    if db is not None:
        return db["cached_data"]
    return None

def insert_document(collection, document):
    """Insert a document into a collection"""
    if collection is not None:
        try:
            result = collection.insert_one(document)
            return result.inserted_id
        except Exception as e:
            print(f"Error inserting document: {e}")
    return None

def insert_many_documents(collection, documents):
    """Insert multiple documents into a collection"""
    if collection is not None and documents:
        try:
            result = collection.insert_many(documents)
            return result.inserted_ids
        except Exception as e:
            print(f"Error inserting multiple documents: {e}")
    return None

def find_document(collection, query):
    """Find a document in a collection"""
    if collection is not None:
        try:
            return collection.find_one(query)
        except Exception as e:
            print(f"Error finding document: {e}")
    return None

def find_documents(collection, query=None, projection=None):
    """Find documents in a collection"""
    if collection is not None:
        try:
            return list(collection.find(query or {}, projection))
        except Exception as e:
            print(f"Error finding documents: {e}")
    return []

def update_document(collection, query, update):
    """Update a document in a collection"""
    if collection is not None:
        try:
            result = collection.update_one(query, {"$set": update})
            return result.modified_count
        except Exception as e:
            print(f"Error updating document: {e}")
    return 0

def delete_document(collection, query):
    """Delete a document from a collection"""
    if collection is not None:
        try:
            result = collection.delete_one(query)
            return result.deleted_count
        except Exception as e:
            print(f"Error deleting document: {e}")
    return 0

def cache_nba_data(data_type, season, data):
    """Cache NBA data in MongoDB
    
    Args:
        data_type (str): Type of data ('players', 'teams', etc.)
        season (str): NBA season (e.g., '2022-23')
        data (list): List of dictionaries containing the data
    """
    collection = get_cached_data_collection()
    
    if collection is None:
        # Use local file cache if MongoDB is not available
        try:
            cache_dir = os.path.join(os.getcwd(), '.cache')
            os.makedirs(cache_dir, exist_ok=True)
            
            filename = os.path.join(cache_dir, f"{data_type}_{season}.json")
            with open(filename, 'w') as f:
                # Add timestamp to the cache
                cache_data = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "data": data
                }
                json.dump(cache_data, f)
            return True
        except Exception as e:
            print(f"Error caching data to file: {e}")
            return False
    
    try:
        # Check if data already exists
        existing = collection.find_one({"data_type": data_type, "season": season})
        
        # If data exists, update it
        if existing:
            collection.update_one(
                {"data_type": data_type, "season": season},
                {"$set": {"data": data, "updated_at": datetime.datetime.now()}}
            )
        # Otherwise, insert new data
        else:
            collection.insert_one({
                "data_type": data_type,
                "season": season,
                "data": data,
                "created_at": datetime.datetime.now(),
                "updated_at": datetime.datetime.now()
            })
        return True
    except Exception as e:
        print(f"Error caching data to MongoDB: {e}")
        return False

def get_cached_nba_data(data_type, season):
    """Get cached NBA data from MongoDB
    
    Args:
        data_type (str): Type of data ('players', 'teams', etc.)
        season (str): NBA season (e.g., '2022-23')
        
    Returns:
        list: List of dictionaries containing the data
    """
    collection = get_cached_data_collection()
    
    if collection is None:
        # Use local file cache if MongoDB is not available
        try:
            cache_dir = os.path.join(os.getcwd(), '.cache')
            filename = os.path.join(cache_dir, f"{data_type}_{season}.json")
            
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    cache_data = json.load(f)
                    # Check if cache is expired (more than 24 hours old)
                    timestamp = datetime.datetime.fromisoformat(cache_data["timestamp"])
                    if (datetime.datetime.now() - timestamp).total_seconds() > 86400:
                        # Cache is expired
                        return None
                    return cache_data["data"]
            return None
        except Exception as e:
            print(f"Error loading data from file cache: {e}")
            return None
    
    try:
        cached = collection.find_one({"data_type": data_type, "season": season})
        if cached:
            # Check if cache is expired (more than 24 hours old)
            updated_at = cached.get("updated_at", cached.get("created_at"))
            if updated_at and (datetime.datetime.now() - updated_at).total_seconds() > 86400:
                # Cache is expired
                return None
            return cached["data"]
        return None
    except Exception as e:
        print(f"Error getting cached data from MongoDB: {e}")
        return None