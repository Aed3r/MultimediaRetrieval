import time
from pymongo import MongoClient
from pymongo.errors import CollectionInvalid
import pymongo
from collections import OrderedDict
import json
from tqdm import tqdm
from bson.json_util import dumps

class DatabaseManager:
    def __init__(self):
        self._db = MongoClient("mongodb://localhost:27017/")['mmr']
        self._schemaValidated = False # Does the validation schema of the db correspond to the one in db_schema.json?
        self._validationSchema = None # The validation schema of the db
        #self._basicDataLoaded = False # Is the basic data loaded into the db?
        
        # Load the validation schema
        self._load_validation_schema()

        # Create the collection
        self.create_collection()
        
    # Load the validation schema from db_schema.json
    def _load_validation_schema(self):
        try:
            with open('db_schema.json', 'r') as j:
                self._validationSchema = json.load(j)
        except FileNotFoundError:
            print('db_schema.json not found')
            exit(1)

    # Validate the schema of the collection with the one in db_schema.json
    def _validateSchema(self):
        if self._schemaValidated:
            return True

        # Get the schema of the db
        current_schema = self._db.get_collection('meshes').options().get('validator')

        # Compare the two schemas
        if self._validationSchema == current_schema:
            self._schemaValidated = True
            return True
        else:
            return False

    # Create the collection with the validation schema if it doesn't exist
    def create_collection(self):
        # Check if the collection already exists
        if not 'meshes' in self._db.list_collection_names():
            # Create the collection
            self._db.create_collection('meshes', validator=self._validationSchema)
            self._schemaValidated = True
            return
        
        self.update_validation_schema()

    def update_validation_schema(self):
        # Check if current the schema is valid
        if not self._validateSchema():
            print("New database validation schema detected. (Re-)creating collection...")

            meshesCursor = self.get_all()
            meshes = []
            for mesh in meshesCursor:
                meshes.append(mesh)

            # Drop the collection
            self._db.drop_collection('meshes')

            # Create the collection
            self._db.create_collection('meshes', validator=self._validationSchema)
            self._schemaValidated = True

            # Re-insert the data
            self.insert_data(meshes)

    # Load the given data into the db
    def insert_data(self, data):
        duplicatesCount = 0

        # Set a unique index on the path field
        self._db.meshes.create_index([('path', pymongo.ASCENDING)], name="load_index", unique=True)

        # Insert the data
        for d in tqdm(data, desc="Inserting meshes into database", ncols=150):
            try:
                self._db.meshes.insert_one(d)
            except pymongo.errors.DuplicateKeyError:
                duplicatesCount += 1
            except:
                print("Error inserting mesh: {}".format(d['path']))

        if duplicatesCount == 1:
            print('1 duplicate mesh found and ignored')
        elif duplicatesCount > 1:
            print(f'{duplicatesCount} duplicate meshes found and ignored')
        
        # Remove the unique index
        self._db.meshes.drop_index("load_index")

    # Add the given data to an existing item in the db
    # The data must be a dictionary and contain the path field
    def update_one(self, data):
        for key in data:
            if key != 'path' and key != 'vertices' and key != 'faces':
                self._db.meshes.update_one({'path': data['path']}, {'$set': {key: data[key]}})

    # Update the meshes with the given data
    # The data must be a list of dictionaries which all contain the path field
    def update_all(self, data):
        for d in tqdm(data, desc="Saving features into database", ncols=150):
            self.update_one(d)

    # Return the number of meshes loaded in the db
    def get_mesh_count(self):
        return self._db.meshes.count_documents({})

    # Return the number of meshes with the given shape class
    def get_mesh_count_by_category(self, shapeClass):
        return self._db.meshes.count_documents({'class': shapeClass})

    # Return the number of meshes with extracted features
    def get_mesh_count_with_features(self):
        return self._db.meshes.count_documents({'D4': {'$exists': True}})

    # Purge the database
    def purge(self):
        print("Purging database...")
        start = time.time()
        self._db.drop_collection('meshes')
        self.create_collection()
        print("Database purged successfully. (Took {:.2f} seconds)".format(time.time() - start))

    # Return all mesh infos saved in the database
    def get_all(self):
        return self._db.meshes.find({})

    # Return the mesh with the given path
    def get_by_path(self, path):
        return self._db.meshes.find({'path': path})[0]

    def get_all_by_category(self, shapeClass):
        return self._db.meshes.find({'class': shapeClass})
    
    # Return all mesh infos saved in the database with extracted features
    def get_all_with_extracted_features(self):
        return self._db.meshes.find({'D4': {'$exists': True}})

    def get_x_with_extracted_features(self, x):
        return self._db.meshes.find({'D4': {'$exists': True}}).limit(x)

    def query(self, query, filter={}, limit=None):
        if limit is None:
            return self._db.meshes.find(query, filter)
        else:
            return self._db.meshes.find(query, filter).limit(limit)

    # Returns a cursor to find all paths and names of the meshes
    def get_all_paths(self):
        return self._db.meshes.find({}, {'path': True, 'name': True, '_id': False})

    # Runs the given function on all meshes in the db
    def for_each(self, func):
        for mesh in self.get_all():
            self.update_one(func(mesh))

    def export_db(self, path):
        print("Exporting database to {}...".format(path))
        start = time.time()
        meshes = self._db.meshes.find({}, {'_id': False, 'vertices': False, 'faces': False})
        with open(path, 'w') as f:
            f.write('[')
            for mesh in tqdm(meshes, desc="Exporting database", ncols=150, total=self.get_mesh_count()):
                f.write(dumps(mesh))
                if meshes.alive:
                    f.write(',')
            f.write(']')
        print("Database exported successfully. (Took {:.2f} seconds)".format(time.time() - start))

    def import_db(self, path):
        print("Importing database from {}...".format(path))
        start = time.time()
        with open(path, 'r') as f:
            data = json.load(f)
        self.insert_data(data)
        print("Database imported successfully. (Took {:.2f} seconds)".format(time.time() - start))

    def unset_fields(self):
        self._db.meshes.update({}, {"$unset": {"vertices": 1}} , {"multi": True})

def main():
    import load_meshes

    dbmngr = DatabaseManager()

    # Load the data
    #data = load_meshes.get_meshes(fromLPSB=True, fromPRIN=False, fromNORM=False, randomSample=-1, returnInfoOnly=True)

    # Insert the data into the db
    #dbmngr.insert_data(data)

    # Print the number of meshes loaded in the db
    #print(f'{dbmngr.get_mesh_count()} meshes loaded in the db')

    # Print the number of meshes with the given shape class
    #print(f'{dbmngr.get_mesh_count_by_category("Airplane")} meshes with shape class \'Airplane\' loaded in the db')

    #paths = dbmngr.get_all_paths()
    print("All done")


if __name__ == "__main__":
    main()

