from curses.ascii import BS
from pymongo import MongoClient
from pymongo.errors import CollectionInvalid
import pymongo
from collections import OrderedDict
import json
import os
import load_meshes as lm
from tqdm import tqdm

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
        
        # Check if current the schema is valid
        if not self._validateSchema():
            # Drop the collection
            self._db.drop_collection('meshes')
            # Create the collection
            self._db.create_collection('meshes', validator=self._validationSchema)
            self._schemaValidated = True
        return

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

        if duplicatesCount == 1:
            print('1 duplicate mesh found and ignored')
        elif duplicatesCount > 1:
            print(f'{duplicatesCount} duplicate meshes found and ignored')
        
        # Remove the unique index
        self._db.meshes.drop_index("load_index")

    # Return the number of meshes loaded in the db
    def get_mesh_count(self):
        return self._db.meshes.count_documents({})

    # Return the number of meshes with the given shape class
    def get_mesh_count_by_category(self, shapeClass):
        return self._db.meshes.count_documents({'class': shapeClass})

    # Purge the database
    def purge(self):
        self._db.drop_collection('meshes')
        self.create_collection()


def main():
    import load_meshes

    dbmngr = DatabaseManager()

    # Load the data
    data = load_meshes.get_meshes(fromLPSB=True, fromPRIN=False, randomSample=-1, returnInfoOnly=True)

    # Insert the data into the db
    dbmngr.insert_data(data)

    # Print the number of meshes loaded in the db
    print(f'{dbmngr.get_mesh_count()} meshes loaded in the db')

    # Print the number of meshes with the given shape class
    print(f'{dbmngr.get_mesh_count_by_category("Airplane")} meshes with shape class \'Airplane\' loaded in the db')


if __name__ == "__main__":
    main()

