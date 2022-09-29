from curses.ascii import BS
from pymongo import MongoClient
from pymongo.errors import CollectionInvalid
from collections import OrderedDict
import json
import os
import load_meshes as lm

LABELED_PSB_DB = "data/LabeledDB_new"
PRINCETON_DB = "data/psb_v1"

class DatabaseManager:
    def __init__(self):
        self.db = MongoClient("mongodb://localhost:27017/")['mmr']
        self._schemaValidated = False # Does the validation schema of the db correspond to the one in db_schema.json?
        self.validationSchema = None # The validation schema of the db
        self._basicDataLoaded = False # Is the basic data loaded into the db?
        
        # Load the validation schema
        self.loadValidationSchema()

        # Create the collection
        self.createCollection()

        # Load the basic data
        self.load_labeled_psb()
        
    # Load the validation schema from db_schema.json
    def loadValidationSchema(self):
        try:
            with open('db_schema.json', 'r') as j:
                self.validationSchema = json.load(j)
        except FileNotFoundError:
            print('db_schema.json not found')
            exit(1)

    # Validate the schema of the collection with the one in db_schema.json
    def validateSchema(self):
        if self._schemaValidated:
            return True

        # Get the schema of the db
        current_schema = self.db.get_collection('meshes').options().get('validator')

        # Compare the two schemas
        if self.validationSchema == current_schema:
            self._schemaValidated = True
            return True
        else:
            return False

    # Create the collection with the validation schema if it doesn't exist
    def createCollection(self):
        # Check if the collection already exists
        if not 'meshes' in self.db.list_collection_names():
            # Create the collection
            self.db.create_collection('meshes', validator=self.validationSchema)
            self._schemaValidated = True
            return
        
        # Check if current the schema is valid
        if not self.validateSchema():
            # Drop the collection
            self.db.drop_collection('meshes')
            # Create the collection
            self.db.create_collection('meshes', validator=self.validationSchema)
            self._schemaValidated = True
        return

    def load_labeled_psb(self):
        if self._basicDataLoaded:
            return

        # Find all directories in the labeled psb database
        directories = [d for d in os.listdir(LABELED_PSB_DB) if os.path.isdir(os.path.join(LABELED_PSB_DB, d))]
        
        # Load the data from each directory
        for d in directories:
            # Iterate over all files in the directory
            for f in os.listdir(os.path.join(LABELED_PSB_DB, d)):
                # Check if the file is a .ply  or .off file
                if f[-4:] == '.ply':
                    # Load the data from the file
                    num_verts, num_faces, face_type = lm.load_PLY(os.path.join(LABELED_PSB_DB, d, f), returnInfoOnly=True)
                elif f[-4:] == '.off':
                    # Load the data from the file
                    num_verts, num_faces, face_type = lm.load_OFF(os.path.join(LABELED_PSB_DB, d, f), returnInfoOnly=True)
                else:
                    continue
                
                # Aggregate the data
                data = OrderedDict()
                data['name'] = f
                data['path'] = os.path.join(LABELED_PSB_DB, d, f)
                data['class'] = d
                data['num_verts'] = num_verts
                data['num_faces'] = num_faces
                data['face_type'] = face_type

                # Insert the data into the db
                #self.db.meshes.insert_one(data)
                print(str(data['path']) + ',' + str(data['num_verts']) + ',' + str(data['num_faces']) + ',' + str(data['face_type']))
        
        self._basicDataLoaded = True

def main():
    dbmngr = DatabaseManager()


if __name__ == "__main__":
    main()

