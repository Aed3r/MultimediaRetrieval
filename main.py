import database as db
import load_meshes as lm
import normalization as nrmlz
import sys
import ShapeDescriptors as sd
import visualizer as vis
import time
import numpy as np
import util

dbmngr = db.DatabaseManager()
normalized_meshes = []

# Loads all the meshes from the 'meshes' folder and saves them to the db
def gen_database():
    global normalized_meshes, dbmngr

    print("Generating database from normalized meshes...")
    start = time.time()

    # Extract features from the meshes
    if normalized_meshes == []:
        normalized_meshes = lm.get_meshes(fromLPSB=False, fromPRIN=False, fromNORM=True, randomSample=-1, returnInfoOnly=False)

    # Insert the meshes into the db
    dbmngr.insert_data(normalized_meshes)

    # Print the number of meshes loaded in the db
    print(f'{dbmngr.get_mesh_count()} meshes loaded in the db')

    print(f"Database generated successfully. ({round(time.time() - start)}s)")

def save_normalize_meshes(fromLPSB=False, fromPRIN=False):
    global normalized_meshes
    
    if fromLPSB and fromPRIN:
        print("Normalizing and saving all Labeled PSB and Princeton meshes...")
    elif fromLPSB:
        print("Normalizing and saving all Labeled PSB meshes...")
    elif fromPRIN:
        print("Normalizing and saving all Labeled PSB meshes...")
    start = time.time()

    meshes = lm.get_meshes(fromLPSB, fromPRIN, fromNORM=False, randomSample=-1, returnInfoOnly=False)
    meshes = nrmlz.normalize_all(meshes)
    lm.save_all_meshes(meshes, lm.NORMALIZEDMESHESFOLDER)
    normalized_meshes.append(meshes)

    print(f"All meshes normalized and saved successfully. ({round(time.time() - start)}s)")

    return meshes

def extract_features():
    global normalized_meshes, dbmngr

    print("Extracting and saving features from normalized meshes...")
    start = time.time()

    if normalized_meshes == []:
        normalized_meshes = lm.get_meshes(fromLPSB=False, fromPRIN=False, fromNORM=True, randomSample=-1, returnInfoOnly=False)
    
    res = sd.extract_all_features_from_meshes(normalized_meshes)

    dbmngr.update_all(res)

    print(f"Features extracted and saved successfully. ({round(time.time() - start)}s)")

def gen_feature_plots():
    meshes = lm.get_meshes(fromLPSB=False, fromPRIN=False, fromNORM=True, randomSample=-1, returnInfoOnly=False)
    sd.genFeaturePlots(meshes)

def gen_thumbnails():
    global dbmngr

    print("Generating thumbnails...")
    start = time.time()

    meshes = dbmngr.get_all()
    outputPaths = vis.gen_thumbnails(meshes)

    dbmngr.update_all(outputPaths)

    print(f"Thumbnails generated successfully. ({round(time.time() - start)}s)")

def update_db_schema():
    global dbmngr

    print("Updating db schema...")
    start = time.time()

    dbmngr.update_validation_schema()

    print(f"Db schema updated successfully. ({round(time.time() - start)}s)")


def standardize_db():
    global dbmngr

    print("Standardizing db...")
    start = time.time()

    meshes = dbmngr.get_all_with_extracted_features()
    meshes = util.standardize_all(meshes)
    dbmngr.update_all(meshes)

    print(f"Db standardized successfully. ({round(time.time() - start)}s)")

def main():
    global dbmngr

    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "gen" or sys.argv[1] == "genlpsb":
            save_normalize_meshes(fromLPSB=True)
            gen_database()
            gen_thumbnails()
            extract_features()
            standardize_db()
        elif sys.argv[1] == "genprin":
            save_normalize_meshes(fromPRIN=True)
            gen_database()
            gen_thumbnails()
            extract_features()
            standardize_db()
        elif sys.argv[1].lower() == "gendb":
            gen_database()
        elif sys.argv[1].lower() == "normlpsb":
            save_normalize_meshes(fromLPSB=True)
        elif sys.argv[1].lower() == "normprinc":
            save_normalize_meshes(fromPRIN=True)
        elif sys.argv[1].lower() == "norm":
            save_normalize_meshes(fromLPSB=True, fromPRIN=True)
        elif sys.argv[1].lower() == "purge":
            dbmngr.purge()
        elif sys.argv[1].lower() == "count":
            print(f'{dbmngr.get_mesh_count()} meshes loaded in the db')
        elif sys.argv[1].lower() == "countbycat":
            print(f'{dbmngr.get_mesh_count_by_category(sys.argv[2])} meshes with the category {sys.argv[2]}')
        elif sys.argv[1].lower() == "extract":
            extract_features()
        elif sys.argv[1].lower() == "genfeatureplots":
            gen_feature_plots()
        elif sys.argv[1].lower() == "genthumbnails":
            gen_thumbnails()
        elif sys.argv[1].lower() == "updatedbschema":
            update_db_schema()
        elif sys.argv[1].lower() == "standardizedb":
            standardize_db()
        elif sys.argv[1].lower() == "help":
            print("Available commands:")
            print("gen/genLPSB: Normalizes and generates the database using all the meshes in the Labeled PSB dataset. Extracts features and generates thumbnails")
            print("genPrin: Normalizes and generates the database using all the meshes in the Princeton dataset. Extracts features and generates thumbnails")
            print("genDB: Generates the database using the normalized meshes")
            print("normLPSB: Loads and normalizes all meshes from the Labeled PSB dataset and saves them to the 'data/normalized' folder")
            print("normPRINC: Loads and normalizes all meshes from the Princeton dataset and saves them to the 'data/normalized' folder")
            print("purge: Purges the database")
            print("count: Prints the number of meshes loaded in the database")
            print("countbycat: Prints the number of meshes with the given shape class")
            print("extract: Extracts features from the normalized meshes and saves them to the database")
            print("genFeaturePlots: Generates plots of the extracted features")
            print("genThumbnails: Generates thumbnails of the normalized meshes")
            print("updateDBSchema: Updates the database schema")
            print("standardizeDB: Standardizes the single features of all the meshes in the database")
            print("help: Prints this help message")
        else:
            print("Invalid argument")
    else:
        print("No argument provided")
        print("Use 'python main.py help' to see the available commands")

if __name__ == "__main__":
    main()