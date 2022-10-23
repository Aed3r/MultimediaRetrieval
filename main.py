import database as db
import load_meshes as lm
import normalization as nrmlz
import sys
import ShapeDescriptors as sd
from tqdm import tqdm
import visualizer as vis
import time

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

    print(f"Database generated successfully. ({time.time() - start}s)")

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

    print(f"All meshes normalized and saved successfully. ({time.time() - start}s)")

    return meshes

def extract_features():
    global normalized_meshes, dbmngr

    print("Extracting and saving features from normalized meshes...")
    start = time.time()

    if normalized_meshes == []:
        normalized_meshes = lm.get_meshes(fromLPSB=False, fromPRIN=False, fromNORM=True, randomSample=-1, returnInfoOnly=False)
    
    res = sd.extract_all_features_from_meshes(normalized_meshes)

    dbmngr.update_all(res)

    print(f"Features extracted and saved successfully. ({time.time() - start}s)")

def main():
    global dbmngr

    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == "gen":
            gen_database()
        elif sys.argv[1].lower() == "normLPSB":
            save_normalize_meshes(fromLPSB=True)
        elif sys.argv[1].lower() == "normPRINC":
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
        elif sys.argv[1].lower() == "help":
            print("Available commands:")
            print("gen: Generates the database using the normalized meshes")
            print("normLPSB: Loads and normalizes all meshes from the Labeled PSB dataset and saves them to the 'data/normalized' folder")
            print("normPRINC: Loads and normalizes all meshes from the Princeton dataset and saves them to the 'data/normalized' folder")
            print("purge: Purges the database")
            print("count: Prints the number of meshes loaded in the database")
            print("countbycat: Prints the number of meshes with the given shape class")
            print("extract: Extracts features from the normalized meshes and saves them to the database")
            print("help: Prints this help message")
        else:
            print("Invalid argument")
    else:
        print("No argument provided")
        print("Use 'python main.py help' to see the available commands")

if __name__ == "__main__":
    main()