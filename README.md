# Multimedia Retrieval Assignment

## Installation & Setup
 - Install Python 3.9.13: https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe - Make sure to select 'Add Python to Path' in the installer!
 - Install mongodb: https://www.mongodb.com/docs/manual/installation/
 - Install all python modules: open a new terminal at the project root and run `pip3 install -r requirements.txt`
 - Run `python3.9 open3d_test.py` to check that open3d works
 - Build the basic database with `python3.9 database.py` to check that mongodb and pymongo work

## Datasets
 - The datasets should be unzipped into the data directory:
   - The Princeton shape benchmark (https://web.archive.org/web/20190323023058/http://shape.cs.princeton.edu/benchmark/download/psb_v1.zip) to ./data/psb_v1 such that ./data/psb_v1/benchmark exists
   - The Labeled PSB dataset (https://people.cs.umass.edu/~kalo/papers/LabelMeshes/labeledDb.7z) to ./data/LabeledDB_new such that ./data/LabeledDB_new/Airplane exists

## Modules
 - Run `python load_meshes.py` to load a sample file and visualize it. Run `python load_meshes.py <path>` to visualize the OFF/PLY file located at `<path>` 
   - Command Example: `python load_meshes.py data/LabeledDB_new/Airplane/61.off`
