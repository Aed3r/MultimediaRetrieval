# Multimedia Retrieval Assignment

## Installation & Setup
 - Install Python 3.9.13 (not higher): https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe - Make sure to select 'Add Python to Path' in the installer!
 - Install mongodb: https://www.mongodb.com/docs/manual/installation/
 - Microsoft Visual C++ 14.0 or greater is required. Get https://visualstudio.microsoft.com/visual-cpp-build-tools/ and select the "Desktop development with C++" workload. Then for Individual Components, select only: "Windows 10 SDK" and "C++ x64/x86 build tools".
    - Or run `vs_buildtools.exe --norestart --passive --downloadThenInstall --includeRecommended --add Microsoft.VisualStudio.Workload.NativeDesktop --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Workload.MSBuildTools`
 - Install all python modules: open a new terminal at the project root and run `pip3 install -r requirements.txt`
    - You might also need to install 'windows-curses' if your operating system is Windows: `pip3 install windows-curses`
 - Run `python3.9 open3d_test.py` to check that open3d works
 - Build the basic database with `python3.9 main.py importDB data/db.json` to check that mongodb and pymongo work

## Datasets
 - The datasets should be unzipped into the data directory:
   - The Labeled PSB dataset (https://people.cs.umass.edu/~kalo/papers/LabelMeshes/labeledDb.7z) to ./data/LabeledDB_new such that ./data/LabeledDB_new/Airplane exists
   - (optionally) The Princeton shape benchmark (https://web.archive.org/web/20190323023058/http://shape.cs.princeton.edu/benchmark/download/psb_v1.zip) to ./data/psb_v1 such that ./data/psb_v1/benchmark exists
   
## Operations
 - The main.py module controls most other modules. Run `python main.py help` to see all available commands. The most important ones are:
   - `python main.py gen`: Normalizes and generates the database using all the meshes in the Labeled PSB dataset. Extracts features, generates thumbnails and creates the ANN index. This takes a while, importing an existing database with `python3.9 main.py importDB data/db.json` is recommended.
   - `python main.py genTTSimple`: Generates the truth tables for the simple CBSR.
   - `python main.py genTTANN`: Generates the truth tables for the ANN CBSR.
   - `python main.py qualMetrics`: Runs the quality metrics for the currently generated truth tables.
 - Run `python visualizer.py [<path>]` to open the visualizer with the mesh located at `<path>`. Without indicating a path, a random mesh is loaded.
 - Run `python gui.py` to use the graphic user interface, you can load a mesh from files or from database.
