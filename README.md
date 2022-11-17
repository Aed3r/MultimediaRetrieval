# Multimedia Retrieval Assignment

## Installation & Setup
 - Install Python 3.9.13 (not higher): https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe - Make sure to select 'Add Python to Path' in the installer!
 - Install mongodb: https://www.mongodb.com/docs/manual/installation/
 - Microsoft Visual C++ 14.0 or greater is required. Get https://visualstudio.microsoft.com/visual-cpp-build-tools/ and select the "Desktop development with C++" workload. Then for Individual Components, select only: "Windows 10 SDK" and "C++ x64/x86 build tools".
    - Or run `vs_buildtools.exe --norestart --passive --downloadThenInstall --includeRecommended --add Microsoft.VisualStudio.Workload.NativeDesktop --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Workload.MSBuildTools`
 - Install all python modules: open a new terminal at the project root and run `pip3 install -r requirements.txt`
    - You might also need to install 'windows-curses' if your operating system is Windows: `pip3 install windows-curses`
 - Run `python3.9 open3d_test.py` to check that open3d works
 - Build the basic database with `python3.9 database.py` to check that mongodb and pymongo work

## Datasets
 - The datasets should be unzipped into the data directory:
   - The Labeled PSB dataset (https://people.cs.umass.edu/~kalo/papers/LabelMeshes/labeledDb.7z) to ./data/LabeledDB_new such that ./data/LabeledDB_new/Airplane exists
   
## Import Database
 - The database can be downloaded here (https://drive.google.com/drive/folders/1pH79l-7HWPmV4fVzRe2rTZ4R7bE_aJN4?usp=sharing)

## Modules
 - Run `python load_meshes.py` to load a sample file and visualize it. Run `python load_meshes.py <path>` to visualize the OFF/PLY file located at `<path>` 
   - Command Example: `python load_meshes.py data/LabeledDB_new/Airplane/61.off`
   
## GUI
 - Run `python gui.py` to use the graphic user interface, you can load a mesh from files or from database.

## Run metrics
 - 
