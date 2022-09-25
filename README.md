# Multimedia Retrieval Assignment

## Installation & Setup
 - Run `wsl --install` in a Administrator Terminal (right click on Powershell to 'Run as Administrator') to install WSL 2
 - Install Docker: https://www.docker.com/products/docker-desktop/ and run Docker Desktop
 - Install vscode: https://code.visualstudio.com/
 - Install the Remote Development extension: https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack
 - Install VcXsrv Windows X Server: https://sourceforge.net/projects/vcxsrv/files/latest/download
 - Inside vscode, hit F1 and run the command `Remote-Containers: Open Folder in Container..` and select the "MultimediaRetrieval" folder in the selection screen
 - The container should start up, click on the message in the bottom left corner to see the progression and potential errors
 - In the meantime, run XLaunch, select "One large window", and go through all steps by leaving everything default
 - A big black window should appear, you can reduce its size
 - When the container is done building in vscode, open a terminal if there isn't one open already: 'Terminal' > 'New Terminal' at the top of the screen
 - Run following command: `mkdir -p build; cd build`
 - To generate the project do: `cmake ..`
 - To build the project do: `make`
 - To run the project do: `./main`
 - If everything worked correctly, the XLaunch window should show the libigl program
 - After any changes to the files re-run `make` to see the changes. After any project structure changes (e.g. new files), re-run `cmake ..`
 - Git should work as expected in the container, and all vscode extensions copy over. Here are a few convenient ones for C++ development:
    - https://marketplace.visualstudio.com/items?itemName=jeff-hykin.better-cpp-syntax
    - https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools
    - https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools-extension-pack
    - https://marketplace.visualstudio.com/items?itemName=tonybaloney.vscode-pets

## Datasets
 - The datasets should be unzipped into the data directory:
   - The Princeton shape benchmark (https://web.archive.org/web/20190323023058/http://shape.cs.princeton.edu/benchmark/download/psb_v1.zip) to ./data/psb_v1 such that ./data/psb_v1/benchmark exists
   - The Labeled PSB dataset (https://people.cs.umass.edu/~kalo/papers/LabelMeshes/labeledDb.7z) to ./data/LabeledDB_new such that ./data/LabeledDB_new/Airplane exists
