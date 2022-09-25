FROM ubuntu:latest
RUN apt-get update && apt-get install -y python3 wget libssl-dev libsasl2-dev cmake build-essential git libx11-dev libxrandr-dev mesa-common-dev libgl1-mesa-dev libglu1-mesa-dev libxi-dev libxmu-dev libblas-dev libxinerama-dev libxcursor-dev x11-apps
#WORKDIR /usr/src/app/
#RUN wget https://github.com/mongodb/mongo-c-driver/releases/download/1.23.0/mongo-c-driver-1.23.0.tar.gz
#RUN wget https://github.com/mongodb/mongo-cxx-driver/releases/download/r3.6.7/mongo-cxx-driver-r3.6.7.tar.gz
#RUN tar -xzf mongo-c-driver-1.23.0.tar.gz
#RUN tar -xzf mongo-cxx-driver-r3.6.7.tar.gz
#WORKDIR /usr/src/app/mongo-c-driver-1.23.0/cmake-build
#RUN cmake -DENABLE_AUTOMATIC_INIT_AND_CLEANUP=OFF ..
#RUN cmake --build .
#RUN cmake --build . --target install
#WORKDIR /usr/src/app/mongo-cxx-driver-r3.6.7/build
#RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DBSONCXX_POLY_USE_MNMLSTC=1
#RUN cmake --build . --target EP_mnmlstc_core
#RUN cmake --build .
#RUN cmake --build . --target install
#WORKDIR /usr/src/app/