FROM ubuntu:latest
RUN apt-get update && apt-get install -y cmake build-essential git libx11-dev libxrandr-dev mesa-common-dev libgl1-mesa-dev libglu1-mesa-dev libxi-dev libxmu-dev libblas-dev libxinerama-dev libxcursor-dev x11-apps
#COPY . /usr/src/app
#WORKDIR /usr/src/app
#RUN mkdir -p build
#COPY ./data /usr/src/app/build
#WORKDIR /usr/src/app/build
#RUN cmake ..
#RUN make
#CMD ["code"]