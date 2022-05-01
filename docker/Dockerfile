FROM tensorflow/tensorflow:latest-gpu

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei
COPY ./requirements.txt ./requirements.txt 

RUN /usr/bin/python3 -m pip install --upgrade pip

# public GPG key error
RUN apt-get install wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"

RUN apt-get update

RUN apt-get install vim -y
RUN pip3 install -r requirements.txt
RUN pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install tensorflow==2.8.0
RUN pip3 install tf-agents==0.12.0

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-get install curl -y
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt update
RUN apt-get install ros-noetic-desktop-full -y
RUN source /opt/ros/noetic/setup.bash
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN source ~/.bashrc
RUN apt-get update

RUN apt-get install ros-noetic-soem -y --fix-missing
RUN apt-get install ros-noetic-ddynamic-reconfigure -y --fix-missing
RUN pip3 install rospkg
RUN pip3 install pyyaml
