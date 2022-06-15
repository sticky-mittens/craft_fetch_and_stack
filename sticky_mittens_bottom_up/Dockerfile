FROM ubuntu:bionic

# Adding this Repo to locate jasper
RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get install -y software-properties-common
RUN apt-get update
RUN add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
RUN apt update

RUN apt-get update && apt-get install -y build-essential git cmake autoconf libtool pkg-config libssl-dev
RUN apt-get install -y cmake git libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev
RUN apt-get install -y python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
RUN apt-get install -y wget


#Install CUDA
RUN apt-get install -y linux-headers-$(uname -r)
WORKDIR /home/Documents/
RUN mkdir cuda_installation
WORKDIR /home/Documents/cuda_installation/

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
RUN mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600

RUN wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
RUN apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install cuda
RUN apt-get -y install cuda-drivers

ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda-10.2/bin${PATH:+:${PATH}}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-10.2/lib64 ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
RUN ldconfig

RUN echo $PATH
RUN echo $LD_LIBRARY_PATH
RUN echo $CUDA_HOME

RUN /usr/local/cuda/bin/nvcc --version
WORKDIR  /usr/local/cuda/samples/1_Utilities/deviceQuery
RUN make
RUN ./deviceQuery


#Installing Python
RUN apt-get -y install python3.6 python3.6-dev
RUN apt-get -y install python3-pip
RUN pip3 install numpy scipy Pillow
RUN pip3 install google-cloud-translate
RUN pip3 install -U scikit-image
RUN pip3 install seaborn
RUN pip3 install torch torchvision torchaudio

WORKDIR /home/Documents/
RUN mkdir code
WORKDIR /home/Documents/code/
COPY ./ ./
#CMD ["python3.6 play_recursive_craft.py"]
