FROM ubuntu:22.04
WORKDIR /tmp

# Install wget to be able to download OpenMPI
RUN apt update -y && \
    apt install -y wget vim build-essential
RUN apt install -y openmpi-bin openmpi-doc libopenmpi-dev openmpi-common

