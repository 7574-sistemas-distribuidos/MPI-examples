FROM centos:latest
WORKDIR /tmp

# Install wget to be able to download OpenMPI
RUN yum update -y && \
    yum groupinstall "Development Tools" -y && \
    yum install wget vim -y

# Compile and install OpenMPI
RUN wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.gz && \
    tar -xvzf openmpi-3.0.0.tar.gz &&\
    cd openmpi-3.0.0 && \
    ./configure --prefix=/opt/openmpi-3.0.0 && \
    make -j4 && make install && \
    cd /tmp && rm -rf /tmp/openmpi-3.0.0*

COPY env /opt/env
