# NVIDIA CUDA image as base
FROM nvidia/cuda:11.1-base

# Execute RUN and CMD commands from here
WORKDIR "/home/dev"

# Update ubuntu and install desired packages (using as few RUN commands as possible)
RUN apt-get -q update && apt-get -q install -y wget && apt-get -y install gcc

# Download Miniconda, install, remove uneeded files, then update
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh --quiet --output-document miniconda.sh
RUN bash miniconda.sh -b
RUN rm miniconda.sh 

# Set path to locate miniconda
ENV PATH $PATH:/root/miniconda3/bin

# Update conda and initialise shell
RUN conda update conda && conda update --all
RUN conda init bash

# Get the dependencies file (can't interact with volumes during
# build process) and use it to make an environment
COPY env.yml .
RUN conda env create --file env.yml
# Remove it once used, we'll get a copy when the volumes mount
RUN rm env.yml