ARG  TAG_VERSION
FROM tensorflow/tensorflow:${TAG_VERSION}
MAINTAINER kbogdan

# https://askubuntu.com/a/1013396
ARG DEBIAN_FRONTEND=noninteractive
ARG LOCAL

ENV NUM_CORES 8

WORKDIR /

# Install requirements for OpenCV and other packages
RUN apt-get -y update -qq && \
    apt-get -y install wget \
                       unzip \
                       # Required
                       build-essential \
                       cmake \
                       git \
                       pkg-config \
                       nano \
                       yasm \
                       vim \
                       default-jre  # Java (for meteor)


RUN pip install --upgrade pip setuptools

# Install requirements
COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt

# Install packages for language model # =================================
WORKDIR /

# RUN pip install scikit-learn
# RUN pip install nltk

# https://github.com/heryandi/docker-python3-nltk-gensim/blob/master/Dockerfile
# python -m nltk.downloader -d /usr/local/share/nltk_data all
RUN python3 -m nltk.downloader -d /usr/share/nltk_data punkt
RUN python3 -m nltk.downloader -d /usr/share/nltk_data words
# RUN pip install --upgrade gensim

WORKDIR /
# =======================================================================

# Cleaning temporary files and others
RUN apt-get autoclean autoremove &&\
    rm -rf /var/lib/apt/lists/* \
           /tmp/* \
           /var/tmp/*

ARG UID
ARG GID
ARG USER_NAME
ARG GROUP

RUN echo $UID
RUN echo $GID
RUN echo $USER_NAME
RUN echo $GROUP

RUN   groupadd --gid $GID $GROUP && \
      useradd --create-home --shell /bin/bash --uid $UID --gid $GID $USER_NAME && \
      adduser $USER_NAME sudo && \
      su -l $USER
USER $USER_NAME


CMD ["/bin/bash"]
