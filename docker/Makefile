
# Dockerfile to be used
GPU_DOCKER_FILE=salstm.Dockerfile

# Docker image variables
SVC_GPU=salstm-tensorflow-py3
VERSION=1.13.2_latest
REGISTRY_URL=kbogdan

TAG_VERSION_TF_GPU=1.13.2-gpu-py3

# Volumes
# Since the makefile is generally inside a 'docker' folder, we get
# the parent of current dir
MY_INSIDE_VOLUME_GPU=/work

# Container related
NAME_CONTAINER=salstm

# Check the docker version to correctly set the gpu command
docker_version=$(shell docker version --format '{{.Server.APIVersion}}')
ifeq ($(shell expr $(docker_version) \>= 1.40), 1)
    docker_gpu=--gpus all
else
    docker_gpu=--runtime=nvidia
endif


build-image:
	@echo "[build] Building gpu docker image..."
	@echo `id -ng ${USER}`
	@docker build \
	    --build-arg TAG_VERSION=$(TAG_VERSION_TF_GPU) \
	    --build-arg UID=`id -u` \
	    --build-arg GID=`id -g` \
	    --build-arg USER_NAME=${USER} \
	    --build-arg GROUP=`id -ng ${USER}` \
	    -t $(REGISTRY_URL)/$(SVC_GPU):$(VERSION) -f $(GPU_DOCKER_FILE) .
	@echo "\n[build] Delete old image versions..."
	@docker images|sed "1 d"|grep "<none> *<none>"|awk '{print $$3}'|sort|uniq|xargs docker rmi -f
run-container:
	@echo "[run] Running container for a gpu-based image..."
	@docker run -it $(docker_gpu) \
	    -p 6010:6006 -p 8890:8888 \
	    -v ${shell cd ../ && pwd}:$(MY_INSIDE_VOLUME_GPU) \
	    --name $(NAME_CONTAINER) $(REGISTRY_URL)/$(SVC_GPU):$(VERSION) /bin/bash
clean:
	@echo "[clean] Cleaning docker images..."
	@docker rmi -f $(REGISTRY_URL)/$(SVC_GPU):$(VERSION)
