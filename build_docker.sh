#!/bin/bash
DIRNAME=${PWD##*/}
IMAGE_NAME=${DIRNAME}_img
CONTAINER_NAME=${DIRNAME}_container

docker build -t $IMAGE_NAME . 

# -d OR --rm
# -d = detach = in background
# --rm = remove container afterwards
# it= interactive terminal
# --user = open container as current user
# -e = for colored terminal
# -v = volume
docker run \
--rm \
-it \
-v ${PWD}:'/code' \
--gpus all \
--name 'tmp' \
$IMAGE_NAME


# this works, but vscode requires
--user $(id -u):$(id -g) \

# this doesn't seem to work?!
-e "TERM=xterm-256color" \