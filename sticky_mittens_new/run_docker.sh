#!/bin/bash
docker build . -t souradeep_docker_rl_image

docker run --name souradeep_docker_rl -it \
souradeep_docker_rl_image


docker cp souradeep_docker_rl:/home/Documents/code/models /home/duttaso/reinforcement_learning/sticky_mittens
docker cp souradeep_docker_rl:/home/Documents/code/call_logs /home/duttaso/reinforcement_learning/sticky_mittens
docker rm souradeep_docker_rl
