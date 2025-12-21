#!/bin/bash
# Log everything to start_docker.log
exec > /home/ubuntu/start_docker.log 2>&1

echo "Logging in to ECR..."
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 794431322868.dkr.ecr.ap-south-1.amazonaws.com

echo "Pulling Docker image..."
docker pull 794431322868.dkr.ecr.ap-south-1.amazonaws.com/satya-ecr:latest

echo "Checking for existing container..."
if [ "$(docker ps -q -f name=satya-app)" ]; then
    echo "Stopping existing container..."
    docker stop satya-app
fi

if [ "$(docker ps -aq -f name=satya-app)" ]; then
    echo "Removing existing container..."
    docker rm satya-app
fi

echo "Starting new container..."
docker run -d -p 80:5000 --name satya-app -e AWS_ACCESS_KEY_ID="replace_me" -e AWS_SECRET_ACCESS_KEY="replace_me" 794431322868.dkr.ecr.ap-south-1.amazonaws.com/satya-ecr:latest

echo "Container started successfully."
