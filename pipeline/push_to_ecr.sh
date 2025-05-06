#!/bin/bash

# Set your algorithm name (ECR repo and image tag)
algorithm_name="bai-pet-v3"

# Make sure `serve` is executable (optional — only if you have it)
chmod +x inference/serve 2>/dev/null

# Get AWS account ID
account=$(aws sts get-caller-identity --query Account --output text)

# Get AWS region or default to us-east-1
region=$(aws configure get region)
region=${region:-us-east-1}

# Full ECR image URI
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"

# Create the ECR repository if it doesn't exist
aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "ECR repository ${algorithm_name} does not exist. Creating..."
    aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
fi

# Authenticate Docker with ECR
aws ecr get-login-password --region "${region}" | docker login --username AWS --password-stdin "${account}.dkr.ecr.${region}.amazonaws.com"

# Build, tag, and push the Docker image
echo "Building Docker image..."
docker build -t "${algorithm_name}" .

echo "Tagging image with ECR URI..."
docker tag "${algorithm_name}" "${fullname}"

echo "Pushing image to ECR..."
docker push "${fullname}"

echo "Done. Image pushed to: ${fullname}"
