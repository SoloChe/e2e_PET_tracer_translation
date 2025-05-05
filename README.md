# End-to-end PET Tracer Translation (FBP -> PiB)

1. Create a repo in ECR: `aws ecr create-repository --repository-name pytorch-inference`
2. Authenticate Docker to ECR: `aws ecr get-login-password --region us-west-2 
| docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-west-2.amazonaws.com`
3. Build and tag your Docker image: `docker build -t pytorch-inference`
4. Tag it so it points to the ECR repository: `docker tag pytorch-inference:latest <account-id>.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:latest`
1.  Push the image to ECR: `docker push <account-id>.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:latest`
`
