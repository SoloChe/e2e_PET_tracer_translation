from sagemaker import Session
from sagemaker.model import Model
import boto3
import time

# Replace these values
role = "arn:aws:iam::<account-id>:role/SageMakerExecutionRole"
image_uri = "<account-id>.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:latest"
region = "us-west-2"

# Set up SageMaker session
boto_session = boto3.Session(region_name=region)
sagemaker_session = Session(boto_session=boto_session)

# Define the model
model = Model(
    image_uri=image_uri,
    role=role,
    sagemaker_session=sagemaker_session
)

# Deploy the model as a SageMaker endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large"
)

endpoint_name = predictor.endpoint_name
print(f"Endpoint deployed: {endpoint_name}")

print("🕒 Endpoint will be auto-deleted in 15 minutes...")
time.sleep(900)

print(f"Deleting endpoint: {endpoint_name}")
predictor.delete_endpoint()
print("Endpoint deleted.")
