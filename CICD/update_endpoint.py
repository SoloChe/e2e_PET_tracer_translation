import boto3
import os
from time import gmtime, strftime

region = os.environ["AWS_REGION"]
bucket = os.environ["S3_BUCKET"]
image_name = os.environ["IMAGE_NAME"]
endpoint_name = os.environ["ENDPOINT_NAME"]
role = os.environ.get("SAGEMAKER_EXECUTION_ROLE", "arn:aws:iam::123456789012:role/service-role/SageMakerRole")

account_id = boto3.client("sts").get_caller_identity()["Account"]
model_name = f"model-{strftime('%Y%m%d-%H%M%S', gmtime())}"
container_image = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{image_name}:latest"

s3_url = f"s3://{bucket}/{model_name}.tar.gz"

sm = boto3.client("sagemaker", region_name=region)

# Register the model
print(f"Registering model {model_name} with image {container_image}")
container = {
    "Image": container_image,
    "ModelDataUrl": s3_url
}
sm.create_model(ModelName=model_name, ExecutionRoleArn=role, PrimaryContainer=container)

# Create endpoint configuration
config_name = f"{endpoint_name}-config-{strftime('%Y%m%d-%H%M%S', gmtime())}"
print(f"🔧 Creating endpoint config {config_name}")
sm.create_endpoint_config(
    EndpointConfigName=config_name,
    ProductionVariants=[{
        "InstanceType": "ml.m5.large",
        "InitialVariantWeight": 1,
        "InitialInstanceCount": 1,
        "ModelName": model_name,
        "VariantName": "AllTraffic"
    }]
)

# Update the endpoint
print(f"Updating endpoint {endpoint_name}")
sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)

print(f"Endpoint {endpoint_name} updated with model {model_name}")
