import boto3
import pandas as pd
import io
import os

endpoint_name = os.environ["ENDPOINT_NAME"]
runtime_sm_client = boto3.client("sagemaker-runtime")

# Output path for prediction results
output = "./data_PET_Test/Synthetic_PiB.csv"

# Read the input CSV file
with open("./data_PET_Test/paired_FBP_SUVR.csv", "r") as f:
    csv_data = f.read()

# Send request to SageMaker endpoint
response = runtime_sm_client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="text/csv",
    Body=csv_data
)

# Read and decode the response
result = response["Body"].read().decode("utf-8")

# Handle response
if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
    print("Predictions received successfully.")
    with open(output, "w") as f:
        f.write(result)
    print(f"Predictions saved to {output}")
else:
    print("Error during prediction:")
    print(result)
