# Local Testing
import requests

# Define the URL
url = "http://localhost:8080/invocations"

output = "./Synthetic_PiB.csv"

# Read the CSV file
with open("./paired_FBP_SUVR.csv", "r") as f:
    csv_data = f.read()

# Send the request
response = requests.post(
    url,
    headers={"Content-Type": "text/csv"},
    data=csv_data
)

# Print the response
if response.status_code == 200:
    print("Predictions received successfully.")
    # Save the response CSV to a file
    with open(output, "w") as f:
        f.write(response.text)
    print(f"Predictions saved to {output}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)