from flask import Flask, request
import torch
import torch.nn as nn
import pandas as pd
import io
import os
from models import Generater_MLP_Skip  # assumes this file is copied to the container

app = Flask(__name__)

# --- Load the model ---
def load_model(model_dir="/app"):
    checkpoint = torch.load(os.path.join(model_dir, "model.pth"), map_location="cpu")
    feat_size, Gen_size, Gen_depth = 85, 300, 6
    model = Generater_MLP_Skip(feat_size, Gen_size, feat_size, Gen_depth)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

model = load_model()

# --- Flask endpoints ---
@app.route("/ping", methods=["GET"])
def ping():
    return "pong", 200

@app.route("/invocations", methods=["POST"])
def invoke():
    if request.content_type != "text/csv":
        return "Only 'text/csv' supported", 415

    try:
        # Read CSV input
        csv_data = request.data.decode("utf-8")
        df = pd.read_csv(io.StringIO(csv_data))
        input_tensor = torch.tensor(df.values, dtype=torch.float32)

        # Predict
        with torch.no_grad():
            output = model(input_tensor)
        pred_df = pd.DataFrame(output.numpy())

        # Convert result to CSV
        result_csv = io.StringIO()
        pred_df.to_csv(result_csv, index=False)
        return result_csv.getvalue(), 200, {"Content-Type": "text/csv"}

    except Exception as e:
        return f"Error during prediction: {str(e)}", 500
