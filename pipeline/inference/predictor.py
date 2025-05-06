from __future__ import print_function

import io
import os

import flask
import pandas as pd
import torch
from models import Generater_MLP_Skip

import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

logger.info("Starting the predictor...")

model_path = os.environ['MODEL_PATH']

# --- ScoringService: loads and uses your PyTorch model ---
class ScoringService(object):
    model = None

    @classmethod
    def get_model(cls):
        if cls.model is None:
            logger.info("Loading model...")
            checkpoint = torch.load(os.path.join(model_path, "model.pth"), map_location="cpu")
            feat_size, Gen_size, Gen_depth = 85, 300, 6
            cls.model = Generater_MLP_Skip(feat_size, Gen_size, feat_size, Gen_depth)
            cls.model.load_state_dict(checkpoint)
            cls.model.eval()
            logger.info("Model loaded")
        return cls.model

    @classmethod
    def predict(cls, input_df):
        model = cls.get_model()
        input_tensor = torch.tensor(input_df.values, dtype=torch.float32)
        with torch.no_grad():
            output = model(input_tensor)
        return output.numpy()

# --- Flask app ---
app = flask.Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    """Health check"""
    try:
        model = ScoringService.get_model()
        health = model is not None
        logger.info("/ping OK")
        return flask.Response(response="\n", status=200 if health else 404, mimetype="application/json")
    except Exception as e:
        logger.info(f"/ping failed: {e}")
        return flask.Response(response="\n", status=500, mimetype="application/json")

@app.route("/invocations", methods=["POST"])
def transformation():
    """Prediction endpoint that accepts CSV and returns CSV."""
    if flask.request.content_type != "text/csv":
        return flask.Response(response="This predictor only supports text/csv", status=415, mimetype="text/plain")

    try:
        # Parse CSV input
        csv_data = flask.request.data.decode("utf-8")
        df = pd.read_csv(io.StringIO(csv_data))

        logger.info(f"Received {df.shape[0]} rows")

        # Predict
        preds = ScoringService.predict(df)

        # Return predictions as CSV
        output_csv = io.StringIO()
        pd.DataFrame(preds).to_csv(output_csv, index=False, header=False)
        return flask.Response(response=output_csv.getvalue(), status=200, mimetype="text/csv")

    except Exception as e:
        logger.info(f"Error during prediction: {e}")
        return flask.Response(response=f"Error: {str(e)}", status=500, mimetype="text/plain")
    
