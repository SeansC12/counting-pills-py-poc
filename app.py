from flask import Flask, request
from flask_cors import CORS, cross_origin
import cv2
import base64
import supervision as sv
from inference_sdk import InferenceConfiguration, InferenceHTTPClient
import os
from dotenv import load_dotenv

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

load_dotenv()

MODEL_ID = "trgoh/1"
MODEL_ID_2 = "kkh7-pill-counter-damaged/1"

config = InferenceConfiguration(confidence_threshold=0.5, iou_threshold=0.5)

counting_client = InferenceHTTPClient(
    api_url=os.getenv("INFERENCE_SERVER_URL"),
    api_key=os.getenv("ROBOFLOW_API_KEY"),
)

damaged_client = InferenceHTTPClient(
    api_url=os.getenv("INFERENCE_SERVER_URL"),
    api_key=os.getenv("ROBOFLOW_API_KEY"),
)

counting_client.configure(config)
counting_client.select_model(MODEL_ID)

damaged_client.configure(config)
damaged_client.select_model(MODEL_ID_2)

def getInference(image):
    counting_class_ids = {}
    damaged_class_ids = {}

    counting_predictions = counting_client.infer(image)
    # damaged_predictions = damaged_client.infer(image)

    return counting_predictions

    for p in counting_predictions["predictions"]:
        class_id = p["class_id"]
        if class_id not in counting_class_ids:
            counting_class_ids[class_id] = p["class"]

    for p in damaged_predictions["predictions"]:
        class_id = p["class_id"]
        if class_id not in damaged_class_ids:
            damaged_class_ids[class_id] = p["class"]

@app.route("/", methods=["GET", "POST"])
@cross_origin(send_wildcard=True)
def index():
    counting_predictions = getInference(request.json["image"])
    return counting_predictions