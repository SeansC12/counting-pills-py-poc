import cv2
import base64
import supervision as sv
from inference_sdk import InferenceConfiguration, InferenceHTTPClient
import os
from dotenv import load_dotenv

load_dotenv()

IMAGE_PATH = "imgs/IMG_3100.jpg"

pre_image = cv2.imread(IMAGE_PATH)
jpg_pre_img = cv2.imencode(".jpg", pre_image)
image = base64.b64encode(jpg_pre_img[1]).decode('utf-8')

# MODEL_ID = "06022023/4"
MODEL_ID = "pill-counter-trgoh/1"
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

counting_class_ids = {}
damaged_class_ids = {}

counting_predictions = counting_client.infer(image)
damaged_predictions = damaged_client.infer(image)

print(counting_predictions)

for p in counting_predictions["predictions"]:
    class_id = p["class_id"]
    if class_id not in counting_class_ids:
        counting_class_ids[class_id] = p["class"]

for p in damaged_predictions["predictions"]:
    class_id = p["class_id"]
    if class_id not in damaged_class_ids:
        damaged_class_ids[class_id] = p["class"]

counting_detections = sv.Detections.from_inference(counting_predictions)
damaged_detections = sv.Detections.from_inference(damaged_predictions)

image = cv2.imread(IMAGE_PATH)

counting_box_annotator = sv.BoxAnnotator(thickness=5, color=sv.Color.BLACK)
damaged_box_annotator = sv.BoxAnnotator(thickness=5, color=sv.Color.RED)
# labels = [
#     f"{class_ids[class_id]} {confidence:0.2f}"
#     for a, b, confidence, class_id, c in detections
# ]

# labels = [
#     f"{counting_class_ids[class_id]} {confidence:0.2f}"
#     for _, _, confidence, class_id, _, _ in counting_detections
# ]

print(damaged_detections)

annotated_frame = counting_box_annotator.annotate(
    scene=image.copy(), detections=counting_detections
)

final_frame = damaged_box_annotator.annotate(
    scene=annotated_frame.copy(), detections=damaged_detections
)

sv.plot_image(image=final_frame, size=(16, 16))