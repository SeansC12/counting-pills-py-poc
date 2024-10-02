import numpy as np
from scipy import stats

def find_damaged_pills_by_difference(counting_predictions, blob_predictions, distance_betw_trgoh_and_blob_max): # Finding anomalous pills where trgoh does not detect it while blob detection does
    # For each blob_prediction, find the nearest counting_prediction based on their x and y value
    for counting_prediction in counting_predictions:
        x_counting, y_counting = counting_prediction["x"], counting_prediction["y"]
        min_distance = float("inf")
        for blob in blob_predictions:
            x, y = blob
            distance = ((x - x_counting) ** 2 + (y - y_counting) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
        if min_distance > distance_betw_trgoh_and_blob_max:
            counting_prediction["is_damaged"] = True
            counting_prediction["damaged_index"] = 0
            counting_prediction["damaged_signature"] = "Difference between blob and trgoh detection."

def find_damaged_pills_by_area(counting_predictions, area_threshold):
    ROUND_BASE = 5
    # Calculate median area of counting_predictions
    areas = [counting_prediction["width"] * counting_prediction["height"] for counting_prediction in counting_predictions]
    areas_rounded = [ROUND_BASE * round(counting_prediction["width"] * counting_prediction["height"] / ROUND_BASE) for counting_prediction in counting_predictions]

    mode = stats.mode(areas_rounded)

    print(mode, areas_rounded)

    for counting_prediction in counting_predictions:
        area = counting_prediction["width"] * counting_prediction["height"]
        if abs(area - mode[0]) > (0.3 * mode[0]):
            counting_prediction["is_damaged"] = True
            counting_prediction["damaged_index"] = 1
            counting_prediction["damaged_signature"] = "Area too different from the mode."

def generate_final_pill_dict(counting_predictions, blob_predictions, distance_betw_trgoh_and_blob_max, area_threshold):
    for counting_prediction in counting_predictions:
        counting_prediction["is_damaged"] = False
        counting_prediction["damaged_signature"] = "Healthy"
        counting_prediction["damaged_index"] = -1
    find_damaged_pills_by_difference(counting_predictions, blob_predictions, distance_betw_trgoh_and_blob_max)
    find_damaged_pills_by_area(counting_predictions, area_threshold)
    return counting_predictions