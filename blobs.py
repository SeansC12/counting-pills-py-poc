import cv2
import numpy as np
import base64

def readb64(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_all_blob_coordinates(image):
    # Load the image
    image = readb64(image)

    brightness = -100
    contrast = 100
    image = np.int16(image)
    image = image * (contrast/127+1) - contrast + brightness
    image = np.clip(image, 0, 255)
    image = np.uint8(image)

    cv2.imwrite("data/preprocessed.jpg", image)

    lower = np.array([0, 0, 0])
    upper = np.array([55, 55, 55])

    # Create mask to only select black
    thresh = cv2.inRange(image, lower, upper)

    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Set up the blob detector with parameters tuned for pill detection
    params = cv2.SimpleBlobDetector_Params()

    # Filter by area: Pills are generally medium-sized objects
    params.filterByArea = True
    params.minArea = 30  # Adjusted for the image resolution and pill size
    params.maxArea = 100000000  # Tuned to ignore overly large objects

    # Filter by circularity: Pills are often round or oval
    params.filterByCircularity = False
    # params.minCircularity = 0.6  # Set lower to accommodate oval pills

    # Filter by convexity: Pills typically have a convex shape
    params.filterByConvexity = False
    # params.minConvexity = 0.7  # Can be adjusted for different pill shapes

    # Filter by inertia: Helps to detect elongated objects like capsules
    params.filterByInertia = False
    # params.minInertiaRatio = 0.1  # Allows detection of both circular and elongated pills

    # Set up the detector with the tuned parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs (pills)
    keypoints = detector.detect(morph)

    def draw_keypoints(img, keypoints, color):
        for kp in keypoints:
            x, y = kp.pt
            cv2.circle(img, (int(x), int(y)), color=color, radius=10, thickness=3) # you can change the radius and the thickness
    
    draw_keypoints(morph, keypoints, color=255)
    cv2.imwrite("data/morph.jpg", morph)

    coordinates_of_blobs = list()

    for kp in keypoints:
        x, y = kp.pt
        coordinates_of_blobs.append((int(x), int(y)))

    return coordinates_of_blobs