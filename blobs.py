import cv2
import numpy as np
import base64

def readb64(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    return img

def get_all_blob_coordinates(image):
    # Load the image
    image = readb64(image)

    # Apply a Gaussian blur to reduce noise and improve blob detection
    blurred_image = cv2.GaussianBlur(image, (7, 7), 0)
    blurred_image = cv2.bitwise_not(blurred_image)

    brightness = 50
    contrast = 40
    blurred_image = np.int16(blurred_image)
    blurred_image = blurred_image * (contrast/127+1) - contrast + brightness
    blurred_image = np.clip(blurred_image, 0, 255)
    blurred_image = np.uint8(blurred_image)

    # Set up the blob detector with parameters tuned for pill detection
    params = cv2.SimpleBlobDetector_Params()

    # Filter by area: Pills are generally medium-sized objects
    params.filterByArea = False
    # params.minArea = 300  # Adjusted for the image resolution and pill size
    # params.maxArea = 100000000  # Tuned to ignore overly large objects

    # Filter by circularity: Pills are often round or oval
    params.filterByCircularity = True
    params.minCircularity = 0.6  # Set lower to accommodate oval pills

    # Filter by convexity: Pills typically have a convex shape
    params.filterByConvexity = True
    params.minConvexity = 0.7  # Can be adjusted for different pill shapes

    # Filter by inertia: Helps to detect elongated objects like capsules
    params.filterByInertia = True
    params.minInertiaRatio = 0.1  # Allows detection of both circular and elongated pills

    # Set up the detector with the tuned parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs (pills)
    keypoints = detector.detect(blurred_image)

    coordinates_of_damaged = list()

    for kp in keypoints:
        x, y = kp.pt
        coordinates_of_damaged.append((int(x), int(y)))

    return coordinates_of_damaged