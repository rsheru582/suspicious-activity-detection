import cv2
import imutils
import numpy as np
import os

# Check if the YOLO files exist
cfg_path = "yolov3.cfg"
weights_path = "yolov3.weights"

if not os.path.exists(cfg_path):
    raise FileNotFoundError(f"{cfg_path} not found. Please download the YOLOv3 config file.")

if not os.path.exists(weights_path):
    raise FileNotFoundError(f"{weights_path} not found. Please download the YOLOv3 weights file.")

# Load YOLO weights and config file
try:
    net = cv2.dnn.readNet(weights_path, cfg_path)
except cv2.error as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# Get YOLO output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize the first frame for motion detection
first_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for faster processing
    frame = imutils.resize(frame, width=500)
    
    # Convert the frame to grayscale for motion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = gray
        continue

    # Compute the absolute difference between the current frame and the first frame
    frame_delta = cv2.absdiff(first_frame, gray)
    thresh = cv2.threshold(frame_delta, 200, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    suspicious = False  # Flag for suspicious activity

    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        suspicious = True  # Motion detected, set suspicious flag

    # Object detection with YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(obj[0] * frame.shape[1])
                center_y = int(obj[1] * frame.shape[0])
                w = int(obj[2] * frame.shape[1])
                h = int(obj[3] * frame.shape[0])

                # Draw bounding box for detected objects
                cv2.rectangle(frame, (center_x, center_y), (center_x + w, center_y + h), (0, 255, 0), 2)

                # If a person is detected and motion is flagged, classify as suspicious activity
                if class_id == 0 and suspicious:
                    cv2.putText(frame, "Suspicious Activity Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame with detected motion and objects
    cv2.imshow('Suspicious Activity Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
