import cv2
import imutils
import mediapipe as mp
import numpy as np


mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


cfg_path = "yolov3.cfg"
weights_path = "yolov3.weights"
names_path = "coco.names"


net = cv2.dnn.readNet(weights_path, cfg_path)
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


KNIFE_CLASS_ID = classes.index("knife") if "knife" in classes else -1
STICK_CLASS_ID = classes.index("stick") if "stick" in classes else -1


def detect_suspicious_activity(landmarks, hand_landmarks):
    suspicious_activity = ""

   
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y

    if left_wrist < left_shoulder and right_wrist < right_shoulder:
        suspicious_activity = "Suspicious Fighting detected"

   
    if hand_landmarks:
        for hand in hand_landmarks:
            index_tip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            thumb_tip = hand.landmark[mp_hands.HandLandmark.THUMB_TIP]

            if index_tip.y < middle_tip.y:
                suspicious_activity = "Finger pointing detected"
            if index_tip.x > thumb_tip.x:
                suspicious_activity = "Gun pointing detected"

    return suspicious_activity

# Initialize webcam
cap = cv2.VideoCapture(0)
previous_frame = None 

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

      
        frame_resized = imutils.resize(frame, width=500)

        
        gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        
        motion_detected = False
        if previous_frame is not None:
            frame_delta = cv2.absdiff(previous_frame, gray_frame)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

           
            for contour in contours:
                if cv2.contourArea(contour) > 500:  
                    motion_detected = True
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(frame_resized, "Motion Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    break

        # Update the previous frame
        previous_frame = gray_frame

        # YOLO Object Detection
        blob = cv2.dnn.blobFromImage(frame_resized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        detections = net.forward(output_layers)

        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x = int(obj[0] * frame_resized.shape[1])
                    center_y = int(obj[1] * frame_resized.shape[0])
                    w = int(obj[2] * frame_resized.shape[1])
                    h = int(obj[3] * frame_resized.shape[0])

                    # Draw bounding box and label
                    cv2.rectangle(frame_resized, (center_x - w // 2, center_y - h // 2),
                                  (center_x + w // 2, center_y + h // 2), (0, 255, 0), 2)

                    label = ""
                    if class_id == KNIFE_CLASS_ID:
                        label = "Knife Detected"
                    elif class_id == STICK_CLASS_ID:
                        label = "Stick Detected"

                    if label:
                        cv2.putText(frame_resized, label, (center_x - 10, center_y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Process Pose and Hands for suspicious activity
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_frame)
        hand_results = hands.process(rgb_frame)

        # Draw Pose and Hands landmarks
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame_resized, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame_resized, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Check for suspicious activity
        if pose_results.pose_landmarks:
            suspicious_activity = detect_suspicious_activity(pose_results.pose_landmarks.landmark,
                                                              hand_results.multi_hand_landmarks)
            if suspicious_activity:
                cv2.putText(frame_resized, suspicious_activity, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("Suspicious Activity and Motion Detection", frame_resized)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
