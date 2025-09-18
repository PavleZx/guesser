import cv2
import mediapipe as mp
import random
import numpy as np
import time
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
cv2.namedWindow("MUNJA I DINAMO!", cv2.WND_PROP_FULLSCREEN)

HEAD_LANDMARKS = {
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.LEFT_EYE_INNER,
    mp_pose.PoseLandmark.LEFT_EYE,
    mp_pose.PoseLandmark.LEFT_EYE_OUTER,
    mp_pose.PoseLandmark.RIGHT_EYE_INNER,
    mp_pose.PoseLandmark.RIGHT_EYE,
    mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
    mp_pose.PoseLandmark.LEFT_EAR,
    mp_pose.PoseLandmark.RIGHT_EAR,
    mp_pose.PoseLandmark.MOUTH_LEFT,
    mp_pose.PoseLandmark.MOUTH_RIGHT
}

def draw_lightning(image, start_x, start_y):
    x, y = start_x, start_y
    for i in range(5):
        x_offset = random.randint(-20, 20)
        y_offset = -random.randint(20, 40)
        new_x = x + x_offset
        new_y = y + y_offset
        color = (100, 255, 0)
        cv2.line(image, (x, y), (new_x, new_y), color, 2)
        x, y = new_x, new_y

def process_single_person(frame, landmarks, current_color_mode, elapsed_time):
    h, w, _ = frame.shape
    
    # Set colors based on current mode
    if current_color_mode == 'red':
        line_color = (0, 0, 255)
        fill_color = (0, 0, 255)
    elif current_color_mode == 'blue':
        line_color = (255, 0, 0)
        fill_color = (255, 0, 0)
    elif current_color_mode == 'yellow':
        line_color = (0, 255, 255)
        fill_color = (0, 255, 255)
    elif current_color_mode == 'orange':
        line_color = (0, 165, 255)
        fill_color = (0, 165, 255)
    elif current_color_mode == 'green':
        line_color = (0, 255, 0)
        fill_color = (0, 255, 0)
    elif current_color_mode == 'purple':
        line_color = (128, 0, 128)
        fill_color = (128, 0, 128)
    elif current_color_mode == 'black':
        line_color = (255, 255, 255)
        fill_color = (0, 0, 0)
    elif current_color_mode == 'white':
        line_color = (255, 255, 255)
        fill_color = (255, 255, 255)
    else:
        r = int((math.sin(elapsed_time * 2) + 1) * 127.5)
        g = int((math.sin(elapsed_time * 2 + 2) + 1) * 127.5)
        b = int((math.sin(elapsed_time * 2 + 4) + 1) * 127.5)
        line_color = (b, g, r)
        fill_color = (b, g, r)

    # Get key landmarks
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
    right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
    right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]

    # Calculate shoulder distance
    shoulder_dist = int(np.sqrt((left_shoulder.x - right_shoulder.x)**2 +
                          (left_shoulder.y - right_shoulder.y)**2) * w)

    # Calculate scaling factor
    shoulders = (left_shoulder.y + right_shoulder.y) / 2
    hips = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
    shoulders_hip = abs(hips - shoulders)
    base_shoulder_distance = 300
    scaling_factor_1 = shoulder_dist / base_shoulder_distance
    scaling_factor = max(shoulders_hip, scaling_factor_1) * 1.1

    # Convert landmark positions to pixel coordinates
    nose_x, nose_y = int(nose.x * w), int(nose.y * h)
    left_ear_x, left_ear_y = int(left_ear.x * w), int(left_ear.y * h)
    right_ear_x, right_ear_y = int(right_ear.x * w), int(right_ear.y * h)

    # Filter connections to exclude head landmarks
    pose_connections = mp_pose.POSE_CONNECTIONS
    filtered_connections = [
        connection for connection in pose_connections 
        if connection[0] not in HEAD_LANDMARKS and connection[1] not in HEAD_LANDMARKS
    ]

    # Draw black thicker lines first
    for connection in filtered_connections:
        start_idx, end_idx = connection
        start = landmarks[start_idx]
        end = landmarks[end_idx]
        start_point = (int(start.x * w), int(start.y * h))
        end_point = (int(end.x * w), int(end.y * h))
        cv2.line(frame, start_point, end_point, (0, 0, 0), int(47 * scaling_factor * 1.5))

    # Draw colored thinner lines on top
    for connection in filtered_connections:
        start_idx, end_idx = connection
        start = landmarks[start_idx]
        end = landmarks[end_idx]
        start_point = (int(start.x * w), int(start.y * h))
        end_point = (int(end.x * w), int(end.y * h))
        cv2.line(frame, start_point, end_point, line_color, int(35 * scaling_factor))

    # Draw face ellipse
    ellipse_width = int(100 * scaling_factor)
    ellipse_height = int(130 * scaling_factor)
    cv2.ellipse(
        frame,
        (nose_x, nose_y),
        (ellipse_width, ellipse_height),
        0, 0, 360,
        (0, 0, 0),
        int(20 * scaling_factor * 1.5)
    )
    cv2.ellipse(
        frame,
        (nose_x, nose_y),
        (ellipse_width, ellipse_height),
        0, 0, 360,
        fill_color,
        -1
    )

    # Draw torso
    body_parts = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_HIP
    ]
    pts = []
    for part in body_parts:
        lm = landmarks[part]
        pt = (int(lm.x * w), int(lm.y * h))
        pts.append(pt)
    pts_array = [np.array(pts, dtype=np.int32)]
    cv2.fillPoly(frame, pts_array, fill_color)

# Initialize pose with static_image_mode=False for better performance
# We'll handle multi-person detection differently
with mp_pose.Pose(static_image_mode=False, 
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 model_complexity=1) as pose:
    start_time = time.time()
    current_color_mode = 'rgb'

    while cap.isOpened():
        elapsed_time = time.time() - start_time
        
        # Handle keyboard input
        key = cv2.waitKey(10) & 0xFF
        if key == ord('r'):
            current_color_mode = 'red'
        elif key == ord('b'):
            current_color_mode = 'blue'
        elif key == ord('n'):
            current_color_mode = 'yellow'
        elif key == ord('o'):
            current_color_mode = 'orange'
        elif key == ord('e'):
            current_color_mode = 'green'
        elif key == ord('p'):
            current_color_mode = 'purple'
        elif key == ord('c'):
            current_color_mode = 'black'
        elif key == ord('w'):
            current_color_mode = 'white'
        elif key != 255:
            current_color_mode = 'rgb'

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Make the screen green
        # frame[:] = (0, 255, 0)
        
        # Process the frame to detect poses
        results = pose.process(rgb)

        if results.pose_landmarks:
            # Process the detected person
            process_single_person(frame, results.pose_landmarks.landmark, current_color_mode, elapsed_time)

        cv2.setWindowProperty("MUNJA I DINAMO!", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("MUNJA I DINAMO!", frame)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()