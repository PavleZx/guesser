import cv2
import mediapipe as mp
import random
import numpy as np
from mediapipe.framework.formats import landmark_pb2
import time
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
cv2.namedWindow("MUNJA I DINAMO!", cv2.WND_PROP_FULLSCREEN)
extra_arm_landmarks = None 

# Add these variables for image size and position control
image_size_multiplier = 1.0  # Default size multiplier
image_y_offset = 0  # Vertical offset for image position

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

def shift_point(pt):
    return (pt[0] + dx, pt[1] + dy)

with mp_pose.Pose() as pose:

    start_time = time.time()
    current_color_mode = 'rgb'

    while cap.isOpened():

        elapsed_time = time.time() - start_time

        key = cv2.waitKey(10) & 0xFF
        # Reaction to keys - existing color controls
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
        # Add new controls for image size and position
        elif key == ord('k'):  # Make image larger
            image_size_multiplier += 0.1
            image_size_multiplier = min(image_size_multiplier, 3.0)  # Max 3x size
        elif key == ord('l'):  # Make image smaller
            image_size_multiplier -= 0.1
            image_size_multiplier = max(image_size_multiplier, 0.3)  # Min 0.3x size
        elif key == ord('u'):  # Up arrow key
            image_y_offset -= 10  # Move image up
        elif key == ord('j'):  # Down arrow key
            image_y_offset += 10  # Move image down
        elif key != 255:
            current_color_mode = 'rgb'
        outer_color = (0, 0, 0)  # Default outer color

        # Color setting according to current mode
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
            line_color = (0, 0, 0)
            fill_color = (0, 0, 0)
            outer_color = (255, 255, 255)  # Outer color for black mode
        elif current_color_mode == 'white':
            line_color = (255, 255, 255)
            fill_color = (255, 255, 255)
        else:
            r = int((math.sin(elapsed_time * 2) + 1) * 127.5)
            g = int((math.sin(elapsed_time * 2 + 2) + 1) * 127.5)
            b = int((math.sin(elapsed_time * 2 + 4) + 1) * 127.5)
            line_color = (b, g, r)
            fill_color = (b, g, r)

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        # Make the screen green
        frame[:] = (0, 255, 0)
        

        if results.pose_landmarks:
            
            landmarks = results.pose_landmarks.landmark
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
            right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]

            h, w, _ = frame.shape
            left_y = int(left_wrist.y * h)
            right_y = int(right_wrist.y * h)

            # If both hands are high
            if left_y < h // 3 and right_y < h // 3:
                lx, ly = int(left_eye.x * w), int(left_eye.y * h)
                rx, ry = int(right_eye.x * w), int(right_eye.y * h)

                if extra_arm_landmarks is None:
                    # Copy positions and move them down
                    extra_arm_landmarks = []
                    for lm in landmarks:
                        new_lm = landmark_pb2.NormalizedLandmark()
                        new_lm.x = lm.x
                        new_lm.y = min(lm.y + 0.2, 1.0)  # move down
                        new_lm.z = lm.z
                        new_lm.visibility = lm.visibility
                        extra_arm_landmarks.append(new_lm)

                draw_lightning(frame, lx, ly)
                draw_lightning(frame, rx, ry)

                # Add blue text "DINAMO ZAGREB" over the entire Zagreb (screen)
                # cv2.putText(
                #     frame,
                #     "HAJDUK SPLIT!! 🗿🗿         .",
                #     (int(w * 0.03), int(h * 0.3)),
                #     cv2.FONT_HERSHEY_DUPLEX,
                #     3,
                #     (255, 0, 0),  # blue color (BGR)
                #     5,
                #     cv2.LINE_AA
                #     )
            else:
                # If hands are not high, reset extra_arm_landmarks
                extra_arm_landmarks = None

            pose_connections = mp_pose.POSE_CONNECTIONS

            # Nose position
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            nose_x = int(nose.x * w)
            nose_y = int(nose.y * h)

            # Ear positions
            left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
            right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]

            # Shoulder positions
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            # Calculate shoulder distance (in pixels)
            shoulder_dist = int(np.sqrt((left_shoulder.x - right_shoulder.x)**2 +
                                        (left_shoulder.y - right_shoulder.y)**2) * w)

            # Determine square size proportional to that distance
            img_size = int(shoulder_dist * 1.0)
            img_size = max(40, min(img_size, 200))  # Limit min/max size

            # Coordinates where we'll draw the square
            ear_x = int((left_ear.x + right_ear.x) * w // 2)
            ear_y = int((left_ear.y + right_ear.y) * h // 2)

            top_left_x = ear_x - img_size // 2
            top_left_y = ear_y - img_size // 2
            bottom_right_x = top_left_x + img_size
            bottom_right_y = top_left_y + img_size
            
            filtered_connections = [
                connection for connection in pose_connections 
                if connection[0] not in HEAD_LANDMARKS and connection[1] not in HEAD_LANDMARKS
            ]

            # Head coordinates
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
            right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]

            nose_x = int((left_eye.x + right_eye.x) * w / 2)
            nose_y = int((left_eye.y + right_eye.y) * h / 2)
            left_ear_x, left_ear_y = int(left_ear.x * w), int(left_ear.y * h)
            right_ear_x, right_ear_y = int(right_ear.x * w), int(right_ear.y * h)

            # Calculate scaling factor based on ear distance (larger when closer to screen)
            shoulders = (left_shoulder.y + right_shoulder.y) / 2
            hips = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
            shoulders_hip = abs(hips - shoulders)
            base_shoulder_distance = 300  # Reference distance when at normal distance

            scaling_factor_1 = shoulder_dist / base_shoulder_distance  # Clamp between 0.5x and 2.0x
            scaling_factor = max(shoulders_hip, scaling_factor_1) * 1.1
            # Estimate back of head (behind nose, on line between ears)
            head_back_x = int((left_ear_x + right_ear_x) / 2 - (nose_x - (left_ear_x + right_ear_x)/2))
            head_back_y = int((left_ear_y + right_ear_y) / 2)

            # Line height (from eyes downward) - scaled
            line_length = int(0.15 * h * scaling_factor)
            # Center of quarter circle: between eyes
            center_x = int((left_eye.x + right_eye.x) * w / 2)
            center_y = int((left_eye.y + right_eye.y) * h / 2)

            # Radius - scaled
            ear_dist = (left_ear.x - right_ear.x) * w
            radius = int(ear_dist / 2 * scaling_factor)

            overlay_img = cv2.imread('img4.jpg')

            overlay_img = cv2.imread('img4.jpg')

            if overlay_img is not None:
                # Get original image dimensions
                original_height, original_width = overlay_img.shape[:2]
                aspect_ratio = original_width / original_height
                
                # Calculate height based on scaling (keeping your current height calculation)
                img_height = int(340 * scaling_factor * image_size_multiplier)
                # Calculate width to maintain aspect ratio
                img_width = int(img_height * aspect_ratio)
                
                # Resize the overlay image
                overlay_resized = cv2.resize(overlay_img, (img_width, img_height))
                    
                # Calculate top-left position for centering on nose with Y offset
                x_start = nose_x - img_width // 2
                y_start = nose_y - img_height // 2 + image_y_offset
                
                # Make sure the image doesn't go outside frame boundaries
                x_start = max(0, min(x_start, frame.shape[1] - img_width))
                y_start = max(0, min(y_start, frame.shape[0] - img_height))
                
                # Calculate actual end positions
                x_end = x_start + img_width
                y_end = y_start + img_height
                
                # Overlay the image onto the frame
                frame[y_start:y_end, x_start:x_end] = overlay_resized

            # First black thicker line
            for connection in filtered_connections:
                start_idx, end_idx = connection
                start = landmarks[start_idx]
                end = landmarks[end_idx]

                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))

                # Black line - thickness 60
                cv2.line(frame, start_point, end_point, outer_color, int(47 * scaling_factor * 1.5)) # 63

            # Then yellow thinner line over
            for connection in filtered_connections:
                start_idx, end_idx = connection
                start = landmarks[start_idx]
                end = landmarks[end_idx]

                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))

                # Yellow line - thickness 55
                cv2.line(frame, start_point, end_point, line_color, int(35 * scaling_factor))  # 50

                if extra_arm_landmarks:
                    for connection in filtered_connections:
                        start_idx, end_idx = connection
                        start = extra_arm_landmarks[start_idx]
                        end = extra_arm_landmarks[end_idx]

                        start_point = (int(start.x * w), int(start.y * h))
                        end_point = (int(end.x * w), int(end.y * h))

                        # Black line - thickness 60
                        cv2.line(frame, start_point, end_point, outer_color, int(47 * scaling_factor * 1.5)) # 63

                    for connection in filtered_connections:
                        start_idx, end_idx = connection
                        start = extra_arm_landmarks[start_idx]
                        end = extra_arm_landmarks[end_idx]

                        start_point = (int(start.x * w), int(start.y * h))
                        end_point = (int(end.x * w), int(end.y * h))

                        # Yellow line - thickness 55
                        cv2.line(frame, start_point, end_point, line_color, int(35 * scaling_factor)) # 50

            # Draw body parts (keeping this part as it was in both conditions)
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

            cv2.setWindowProperty("MUNJA I DINAMO!", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("MUNJA I DINAMO!", frame)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

