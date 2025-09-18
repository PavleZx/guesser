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
        # image_path = "nes2.png"  # Commented out: Putanja do slike

        key = cv2.waitKey(10) & 0xFF
        # Reakcija na tipke
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

        # Postavljanje boja prema trenutnom modu
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



            pose_connections = mp_pose.POSE_CONNECTIONS

            # Pozicija nosa
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            nose_x = int(nose.x * w)
            nose_y = int(nose.y * h)

            # pozicije usiju
            left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
            right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]

            # Pozicije ramena
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            # Izračun udaljenosti ramena (u pikselima)
            shoulder_dist = int(np.sqrt((left_shoulder.x - right_shoulder.x)**2 +
                                        (left_shoulder.y - right_shoulder.y)**2) * w)

            # Odredi veličinu kvadrata proporcionalno toj udaljenosti
            img_size = int(shoulder_dist * 1.0)
            img_size = max(40, min(img_size, 200))  # Ograniči minimalnu/maksimalnu veličinu

            # Koordinate gdje ćemo nacrtati kvadrat
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

            # Koordinate glave
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
            right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]

            nose_x, nose_y = int(nose.x * w), int(nose.y * h)
            left_ear_x, left_ear_y = int(left_ear.x * w), int(left_ear.y * h)
            right_ear_x, right_ear_y = int(right_ear.x * w), int(right_ear.y * h)

            # Calculate scaling factor based on ear distance (larger when closer to screen)
            shoulders = (left_shoulder.y + right_shoulder.y) / 2
            hips = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
            shoulders_hip = abs(hips - shoulders)
            base_shoulder_distance = 300  # Reference distance when at normal distance


            scaling_factor_1 = shoulder_dist / base_shoulder_distance  # Clamp between 0.5x and 2.0x
            scaling_factor = max(shoulders_hip, scaling_factor_1) * 1.1


            # Procjena stražnje strane glave (iza nosa, na liniji između ušiju)
            head_back_x = int((left_ear_x + right_ear_x) / 2 - (nose_x - (left_ear_x + right_ear_x)/2))
            head_back_y = int((left_ear_y + right_ear_y) / 2)

            # Visina linija (od očiju prema dolje) - scaled
            line_length = int(0.15 * h * scaling_factor)
            # Centar četvrtine kruga: između očiju
            center_x = int((left_eye.x + right_eye.x) * w / 2)
            center_y = int((left_eye.y + right_eye.y) * h / 2)

            # Polumjer - scaled
            ear_dist = (left_ear.x - right_ear.x) * w
            radius = int(ear_dist / 2 * scaling_factor)

            # Prvo crna šira linija
            for connection in filtered_connections:
                start_idx, end_idx = connection
                start = landmarks[start_idx]
                end = landmarks[end_idx]

                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))

                # Crna linija - debljina 60
                cv2.line(frame, start_point, end_point, (0, 0, 0), int(47 * scaling_factor * 1.5)) # 63

            # Onda žuta tanja linija preko
            for connection in filtered_connections:
                start_idx, end_idx = connection
                start = landmarks[start_idx]
                end = landmarks[end_idx]

                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))

                # Žuta linija - debljina 55
                cv2.line(frame, start_point, end_point, line_color, int(35 * scaling_factor))  # 50

            ellipse_center = (nose_x, nose_y)

            # Define ellipse dimensions - slightly taller than wide
            ellipse_width = int(100 * scaling_factor)  # horizontal radius
            ellipse_height = int(130 * scaling_factor)  # vertical radius (taller)

            # Draw the ellipse outline
            cv2.ellipse(
                frame,
                ellipse_center,
                (ellipse_width, ellipse_height),
                0,  # rotation angle
                0,  # start angle (full ellipse)
                360,  # end angle (full ellipse)
                (0, 0, 0),  # black color for outline
                int(20 * scaling_factor * 1.5)  # thickness
            )

            # Fill the ellipse
            cv2.ellipse(
                frame,
                ellipse_center,
                (ellipse_width, ellipse_height),
                0,  # rotation angle
                0,  # start angle (full ellipse)
                360,  # end angle (full ellipse)
                fill_color,  # fill color
                -1  # filled
            )

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


            cv2.setWindowProperty("MUNJA I DINAMO!", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # Dodano
            cv2.imshow("MUNJA I DINAMO!", frame)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()