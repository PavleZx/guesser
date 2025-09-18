import cv2
import mediapipe as mp
import random
import numpy as np
from mediapipe.framework.formats import landmark_pb2
import time
import math
import json


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
cv2.namedWindow("MUNJA I DINAMO!", cv2.WND_PROP_FULLSCREEN)
extra_arm_landmarks = None 

input_name = input(">>")
with open('faces.json', 'r') as f:
    face_points = json.load(f)
    face_points = face_points[input_name]

points_base = []
for key in sorted(face_points.keys(), key=lambda x: int(x[1:])):  # Sort p0, p1, p2, etc.
    points_base.append(tuple(face_points[key]))


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


        # rgb mode ===========================================
        # r = int((math.sin(elapsed_time) + 1) * 127.5)
        # g = int((math.sin(elapsed_time + 2) + 1) * 127.5)
        # b = int((math.sin(elapsed_time + 4) + 1) * 127.5)
        # image_path = "nes2.jpg"  # Putanja do slike

        # line_color = (b, g, r)  # OpenCV koristi BGR redoslijed
        # fill_color = (b, g, r)

        # zuta ===============================================
        # line_color = (0, 255, 255)  # Žuta boja (BGR)
        # fill_color = (0, 255, 255)

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
        elif key == ord('g'):
            current_color_mode = 'brown'
        elif key == ord('x'):
            current_color_mode = 'dark'
        elif key != 255:
            current_color_mode = 'rgb'

        outer_color = (0, 0, 0)  # Default outer color
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
            line_color = (0, 0, 0)
            fill_color = (0, 0, 0)
            r = int((math.sin(elapsed_time * 2) + 1) * 127.5)
            g = int((math.sin(elapsed_time * 2 + 2) + 1) * 127.5)
            b = int((math.sin(elapsed_time * 2 + 4) + 1) * 127.5)
            outer_color = (b, g, r)  # Crna boja za vanjski okvir
            # outer_color = (255, 255, 255)  # Crna boja za vanjski okvir
        elif current_color_mode == 'dark':
            line_color = (0, 0, 0)
            fill_color = (0, 0, 0)
            r = int((math.sin(elapsed_time * 2) + 1) * 127.5)
            outer_color = (255, 255, 255)  # Crna boja za vanjski okvir
        elif current_color_mode == 'white':
            line_color = (255, 255, 255)
            fill_color = (255, 255, 255)
        elif current_color_mode == 'brown':
            line_color = (50, 125, 200)
            fill_color = (50, 125, 200)
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
        # frame[:] = (0, 255, 0)

        # Brown screen
        frame[:] = (0, 20, 125)  # RGB values for brown

        # image = cv2.imread('image.png')

        # # Resize the image to match the frame size if needed
        # image = cv2.resize(image, (frame.shape[1], frame.shape[0]))

        # # Set the frame to the image
        # frame[:] = image
                

        if results.pose_landmarks:
            
            landmarks = results.pose_landmarks.landmark
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
            right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]

            h, w, _ = frame.shape
            left_y = int(left_wrist.y * h)
            right_y = int(right_wrist.y * h)


            # Ako su obje ruke visoko
            if left_y < h // 3 and right_y < h // 3:
                lx, ly = int(left_eye.x * w), int(left_eye.y * h)
                rx, ry = int(right_eye.x * w), int(right_eye.y * h)

                if extra_arm_landmarks is None:
                    # Kopiraj pozicije i pomakni ih niže (npr. +20% visine)
                    extra_arm_landmarks = []
                    for lm in landmarks:
                        new_lm = landmark_pb2.NormalizedLandmark()
                        new_lm.x = lm.x
                        new_lm.y = min(lm.y + 0.2, 1.0)  # pomakni dolje
                        new_lm.z = lm.z
                        new_lm.visibility = lm.visibility
                        extra_arm_landmarks.append(new_lm)

                # draw_lightning(frame, lx, ly)
                # draw_lightning(frame, rx, ry)

                # Dodaj plavi tekst "DINAMO ZAGREB" preko cijelog Zagreba (ekrana)
                # cv2.putText(
                #     frame,
                #     "HAJDUK SPLIT!! 🗿🗿         .",
                #     (int(w * 0.03), int(h * 0.3)),
                #     cv2.FONT_HERSHEY_DUPLEX,
                #     3,
                #     (255, 0, 0),  # plava boja (BGR)
                #     5,
                #     cv2.LINE_AA
                # )
            else:
                # Ako ruke nisu visoko, resetiraj extra_arm_landmarks
                extra_arm_landmarks = None

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

            # # Provjeri granice da ne izađe iz frame-a
            # if (top_left_x >= 0 and top_left_y >= 0 and
            #     bottom_right_y <= frame.shape[0] and bottom_right_x <= frame.shape[1]):
                
            #     # Nacrtaj crni okvir oko kvadrata (outline)
            #     cv2.rectangle(frame,
            #                     (top_left_x-2 + 20, top_left_y-2),
            #                     (bottom_right_x+2 - 20, bottom_right_y+2 - 20),
            #                     (0, 0, 0),
            #                     4)  # Debljina crnog okvira
            
            filtered_connections = [
                connection for connection in pose_connections 
                if connection[0] not in HEAD_LANDMARKS and connection[1] not in HEAD_LANDMARKS
            ]

            # Koordinate glave
            left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
            right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]

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
                cv2.line(frame, start_point, end_point, outer_color, int(47 * scaling_factor * 1.5)) # 63

            # Onda žuta tanja linija preko
            for connection in filtered_connections:
                start_idx, end_idx = connection
                start = landmarks[start_idx]
                end = landmarks[end_idx]

                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))

                # Žuta linija - debljina 55
                cv2.line(frame, start_point, end_point, line_color, int(35 * scaling_factor))  # 50

                if extra_arm_landmarks:
                    for connection in filtered_connections:
                        start_idx, end_idx = connection
                        start = extra_arm_landmarks[start_idx]
                        end = extra_arm_landmarks[end_idx]

                        start_point = (int(start.x * w), int(start.y * h))
                        end_point = (int(end.x * w), int(end.y * h))

                        # Crna linija - debljina 60
                        cv2.line(frame, start_point, end_point, outer_color, int(47 * scaling_factor * 1.5)) # 63

                    for connection in filtered_connections:
                        start_idx, end_idx = connection
                        start = extra_arm_landmarks[start_idx]
                        end = extra_arm_landmarks[end_idx]

                        start_point = (int(start.x * w), int(start.y * h))
                        end_point = (int(end.x * w), int(end.y * h))

                        # Žuta linija - debljina 55
                        cv2.line(frame, start_point, end_point, line_color, int(35 * scaling_factor)) # 50
            
            

            if (head_back_x <= nose_x + int(20 * scaling_factor)):
                # Crtaj 4 vertikalne linije

                # Pretpostavi da je originalni centar (referentna točka) bio:
                center_x = 316
                center_y = 202
                right_ear_x = 247
                right_ear_y = 143

                # Ovo su nove pozicije nosa (dohvaćene iz modela)
                dx = nose_x - 316  # 316 is your reference center_x
                dy = nose_y - 202 

                # Pomakni sve fiksne i izračunate točke
                head_back_x += dx
                head_back_y += dy

                # Apply scaling after shift
                
                
                
                # Također pomakni i središte elipse
                # center_x += dx
                # center_y += dy

                # right_ear_x += dx
                # right_ear_y += dy


                
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
                cv2.fillPoly(frame, pts_array, fill_color) # Žuta boja (BGR)
                

                transformed_points = []
                for point_base in points_base:
                    shifted_point = shift_point(point_base)
                    # Scale from center reference
                    center_ref = (nose_x, nose_y)
                    scaled_point = (
                        int(center_ref[0] + (shifted_point[0] - center_ref[0]) * scaling_factor),
                        int(center_ref[1] + (shifted_point[1] - center_ref[1]) * scaling_factor)
                    )
                    transformed_points.append(scaled_point)

                # Draw lines connecting consecutive points
                # Draw lines connecting consecutive points (including last to first)
                for i in range(len(transformed_points)):
                    cv2.line(frame, transformed_points[i], transformed_points[(i + 1) % len(transformed_points)], outer_color, int(20 * scaling_factor * 1.5))
                # Create polygon from all points except first one for filling
                if len(transformed_points) > 2:
                    polygon_points = np.array(transformed_points, np.int32)
                    cv2.fillPoly(frame, [polygon_points], fill_color)
                # Koordinatni sustav i ellipse
                left_shoulder_y = 0.813735842704773
                right_shoulder_y = 0.8541300296783447
                h = 480
                nose_y = 227
                ellipse_center_base = (316, 202)
                ellipse_center = shift_point(ellipse_center_base)

                ellipse_center = (
                    int(center_ref[0] + (ellipse_center[0] - center_ref[0]) * scaling_factor),
                    int(center_ref[1] + (ellipse_center[1] - center_ref[1]) * scaling_factor)
                )

                radius = int(70 * scaling_factor)

                ellipse_radius_x = int(radius * 1.2)
                ellipse_radius_y = int((163 + int(20 * scaling_factor)) * scaling_factor)            

                cv2.ellipse(
                    frame,
                    ellipse_center,
                    (ellipse_radius_x, ellipse_radius_y),
                    0,
                    -30,
                    90,
                    outer_color,
                    int(20 * scaling_factor* 1.5)
                )

                cv2.ellipse(
                    frame,
                    ellipse_center,
                    (ellipse_radius_x, ellipse_radius_y),
                    0,
                    -30,
                    90,
                    fill_color,
                    -1
                )
                ls_y = int(left_shoulder.y * h)
                rs_y = int(right_shoulder.y * h)

                vertical_radius = ellipse_radius_y
                horizontal_radius = ellipse_radius_x

                # Convert angles to radians
                start_angle_rad = -30 * math.pi / 180  # -30 degrees in radians
                end_angle_rad = 90 * math.pi / 180     # 90 degrees in radians

                # Calculate start point (-30 degrees)

                center_x = ellipse_center[0]
                center_y = ellipse_center[1]
                start_point_x = center_x + horizontal_radius * math.cos(start_angle_rad)
                start_point_y = center_y + vertical_radius * math.sin(start_angle_rad)

                # Calculate end point (90 degrees)
                end_point_x = center_x + horizontal_radius * math.cos(end_angle_rad)
                end_point_y = center_y + vertical_radius * math.sin(end_angle_rad)

                # Round to integers for pixel coordinates
                start_point_x = int(start_point_x)
                start_point_y = int(start_point_y)
                end_point_x = int(end_point_x)
                end_point_y = int(end_point_y)

                
            else:
                # Pomak prema nosu
                center_x = 316
                center_y = 202

                dx = nose_x - center_x
                dy = nose_y - center_y

                head_back_x += dx
                head_back_y += dy

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

                # Mirror and transform points for left side
                transformed_points = []
                for point_base in points_base:
                    # Mirror the point (632 = 2 * 316)
                    mirrored_base = (632 - point_base[0], point_base[1])
                    shifted_point = shift_point(mirrored_base)
                    # Scale from center reference
                    center_ref = (nose_x, nose_y)
                    scaled_point = (
                        int(center_ref[0] + (shifted_point[0] - center_ref[0]) * scaling_factor),
                        int(center_ref[1] + (shifted_point[1] - center_ref[1]) * scaling_factor)
                    )
                    transformed_points.append(scaled_point)

                # Draw lines connecting consecutive points
                # Draw lines connecting consecutive points (including last to first)
                for i in range(len(transformed_points)):
                    cv2.line(frame, transformed_points[i], transformed_points[(i + 1) % len(transformed_points)], outer_color, int(20 * scaling_factor * 1.5))
                
                # Create polygon from all points for filling
                if len(transformed_points) > 2:
                    polygon_points = np.array(transformed_points, np.int32)
                    cv2.fillPoly(frame, [polygon_points], fill_color)

                # Apply same ellipse transformation
                ellipse_center_base = (316, 202)
                ellipse_center = shift_point(ellipse_center_base)

                ellipse_center = (
                    int(center_ref[0] + (ellipse_center[0] - center_ref[0]) * scaling_factor),
                    int(center_ref[1] + (ellipse_center[1] - center_ref[1]) * scaling_factor)
                )

                radius = int(70 * scaling_factor)
                ellipse_radius_x = int(radius * 1.2)
                ellipse_radius_y = int((163 + int(20 * scaling_factor)) * scaling_factor)

                cv2.ellipse(frame, ellipse_center, (ellipse_radius_x, ellipse_radius_y), 0, 90, 210, outer_color, int(20 * scaling_factor* 1.5))
                cv2.ellipse(frame, ellipse_center, (ellipse_radius_x, ellipse_radius_y), 0, 90, 210, fill_color, -1)

                # Calculate points for filling
                vertical_radius = ellipse_radius_y
                horizontal_radius = ellipse_radius_x

                start_angle_rad = 90 * math.pi / 180
                end_angle_rad = 210 * math.pi / 180

                center_x = ellipse_center[0]
                center_y = ellipse_center[1]
                start_point_x = int(center_x + horizontal_radius * math.cos(start_angle_rad))
                start_point_y = int(center_y + vertical_radius * math.sin(start_angle_rad))

                end_point_x = int(center_x + horizontal_radius * math.cos(end_angle_rad))
                end_point_y = int(center_y + vertical_radius * math.sin(end_angle_rad))


            # Popunjavanje tijela žutom
            

            # COMMENTED OUT: Učitavanje slike
            # img_overlay = cv2.imread(image_path)
            
            # ADDED: Replace image loading with drawing a square
            

            # COMMENTED OUT: Dio za prikaz slike na licu
            # if img_overlay is not None:
            #     # Pozicija nosa
            #     nose = landmarks[mp_pose.PoseLandmark.NOSE]
            #     nose_x = int(nose.x * w)
            #     nose_y = int(nose.y * h)
            # 
            #     # Pozicije ramena
            #     left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            #     right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            # 
            #     # Izračun udaljenosti ramena (u pikselima)
            #     shoulder_dist = int(np.sqrt((left_shoulder.x - right_shoulder.x)**2 +
            #                                 (left_shoulder.y - right_shoulder.y)**2) * w)
            # 
            #     # Odredi veličinu slike proporcionalno toj udaljenosti
            #     img_size = int(shoulder_dist * 1.0)
            #     img_size = max(40, min(img_size, 200))  # Ograniči minimalnu/maksimalnu veličinu
            # 
            #     # Resize slike
            #     img_overlay_resized = cv2.resize(img_overlay, (img_size, img_size))
            # 
            #     # Koordinate gdje ćemo zalijepit sliku
            #     top_left_x = nose_x - img_size // 2
            #     top_left_y = nose_y - img_size // 2
            # 
            #     # Provjeri granice da ne izađe iz frame-a
            #     if top_left_x >= 0 and top_left_y >= 0 and top_left_y + img_size <= frame.shape[0] and top_left_x + img_size <= frame.shape[1]:
            #         # Zalijepi sliku
            #         frame[top_left_y:top_left_y+img_size, top_left_x:top_left_x+img_size] = img_overlay_resized

            cv2.setWindowProperty("MUNJA I DINAMO!", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # Dodano
            cv2.imshow("MUNJA I DINAMO!", frame)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()