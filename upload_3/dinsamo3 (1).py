import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture(0)
cv2.namedWindow("YOLOv8 Pose Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("YOLOv8 Pose Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Parovi točaka koje ćemo spojiti linijama
LIMB_CONNECTIONS = [
    (5, 7), (7, 9),      # Left arm
    (6, 8), (8, 10),     # Right arm
    (5, 6),              # Shoulders
    (11, 12),            # Hips
    (5, 11), (6, 12),    # Torso sides
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16)   # Right leg
]

# Boje za svaku kost
LIMB_COLORS = (255, 0, 255)  # Ljubičasta (možeš promijenit)

# Različite boje za kružiće oko nosa
PERSON_COLORS = [
    (255, 0, 0),    # Crvena
    (0, 255, 0),    # Zelena
    (0, 0, 255),    # Plava
    (255, 255, 0),  # Žuta
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 128, 0),  # Narančasta
    (128, 0, 255),  # Ljubičasta
]

def calculate_thickness(p1, p2, scale=0.08, min_thick=2):
    distance = np.linalg.norm(np.array(p1) - np.array(p2))
    return max(int(distance * scale), min_thick) * 2

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    

    results = model(frame)[0]

    frame[:] = (0, 255, 0)

    for idx, person in enumerate(results.keypoints):
        keypoints = person.xy[0].cpu().numpy().tolist()

        # Crtanje točkica
        for x, y in keypoints:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

        # Crtanje udova s duplim linijama
        for start_idx, end_idx in LIMB_CONNECTIONS:
            x1, y1 = keypoints[start_idx]
            x2, y2 = keypoints[end_idx]
            if x1 > 0 and x2 > 0:
                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))

                thickness = calculate_thickness(pt1, pt2)

                # Tanji obojani sloj
                person_color = PERSON_COLORS[idx % len(PERSON_COLORS)]
                cv2.line(frame, pt1, pt2, person_color, thickness)

        # Ispuni torzo (četverokut: ramena i bokovi)
        person_color = PERSON_COLORS[idx % len(PERSON_COLORS)]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        
        # Provjeri da li su sve točke detektirane
        if all(x > 0 and y > 0 for x, y in [left_shoulder, right_shoulder, left_hip, right_hip]):
            torso_points = np.array([
                [int(left_shoulder[0]), int(left_shoulder[1])],
                [int(right_shoulder[0]), int(right_shoulder[1])],
                [int(right_hip[0]), int(right_hip[1])],
                [int(left_hip[0]), int(left_hip[1])]
            ], np.int32)
            
            # Nacrtaj ispunjeni poligon s transparentnošću
            cv2.fillPoly(frame, [torso_points], person_color)

        # Obojani kružić oko nosa
        nose_x, nose_y = map(int, keypoints[0])
        if nose_x > 0 and nose_y > 0:  # Provjeri da li je nos detektiran
            # Uzmi boju za ovu osobu (ciklički kroz dostupne boje)
            person_color = PERSON_COLORS[idx % len(PERSON_COLORS)]
            
            # Izračunaj veličinu kružića na osnovu širine ramena
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            if left_shoulder[0] > 0 and right_shoulder[0] > 0:
                shoulder_distance = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))
                circle_radius = max(int(shoulder_distance * 0.35), 15)  # 15% širine ramena, minimum 15px
            else:
                circle_radius = 25  # Default ako ramena nisu detektirana
            
            if idx == 0:
                face_img = cv2.imread('img1.png', cv2.IMREAD_UNCHANGED)
            else:
                face_img = cv2.imread('img2.png', cv2.IMREAD_UNCHANGED)

            if face_img is not None:
                # Resize image to fit circle
                face_img = cv2.resize(face_img, (circle_radius*2, circle_radius*2))
                
                # Position for image
                y1, y2 = nose_y - circle_radius, nose_y + circle_radius  
                x1, x2 = nose_x - circle_radius, nose_x + circle_radius
                
                # Make sure coordinates are within frame bounds
                if y1 >= 0 and y2 < frame.shape[0] and x1 >= 0 and x2 < frame.shape[1]:
                    if face_img.shape[2] == 4:  # Has alpha channel
                        alpha = face_img[:, :, 3] / 255.0
                        for c in range(3):
                            frame[y1:y2, x1:x2, c] = (alpha * face_img[:, :, c] + 
                                                    (1 - alpha) * frame[y1:y2, x1:x2, c])
                    else:
                        frame[y1:y2, x1:x2] = face_img
            else:
                # Fallback to circle if image not found
                cv2.circle(frame, (nose_x, nose_y), circle_radius, person_color, -1)

            # Ime osobe iznad kružića
            cv2.putText(frame, f"Person {idx + 1}", (nose_x - 30, nose_y - circle_radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, person_color, 2)

    cv2.imshow("YOLOv8 Pose Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()