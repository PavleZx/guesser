import cv2
import json
import numpy as np

# Load face points from JSON
with open('faces.json', 'r') as f:
    face_points = json.load(f)

# Sort and convert to list of tuples
points = []
for key in sorted(face_points.keys(), key=lambda x: int(x[1:])):
    points.append(tuple(face_points[key]))

# Create a window
cv2.namedWindow("Polygon Display", cv2.WINDOW_AUTOSIZE)

# Create a blank image (800x600, white background)
img = np.ones((600, 800, 3), dtype=np.uint8) * 255

# Draw the polygon outline
for i in range(len(points) - 1):
    cv2.line(img, points[i], points[i + 1], (0, 0, 0), 2)

# Fill the polygon
if len(points) > 2:
    polygon_points = np.array(points, np.int32)
    cv2.fillPoly(img, [polygon_points], (255, 255, 0))  # Yellow fill

# Display the image
cv2.imshow("Polygon Display", img)

# Wait for key press and close
cv2.waitKey(0)
cv2.destroyAllWindows()