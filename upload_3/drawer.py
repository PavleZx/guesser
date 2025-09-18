import cv2
import json
import numpy as np
import tkinter as tk
from tkinter import simpledialog
import os

class PolygonEditor:
    def __init__(self):
        # Initial triangle points
        self.points = [(400, 200), (300, 400), (500, 400)]
        self.dragging = False
        self.drag_point_index = -1
        self.drag_offset = (0, 0)
        
        # Vertical scaling factor to compensate for face detection stretching
        # This makes the editor show a slightly more vertically compressed version
        # so it appears correct when stretched by the face detection
        self.vertical_scale_factor = 0.75  # Adjust this value (0.6-0.9) if needed
        
        # Create window with aspect ratio closer to camera feed
        cv2.namedWindow("Polygon Editor", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Polygon Editor", self.mouse_callback)
        
        # Button areas (top of screen)
        self.add_button_area = (10, 10, 120, 50)  # x1, y1, x2, y2
        self.save_button_area = (140, 10, 250, 50)
        
    def point_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def is_point_in_rect(self, point, rect):
        x, y = point
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def find_closest_point(self, mouse_pos, threshold=15):
        for i, point in enumerate(self.points):
            # Apply inverse scaling when checking mouse position
            scaled_point = (point[0], int(point[1] / self.vertical_scale_factor))
            if self.point_distance(mouse_pos, scaled_point) < threshold:
                return i
        return -1
    
    def mouse_callback(self, event, x, y, flags, param):
        mouse_pos = (x, y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking on buttons
            if self.is_point_in_rect(mouse_pos, self.add_button_area):
                self.add_point(mouse_pos)
                return
            elif self.is_point_in_rect(mouse_pos, self.save_button_area):
                self.save_polygon()
                return
            
            # Check if clicking on a point
            point_index = self.find_closest_point(mouse_pos)
            if point_index != -1:
                self.dragging = True
                self.drag_point_index = point_index
                # Calculate offset with scaling compensation
                display_point = (self.points[point_index][0], 
                               int(self.points[point_index][1] / self.vertical_scale_factor))
                self.drag_offset = (x - display_point[0], y - display_point[1])
        
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            # Move the point with scaling compensation
            new_x = x - self.drag_offset[0]
            new_y = int((y - self.drag_offset[1]) * self.vertical_scale_factor)
            self.points[self.drag_point_index] = (new_x, new_y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.drag_point_index = -1
    
    def add_point(self, mouse_pos):
        # Add point at center of screen when button is clicked
        center_x, center_y = 400, int(300 * self.vertical_scale_factor)
        self.points.append((center_x, center_y))
    
    def save_polygon(self):
        # Create a simple dialog using tkinter
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        name = simpledialog.askstring("Save Polygon", "Enter polygon name:")
        root.destroy()
        
        if name:
            self.save_to_json(name)
    
    def save_to_json(self, name):
        # Load existing faces.json or create new structure
        if os.path.exists('faces.json'):
            try:
                with open('faces.json', 'r') as f:
                    data = json.load(f)
            except:
                data = {}
        else:
            data = {}
        
        # Convert points to the required format
        polygon_data = {}
        for i, point in enumerate(self.points):
            polygon_data[f"p{i}"] = list(point)
        
        # Save under the given name
        data[name] = polygon_data
        
        # Write back to file
        with open('faces.json', 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Polygon '{name}' saved with {len(self.points)} points")
    
    def draw_interface(self, img):
        # Clear image
        img.fill(255)
        
        # Draw buttons
        cv2.rectangle(img, (self.add_button_area[0], self.add_button_area[1]), 
                     (self.add_button_area[2], self.add_button_area[3]), (200, 200, 200), -1)
        cv2.rectangle(img, (self.save_button_area[0], self.save_button_area[1]), 
                     (self.save_button_area[2], self.save_button_area[3]), (200, 200, 200), -1)
        
        cv2.putText(img, "Add Point", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img, "Save", (165, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Transform points for display (apply inverse vertical scaling)
        display_points = []
        for point in self.points:
            display_point = (point[0], int(point[1] / self.vertical_scale_factor))
            display_points.append(display_point)
        
        # Draw polygon with transformed points
        if len(display_points) >= 3:
            # Fill polygon
            polygon_points = np.array(display_points, np.int32)
            cv2.fillPoly(img, [polygon_points], (255, 255, 0))  # Yellow fill
            
            # Draw outline
            for i in range(len(display_points)):
                start_point = display_points[i]
                end_point = display_points[(i + 1) % len(display_points)]
                cv2.line(img, start_point, end_point, (0, 0, 0), 2)
        
        # Draw ellipse (green, unmoving) - keep original proportions as reference
        ellipse_center = (316, 202)
        ellipse_radius_x = int(70 * 1.2)
        ellipse_radius_y = int((163 + 20) * 1.0)
        cv2.ellipse(img, ellipse_center, (ellipse_radius_x, ellipse_radius_y), 0, -30, 90, (0, 255, 0), -1)
        
        # Draw points as circles with transformed positions
        for i, display_point in enumerate(display_points):
            color = (0, 0, 255) if i == self.drag_point_index else (0, 255, 0)
            cv2.circle(img, display_point, 8, color, -1)
            cv2.putText(img, str(i), (display_point[0] + 10, display_point[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Add scaling info text
        cv2.putText(img, f"V-Scale: {self.vertical_scale_factor:.2f}", (10, img.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    def run(self):
        # Use 4:3 aspect ratio similar to camera feed (640x480 scaled up)
        img = np.ones((600, 800, 3), dtype=np.uint8) * 255
        
        while True:
            self.draw_interface(img)
            cv2.imshow("Polygon Editor", img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == ord('+') or key == ord('='):  # Increase vertical scaling
                self.vertical_scale_factor = min(1.0, self.vertical_scale_factor + 0.05)
                print(f"Vertical scale factor: {self.vertical_scale_factor:.2f}")
            elif key == ord('-'):  # Decrease vertical scaling
                self.vertical_scale_factor = max(0.5, self.vertical_scale_factor - 0.05)
                print(f"Vertical scale factor: {self.vertical_scale_factor:.2f}")
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    editor = PolygonEditor()
    editor.run()