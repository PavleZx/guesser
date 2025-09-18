import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math

class HandGestureMouseControl:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        pyautogui.FAILSAFE = False
        
        # Control square parameters
        self.square_size = 200
        self.square_x = 50
        self.square_y = 50
        
        # Click detection parameters
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
    def get_finger_tip_position(self, hand_landmarks):
        """Get the position of the index finger tip"""
        if hand_landmarks:
            # Index finger tip is landmark 8
            finger_tip = hand_landmarks.landmark[8]
            return finger_tip.x, finger_tip.y
        return None, None
    
    def get_shoulder_positions(self, pose_landmarks):
        """Get shoulder positions to define the control square"""
        if pose_landmarks:
            left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            return left_shoulder, right_shoulder
        return None, None
    
    def detect_fast_movement(self, finger_x, finger_y):
        """Removed - now using space bar for clicks"""
        return False
    
    def map_to_screen(self, finger_x, finger_y, square_x, square_y, square_size):
        """Map finger position in square to screen coordinates"""
        # Calculate relative position in square (0-1)
        rel_x = (finger_x - square_x) / square_size
        rel_y = (finger_y - square_y) / square_size
        
        # Clamp to square boundaries
        rel_x = max(0, min(1, rel_x))
        rel_y = max(0, min(1, rel_y))
        
        # Map to screen coordinates
        screen_x = int(rel_x * self.screen_width)
        screen_y = int(rel_y * self.screen_height)
        
        return screen_x, screen_y
    
    def draw_control_square(self, frame, square_x, square_y, square_size):
        """Draw the lime green control square"""
        # Draw square outline
        cv2.rectangle(frame, (square_x, square_y), 
                     (square_x + square_size, square_y + square_size), 
                     (0, 255, 0), 2)
        
        # Add corner markers
        corner_size = 10
        corners = [
            (square_x, square_y),
            (square_x + square_size, square_y),
            (square_x, square_y + square_size),
            (square_x + square_size, square_y + square_size)
        ]
        
        for corner in corners:
            cv2.rectangle(frame, corner, 
                         (corner[0] + corner_size, corner[1] + corner_size), 
                         (0, 255, 0), -1)
    
    def run(self):
        """Main loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Hand Gesture Mouse Control Started")
        print("- Position your finger within the green square to control mouse")
        print("- Press SPACE to click at current mouse position")
        print("- Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hands and pose
            hand_results = self.hands.process(rgb_frame)
            pose_results = self.pose.process(rgb_frame)
            
            # Get shoulder positions to position the control square
            left_shoulder, right_shoulder = self.get_shoulder_positions(pose_results.pose_landmarks)
            
            if left_shoulder and right_shoulder:
                # Position square based on shoulders
                shoulder_center_x = int((left_shoulder.x + right_shoulder.x) / 2 * frame_width)
                shoulder_center_y = int((left_shoulder.y + right_shoulder.y) / 2 * frame_height)
                
                # Adjust square position relative to shoulders
                self.square_x = shoulder_center_x - self.square_size // 2
                self.square_y = shoulder_center_y - self.square_size // 2
                
                # Keep square within frame bounds
                self.square_x = max(0, min(frame_width - self.square_size, self.square_x))
                self.square_y = max(0, min(frame_height - self.square_size, self.square_y))
            
            # Draw control square
            self.draw_control_square(frame, self.square_x, self.square_y, self.square_size)
            
            # Process hand landmarks
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    # Draw hand skeleton on the frame
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Get finger tip position
                    finger_x, finger_y = self.get_finger_tip_position(hand_landmarks)
                    
                    if finger_x is not None and finger_y is not None:
                        # Convert to pixel coordinates
                        finger_pixel_x = int(finger_x * frame_width)
                        finger_pixel_y = int(finger_y * frame_height)
                        
                        # Draw finger tip
                        cv2.circle(frame, (finger_pixel_x, finger_pixel_y), 10, (0, 0, 255), -1)
                        
                        # Check if finger is within control square
                        if (self.square_x <= finger_pixel_x <= self.square_x + self.square_size and
                            self.square_y <= finger_pixel_y <= self.square_y + self.square_size):
                            
                            # Map finger position to screen coordinates
                            screen_x, screen_y = self.map_to_screen(
                                finger_pixel_x, finger_pixel_y,
                                self.square_x, self.square_y, self.square_size
                            )
                            
                            # Move mouse to corresponding screen position
                            pyautogui.moveTo(screen_x, screen_y)
                            
                            # Store current mouse position for clicking
                            self.last_mouse_x = screen_x
                            self.last_mouse_y = screen_y
                            
                            # Draw current mouse position indicator
                            cv2.circle(frame, (finger_pixel_x, finger_pixel_y), 15, (255, 0, 0), 2)
                            
                            # Show screen coordinates
                            cv2.putText(frame, f"Screen: ({screen_x}, {screen_y})", 
                                       (finger_pixel_x + 20, finger_pixel_y - 20),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw pose landmarks if available
            if pose_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            # Add instructions
            cv2.putText(frame, "Position finger in green square to control mouse", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press SPACE to click at current position", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Hand Gesture Mouse Control', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            # Exit on 'q' press
            if key == ord('q'):
                break
            
            # Click on space press
            if key == ord(' '):
                if hasattr(self, 'last_mouse_x') and hasattr(self, 'last_mouse_y'):
                    pyautogui.click(self.last_mouse_x, self.last_mouse_y)
                    print(f"Click performed at screen position: ({self.last_mouse_x}, {self.last_mouse_y})")
                else:
                    print("No mouse position available - move finger in green square first")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = HandGestureMouseControl()
    controller.run()