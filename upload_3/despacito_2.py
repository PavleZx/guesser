import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math
from collections import deque

class HandGestureMouseControl:
    def __init__(self):
        # ADJUSTMENT VARIABLE - Change this to make the detection rectangle bigger/smaller
        self.square_multiplier = 1.5  # 1.0 = original size, 1.5 = 50% bigger, 0.8 = 20% smaller
        
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
        
        # Control rectangle parameters (base size, will be multiplied by square_multiplier)
        base_width = 250
        base_height = 180
        self.square_width = int(base_width * self.square_multiplier)
        self.square_height = int(base_height * self.square_multiplier)
        self.square_x = 50
        self.square_y = 50
        
        # Smoothing parameters
        self.smoothing_frames = 5  # Number of frames to average
        self.position_buffer = deque(maxlen=self.smoothing_frames)
        self.smoothing_factor = 0.7  # Higher = more smoothing (0-1)
        self.last_smooth_pos = None
        
        # Movement threshold to reduce jitter
        self.movement_threshold = 2  # pixels
        
        # Click detection parameters
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        # Click state tracking
        self.is_holding_click = False
        self.space_pressed = False
        
        # Mouse movement settings
        pyautogui.PAUSE = 0  # Remove any pause between mouse movements
        
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
    
    def smooth_position(self, new_pos):
        """Apply smoothing to reduce jitter and create fluid movement"""
        if new_pos is None:
            return None
            
        # Add new position to buffer
        self.position_buffer.append(new_pos)
        
        # Calculate weighted average with exponential smoothing
        if self.last_smooth_pos is None:
            smoothed_pos = new_pos
        else:
            # Exponential smoothing
            smoothed_pos = (
                self.smoothing_factor * self.last_smooth_pos[0] + (1 - self.smoothing_factor) * new_pos[0],
                self.smoothing_factor * self.last_smooth_pos[1] + (1 - self.smoothing_factor) * new_pos[1]
            )
        
        # Additional averaging with buffer
        if len(self.position_buffer) >= 2:
            avg_x = sum(pos[0] for pos in self.position_buffer) / len(self.position_buffer)
            avg_y = sum(pos[1] for pos in self.position_buffer) / len(self.position_buffer)
            
            # Blend exponential smoothing with buffer averaging
            smoothed_pos = (
                0.7 * smoothed_pos[0] + 0.3 * avg_x,
                0.7 * smoothed_pos[1] + 0.3 * avg_y
            )
        
        # Apply movement threshold to reduce micro-movements
        if self.last_smooth_pos is not None:
            distance = math.sqrt(
                (smoothed_pos[0] - self.last_smooth_pos[0])**2 + 
                (smoothed_pos[1] - self.last_smooth_pos[1])**2
            )
            
            if distance < self.movement_threshold:
                smoothed_pos = self.last_smooth_pos
        
        self.last_smooth_pos = smoothed_pos
        return smoothed_pos
    
    def map_to_screen(self, finger_x, finger_y, square_x, square_y, square_width, square_height):
        """Map finger position in rectangle to screen coordinates"""
        # Calculate relative position in rectangle (0-1)
        rel_x = (finger_x - square_x) / square_width
        rel_y = (finger_y - square_y) / square_height
        
        # Clamp to rectangle boundaries
        rel_x = max(0, min(1, rel_x))
        rel_y = max(0, min(1, rel_y))
        
        # Map to screen coordinates
        screen_x = int(rel_x * self.screen_width)
        screen_y = int(rel_y * self.screen_height)
        
        return screen_x, screen_y
    
    def draw_control_square(self, frame, square_x, square_y, square_width, square_height):
        """Draw the lime green control rectangle"""
        # Draw rectangle outline
        cv2.rectangle(frame, (square_x, square_y), 
                     (square_x + square_width, square_y + square_height), 
                     (0, 255, 0), 2)
        
        # Add corner markers
        corner_size = 10
        corners = [
            (square_x, square_y),
            (square_x + square_width, square_y),
            (square_x, square_y + square_height),
            (square_x + square_width, square_y + square_height)
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
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Hand Gesture Mouse Control Started")
        print(f"Detection rectangle size: {self.square_width}x{self.square_height} (multiplier: {self.square_multiplier})")
        print("- Position your finger within the green rectangle to control mouse")
        print("- Press SPACE to left click (hold for drag/scroll)")
        print("- Press 'M' to right click")
        print("- Press 'q' to quit")
        print("- To adjust rectangle size, change 'square_multiplier' at the top of the code")
        
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
                
                # Adjust rectangle position relative to shoulders
                self.square_x = shoulder_center_x - self.square_width // 2
                self.square_y = shoulder_center_y - self.square_height // 2
                
                # Keep rectangle within frame bounds
                self.square_x = max(0, min(frame_width - self.square_width, self.square_x))
                self.square_y = max(0, min(frame_height - self.square_height, self.square_y))
            
            # Draw control rectangle
            self.draw_control_square(frame, self.square_x, self.square_y, self.square_width, self.square_height)
            
            # Initialize variables for this frame
            current_screen_pos = None
            
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
                        
                        # Check if finger is within control rectangle
                        if (self.square_x <= finger_pixel_x <= self.square_x + self.square_width and
                            self.square_y <= finger_pixel_y <= self.square_y + self.square_height):
                            
                            # Map finger position to screen coordinates
                            screen_x, screen_y = self.map_to_screen(
                                finger_pixel_x, finger_pixel_y,
                                self.square_x, self.square_y, self.square_width, self.square_height
                            )
                            
                            # Apply smoothing
                            current_screen_pos = (screen_x, screen_y)
                            smoothed_pos = self.smooth_position(current_screen_pos)
                            
                            if smoothed_pos:
                                # Move mouse to smoothed position
                                pyautogui.moveTo(smoothed_pos[0], smoothed_pos[1], duration=0)
                                
                                # Store current mouse position for clicking
                                self.last_mouse_x = int(smoothed_pos[0])
                                self.last_mouse_y = int(smoothed_pos[1])
                                
                                # Draw current mouse position indicator
                                cv2.circle(frame, (finger_pixel_x, finger_pixel_y), 15, (255, 0, 0), 2)
                                
                                # Show screen coordinates
                                cv2.putText(frame, f"Screen: ({self.last_mouse_x}, {self.last_mouse_y})", 
                                           (finger_pixel_x + 20, finger_pixel_y - 20),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # If no hand detected, gradually clear the smoothing buffer
            if current_screen_pos is None:
                if len(self.position_buffer) > 0:
                    self.position_buffer.clear()
                self.last_smooth_pos = None
            
            # Draw pose landmarks if available
            if pose_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            # Add instructions
            cv2.putText(frame, "Position finger in green rectangle to control mouse", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press SPACE to left click (hold for drag/scroll)", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'M' to right click", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show click state
            if self.is_holding_click:
                cv2.putText(frame, "HOLDING CLICK", 
                           (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Hand Gesture Mouse Control', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            # Exit on 'q' press
            if key == ord('q'):
                # Release any held clicks before quitting
                if self.is_holding_click:
                    pyautogui.mouseUp()
                    self.is_holding_click = False
                break
            
            # Handle space key for left click (toggle hold/release)
            if key == ord(' '):
                if not self.space_pressed:  # Only trigger on first press, not while holding
                    self.space_pressed = True
                    if hasattr(self, 'last_mouse_x') and hasattr(self, 'last_mouse_y'):
                        if not self.is_holding_click:
                            # Start holding click
                            pyautogui.mouseDown(self.last_mouse_x, self.last_mouse_y)
                            self.is_holding_click = True
                            print(f"Started holding click at: ({self.last_mouse_x}, {self.last_mouse_y})")
                        else:
                            # Release click
                            pyautogui.mouseUp()
                            self.is_holding_click = False
                            print(f"Released click at: ({self.last_mouse_x}, {self.last_mouse_y})")
                    else:
                        print("No mouse position available - move finger in green rectangle first")
            else:
                # Reset space pressed state when key is released
                if self.space_pressed:
                    self.space_pressed = False
            
            # Handle 'M' key for right click
            if key == ord('m') or key == ord('M'):
                if hasattr(self, 'last_mouse_x') and hasattr(self, 'last_mouse_y'):
                    pyautogui.rightClick(self.last_mouse_x, self.last_mouse_y)
                    print(f"Right click performed at: ({self.last_mouse_x}, {self.last_mouse_y})")
                else:
                    print("No mouse position available - move finger in green rectangle first")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = HandGestureMouseControl()
    controller.run()