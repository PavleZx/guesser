import pyautogui
import time

# Get screen size
screen_width, screen_height = pyautogui.size()
center_x = screen_width // 2
center_y = screen_height // 2

print("Waiting 3 minutes before clicking...")

# Wait for 3 minutes (180 seconds)
time.sleep(180)

# Click at center of screen
pyautogui.click(center_x, center_y)
print("Clicked!")