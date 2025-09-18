import cv2
import numpy as np
import pyautogui

def main():
    # Get screen size
    screen_width, screen_height = pyautogui.size()

    # Take a screenshot (PIL image)
    screenshot = pyautogui.screenshot()

    # Convert PIL to OpenCV (numpy array, BGR color order)
    img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    # Number of divisions
    n_lines = 50

    # Spacing between lines
    x_step = screen_width // n_lines
    y_step = screen_height // n_lines

    color = (0, 255, 0)  # Green
    thickness = 1        # Very thin line

    # Draw vertical lines
    for x in range(0, screen_width, x_step):
        cv2.line(img, (x, 0), (x, screen_height), color, thickness)

    # Draw horizontal lines
    for y in range(0, screen_height, y_step):
        cv2.line(img, (0, y), (screen_width, y), color, thickness)

    # Save the image
    cv2.imwrite("shot.png", img)
    print("Saved screenshot with grid as shot.png")

    # Show fullscreen window
    cv2.namedWindow("Grid", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Grid", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Grid", img)

    print("Displaying for 3 seconds...")
    cv2.waitKey(3000)  # Wait 3000 ms = 3 seconds

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
