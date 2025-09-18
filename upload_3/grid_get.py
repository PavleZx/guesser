import cv2
import numpy as np
import pyautogui
import base64
from openai import OpenAI

# === Helper functions ===
def encode_image(image_path):
    """Read an image and return base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def take_screenshot_with_grid(filename="shot.png", n_lines=50):
    """Take screenshot, draw grid, save image, and return screen size."""
    screen_width, screen_height = pyautogui.size()

    # Take screenshot (PIL image)
    screenshot = pyautogui.screenshot()

    # Convert PIL to OpenCV (numpy array, BGR color order)
    img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    # Spacing between lines
    x_step = screen_width // n_lines
    y_step = screen_height // n_lines

    color = (0, 255, 0)  # Green
    thickness = 1        # Thin line

    # Draw vertical lines
    for x in range(0, screen_width, x_step):
        cv2.line(img, (x, 0), (x, screen_height), color, thickness)

    # Draw horizontal lines
    for y in range(0, screen_height, y_step):
        cv2.line(img, (0, y), (screen_width, y), color, thickness)

    # Save image
    cv2.imwrite(filename, img)
    print(f" Saved screenshot with grid as {filename}")

    return screen_width, screen_height

def highlight_square(reply, filename="shot.png", n_lines=50):
    """Highlight the chosen square from GPT's reply and overwrite the image."""
    try:
        # Parse GPT response like "column=12, row=34"
        parts = reply.replace(" ", "").split(",")
        col = int(parts[0].split("=")[1])
        row = int(parts[1].split("=")[1])

        img = cv2.imread(filename)
        screen_width, screen_height = pyautogui.size()
        x_step = screen_width // n_lines
        y_step = screen_height // n_lines

        x1, y1 = col * x_step, row * y_step
        x2, y2 = x1 + x_step, y1 + y_step

        # Draw filled rectangle (transparent red overlay)
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
        img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

        cv2.imwrite(filename, img)

        # Show fullscreen briefly
        cv2.namedWindow("Highlighted Grid", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Highlighted Grid", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Highlighted Grid", img)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Could not highlight square: {e}")


# === Main program ===
def main():
    client = OpenAI()

    # 1. Ask user what element they want
    element = input("Enter the element you want to locate on screen: ")

    # 2. Take screenshot with grid
    take_screenshot_with_grid("shot.png")

    # 3. Prepare image as base64
    base64_image = encode_image("shot.png")

    # 4. Send to GPT
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": f"Look at the image of my screen with a 50x50 grid overlay. "
                                      f"Please return ONLY the column and row of the grid square "
                                      f"in which the element '{element}' is located. "
                                      f"Respond strictly in the format: column=<number>, row=<number>"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            }
        ]
    }]

    response = client.chat.completions.create(
        model="gpt-4o-mini",   # or "gpt-4o" if you prefer
        messages=messages
    )

    reply = response.choices[0].message.content.strip()

    # 5. Print GPT response (the row/column)
    print("\n GPT detected position:")
    print(reply)
    
    # === Highlight the chosen square from GPT ===
    try:
        # Parse GPT response like "column=12, row=34"
        parts = reply.replace(" ", "").split(",")
        col = int(parts[0].split("=")[1])
        row = int(parts[1].split("=")[1])

        # Reload the screenshot
        img = cv2.imread("shot.png")

        # Grid setup
        n_lines = 50
        screen_width, screen_height = pyautogui.size()
        x_step = screen_width // n_lines
        y_step = screen_height // n_lines

        # Compute bounding box of the chosen cell
        x1 = col * x_step
        y1 = row * y_step
        x2 = x1 + x_step
        y2 = y1 + y_step

        # Draw filled rectangle (transparent red overlay)
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
        alpha = 0.3
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # Show fullscreen with highlight

        cv2.imwrite("shot.png", img)

        cv2.namedWindow("Highlighted Grid", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Highlighted Grid", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Highlighted Grid", img)

        print("Highlighted grid will close in 3 seconds...")
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f" Could not highlight square: {e}")

        # === Ask GPT again to confirm/correct the chosen square ===
    base64_image = encode_image("shot.png")

    correction_messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": f"Here is the screenshot of my screen with a 50x50 grid overlay. "
                                      f"The red square marks the column and row you chose earlier for the element '{element}'. "
                                      f"Please check carefully if the red square is correct. "
                                      f"If it is correct, respond strictly with the same format column=<number>, row=<number>. "
                                      f"If it is incorrect, respond with the corrected column and row, also strictly in that format."},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            }
        ]
    }]

    correction_response = client.chat.completions.create(
        model="gpt-4o-mini",   # or "gpt-4o"
        messages=correction_messages
    )

    corrected_reply = correction_response.choices[0].message.content.strip()

    print("\nGPT corrected/confirmed position:")
    print(corrected_reply)

    reply = corrected_reply

    # === Highlight the chosen square from GPT ===
    try:
        # Parse GPT response like "column=12, row=34"
        parts = reply.replace(" ", "").split(",")
        col = int(parts[0].split("=")[1])
        row = int(parts[1].split("=")[1])

        # Reload the screenshot
        img = cv2.imread("shot.png")

        # Grid setup
        n_lines = 50
        screen_width, screen_height = pyautogui.size()
        x_step = screen_width // n_lines
        y_step = screen_height // n_lines

        # Compute bounding box of the chosen cell
        x1 = col * x_step
        y1 = row * y_step
        x2 = x1 + x_step
        y2 = y1 + y_step

        # Draw filled rectangle (transparent red overlay)
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
        alpha = 0.3
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # Show fullscreen with highlight

        cv2.imwrite("shot.png", img)

        cv2.namedWindow("Highlighted Grid", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Highlighted Grid", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Highlighted Grid", img)

        print("Highlighted grid will close in 3 seconds...")
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f" Could not highlight square: {e}")

    # posalji mu grid sa crvenim kvadratom
    # i pitaj da provjeri da je to točno

if __name__ == "__main__":
    main()
