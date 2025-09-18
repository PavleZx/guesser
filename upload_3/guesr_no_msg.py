
import cv2
import numpy as np
import pyautogui
import base64
from openai import OpenAI

def encode_image(image_path):
    """Read an image and return base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_first_guess(client, element):
    """Send the initial screenshot with grid and ask GPT for the element position."""
    base64_image = encode_image("shot_one.png")

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": f"Look at the image of my screen with a 50x50 grid overlay. "
                                      f"Please return ONLY the column and row of the grid square "
                                      f"in which the element '{element}' is located. "
                                      f"Respond strictly in the format: column=<number>, row=<number>"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        ]
    }]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return response.choices[0].message.content.strip()


def get_guess_again(client, element):
    """Send the screenshot with red square and ask GPT to confirm/correct it."""
    base64_image = encode_image("shot.png")

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": f"Here is the screenshot of my screen with a 50x50 grid overlay. "
                                     f"Assume that the red suqre was placed most likely correctly the first time. "
                                      f"The red square marks the column and row you chose earlier for the element '{element}'. "
                                      f"Please check carefully if the red square is on the correct place. "
                                      f"If it is correct, you do not need to change the chosen column and row, so respond strictly with the same format column=<number>, row=<number>. "
                                      f"If it is incorrect, respond with the corrected column and row, also strictly in that format."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        ]
    }]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return response.choices[0].message.content.strip()

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
    #print(f" Saved screenshot with grid as {filename}")

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
        img = cv2.addWeighted(overlay, 0.5, img, 0.7, 0)
        

        cv2.imwrite("shot.png", img)

        # Show fullscreen briefly
        cv2.namedWindow("Highlighted Grid", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Highlighted Grid", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Highlighted Grid", img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Could not highlight square: {e}")




def click_target(element):

    # 1. Ask user what element they want
    #element = input("Enter the element you want to locate on screen: ")

    # 2. Take initial screenshot with grid
    take_screenshot_with_grid("shot_one.png")

    # 3. Ask GPT for the first guess
    reply = get_first_guess(client, element)
    print("GPT first guess:", reply)

    # 4. Draw red square on the guess and save again as shot.png
    highlight_square(reply, "shot_one.png")

    # 5. Arbitrary number of retries
    for i in range(1):  # change 2 → any number of retries
        reply = get_guess_again(client, element)
        print(f"GPT guess {i+2}:", reply)
        highlight_square(reply, "shot_one.png")
    
    # click at the center of the last highlighted square
    # Parse the last reply to get column and row
    parts = reply.replace(" ", "").split(",")
    col = int(parts[0].split("=")[1])
    row = int(parts[1].split("=")[1])

    screen_width, screen_height = pyautogui.size()
    x_step = screen_width // 50
    y_step = screen_height // 50

    # Calculate center of the square
    center_x = col * x_step + x_step // 2
    center_y = row * y_step + y_step // 2

    # Move mouse and click
    pyautogui.moveTo(center_x, center_y)
    pyautogui.click()

def tell_gpt_the_goal(goal):
    # send an initial message to gpt telling that we want to achieve some goal
    # by him clicking and that a click is made after one initial guess and 2 additional checks

    """Send an initial message to GPT explaining the goal."""
    messages = [{
        "role": "user",
        "content": f"You will help achieve the following goal by clicking on my screen: '{goal}'. "
                   f"To achieve the goal, which may be to click on something, or open something, you may have to"
                   f"Click or close some other things first."
                   f"Each click will be determined by you. After each click, you will be shown "
                   f"an updated screenshot with a grid and your chosen square highlighted. THE CLICK WILL HAPEN AT THE RED SQUARE THAT YOU CHOSE. "
                   f"You must confirm or correct your choice up to 2 times per click if necessary."
    }]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    print("GPT has been told the goal")

def ask_gpt_for_the_element(goal):
    # with the aims to get to the final goal, and the current content on the screen
    # get the next objective to be clicked by gpt

    """Ask GPT what element to click next to achieve the goal."""
    # Take screenshot with grid
    take_screenshot_with_grid("shot.png")
    base64_image = encode_image("shot.png")

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": f"Here is the current screen with a 50x50 grid overlay. "
                                      f"Our end goal is: '{goal}'. "
                                      f"Please tell me which element I should click next on my screen to progress "
                                      f"toward this goal. A button or an element on the screen that is clickable."
                                      f"Return ONLY the element description in plain text."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        ]
    }]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    element = response.choices[0].message.content.strip()
    print(f"GPT next element: {element}")
    return element   


def main():


    goal = input("Enter your goal to be achieved by gpt clicking on your screen >>")

    tell_gpt_the_goal(goal)

    for i in range(4):
        # 4 click will happen for each click 2 reguesses will be made by gpt
        element = ask_gpt_for_the_element(goal)
        click_target(element)

if __name__ == "__main__":

    client = OpenAI()

    main()

