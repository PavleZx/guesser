import cv2
import numpy as np
import pyautogui
import base64
from openai import OpenAI

def encode_image(image_path):
    """Read an image and return base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def clean_messages_for_new_image(messages):
    """Remove all previous images from message history and replace with small text summaries."""
    cleaned_messages = []
    
    for message in messages:
        if message["role"] in ["user", "assistant"]:
            # If it's a user message with image content
            if (message["role"] == "user" and 
                isinstance(message["content"], list) and 
                any(item.get("type") == "image_url" for item in message["content"])):
                
                # Replace with text-only summary
                text_parts = [item["text"] for item in message["content"] if item.get("type") == "text"]
                text_summary = " ".join(text_parts) if text_parts else "Previous screenshot analysis"
                
                cleaned_messages.append({
                    "role": "user", 
                    "content": f"[Previous screenshot analysis: {text_summary[:100]}...]"
                })
            else:
                # Keep text-only messages as is
                cleaned_messages.append(message)
    
    return cleaned_messages

def get_element_and_first_guess(client, goal, messages):
    """Send screenshot and ask GPT for both the next element AND its coordinates."""
    base64_image = encode_image("shot_one.png")
    
    # Clean previous images from message history
    cleaned_messages = clean_messages_for_new_image(messages)

    cleaned_messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": f"Here is the current screen with a 50x50 grid overlay. "
                                      f"Our end goal is: '{goal}'. "
                                      f"Please do TWO things in your response: "
                                      f"1. Identify which element I should click next to progress toward this goal "
                                      f"2. Tell me the exact column and row coordinates of that element "
                                      f"Respond STRICTLY in this format: column=<number>, row=<number>, \"<element description>\""},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        ]
    })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=cleaned_messages
    )
    
    # Add both the cleaned request and response to original messages
    messages.append({
        "role": "user",
        "content": f"[Screenshot analysis for goal: {goal}]"
    })
    messages.append({
        "role": response.choices[0].message.role,
        "content": response.choices[0].message.content
    })

    return response.choices[0].message.content.strip()

def get_guess_again(client, element, messages):
    """Send the screenshot with red square and ask GPT to confirm/correct it."""
    base64_image = encode_image("shot.png")
    
    # Clean previous images from message history
    cleaned_messages = clean_messages_for_new_image(messages)

    cleaned_messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": f"Here is the screenshot of my screen with a 50x50 grid overlay. "
                                     f"Assume that the red square was placed most likely correctly the first time. "
                                      f"The red square marks the column and row you chose earlier for the element '{element}'. "
                                      f"Please check carefully if the red square is on the correct place. "
                                      f"If it is correct, you do not need to change the chosen column and row, so respond strictly with the same format column=<number>, row=<number>. "
                                      f"If it is incorrect, respond with the corrected column and row, also strictly in that format."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        ]
    })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=cleaned_messages
    )
    
    # Add both the cleaned request and response to original messages
    messages.append({
        "role": "user",
        "content": f"[Verification screenshot for element: {element}]"
    })
    messages.append({
        "role": response.choices[0].message.role,
        "content": response.choices[0].message.content
    })

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
        # Parse GPT response like "column=12, row=34" (ignoring the element description part)
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
        cv2.waitKey(500)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Could not highlight square: {e}")

def parse_element_from_reply(reply):
    """Extract the element description from the combined reply."""
    try:
        # Look for the quoted element description at the end
        if '"' in reply:
            # Find the last quoted part
            parts = reply.split('"')
            if len(parts) >= 2:
                return parts[-2]  # Get the content of the last quote
        
        # Fallback: take everything after the second comma
        parts = reply.split(',')
        if len(parts) >= 3:
            return ','.join(parts[2:]).strip()
        
        return "unknown element"
    except Exception as e:
        print(f"Could not parse element from reply: {e}")
        return "unknown element"

def click_target(goal, messages):
    """Combined function that gets element and coordinates, then clicks."""
    
    # 1. Take initial screenshot with grid
    take_screenshot_with_grid("shot_one.png")

    # 2. Ask GPT for both the element and first guess coordinates
    reply = get_element_and_first_guess(client, goal, messages)
    print("GPT combined response:", reply)
    
    # 3. Extract element description for verification steps
    element = parse_element_from_reply(reply)
    print("Extracted element:", element)

    # 4. Draw red square on the guess and save again as shot.png
    highlight_square(reply, "shot_one.png")

    # 5. Verification retries
    for i in range(1):  # change 1 → any number of retries
        reply = get_guess_again(client, element, messages)
        print(f"GPT guess {i+2}:", reply)
        highlight_square(reply, "shot_one.png")
    
    # 6. Click at the center of the last highlighted square
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

def tell_gpt_the_goal(goal, messages):
    """Send an initial message to GPT explaining the goal."""
    messages.append({
        "role": "user",
        "content": f"You will help achieve the following goal by clicking on my screen: '{goal}'. "
                   f"To achieve the goal, which may be to click on something, or open something, you may have to "
                   f"click or close some other things first. "
                   f"Each click will be determined by you. After each click, you will be shown "
                   f"an updated screenshot with a grid and your chosen square highlighted. THE CLICK WILL HAPPEN AT THE RED SQUARE THAT YOU CHOSE. "
                   f"You must confirm or correct your choice if necessary."
    }) 
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    messages.append({
        "role": response.choices[0].message.role,
        "content": response.choices[0].message.content
    })

    print("GPT has been told the goal")

def main():
    initial_messages = []
    goal = input("Enter your goal to be achieved by gpt clicking on your screen >> ")

    tell_gpt_the_goal(goal, initial_messages)

    for i in range(4):
        # 4 clicks will happen, each with verification steps
        messages = initial_messages.copy()
        click_target(goal, messages)
        print(len(messages), "messages in total")

if __name__ == "__main__":
    client = OpenAI()
    main()