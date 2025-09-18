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
    """Send screenshot and ask GPT for the action type, element, and coordinates."""
    base64_image = encode_image("shot_one.png")
    
    # Clean previous images from message history
    cleaned_messages = clean_messages_for_new_image(messages)

    cleaned_messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": f"ANALYZE this screenshot with 50x50 grid overlay to achieve goal: '{goal}'. "
                                      f"YOU MUST look at the image and determine the next action and coordinates yourself. "
                                      f"DO NOT ask me for coordinates - YOU analyze the screenshot and provide them. "
                                      f"Available actions: click, doubleclick, rightclick, type "
                                      f"RESPOND ONLY in this EXACT format (no extra text): "
                                      f"For click/doubleclick/rightclick: <action>, column=<number>, row=<number>, \"<element description>\" "
                                      f"For typing: type > \"<text to type>\", column=<number>, row=<number>, \"<element description>\" "
                                      f"Examples: "
                                      f"click, column=47, row=2, \"Minimize button\" "
                                      f"doubleclick, column=25, row=35, \"Discord icon\" "
                                      f"type > \"hello\", column=30, row=20, \"Search box\" "
                                      f"ANALYZE THE SCREENSHOT AND GIVE COORDINATES NOW:"},
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

def get_guess_again(client, action_info, messages):
    """Send the screenshot with red square and ask GPT to confirm/correct the action and coordinates."""
    base64_image = encode_image("shot.png")
    
    # Clean previous images from message history
    cleaned_messages = clean_messages_for_new_image(messages)

    cleaned_messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": f"VERIFICATION: The red square shows your choice: {action_info['summary']} "
                                     f"LOOK at the red square position. Is it correct? "
                                     f"If YES: Respond with EXACT same format as before "
                                     f"If NO: Give corrected coordinates in same format "
                                     f"RESPOND ONLY with action and coordinates (no explanations): "
                                     f"Format: <action>, column=<number>, row=<number>, \"<element>\" "
                                     f"Or for type: type > \"<text>\", column=<number>, row=<number>, \"<element>\""},
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
        "content": f"[Verification screenshot for: {action_info['summary']}]"
    })
    messages.append({
        "role": response.choices[0].message.role,
        "content": response.choices[0].message.content
    })

    return response.choices[0].message.content.strip()

def parse_action_from_reply(reply):
    """Parse the action, coordinates, text (if any), and element from GPT's reply."""
    try:
        action_info = {
            'action': 'click',
            'text': None,
            'column': 0,
            'row': 0,
            'element': 'unknown element',
            'summary': 'unknown action'
        }
        
        # Handle type action with text
        if 'type >' in reply.lower():
            # Format: type > "text to type", column=X, row=Y, "element"
            parts = reply.split('>')
            action_info['action'] = 'type'
            
            # Extract the text to type (between quotes after >)
            text_part = parts[1].split(',')[0].strip()
            if '"' in text_part:
                action_info['text'] = text_part.split('"')[1]
            
            # Extract coordinates from the rest
            coord_part = '>'.join(parts[1:])
            
        else:
            # Format: action, column=X, row=Y, "element"
            parts = reply.split(',')
            if len(parts) >= 3:
                action_info['action'] = parts[0].strip().lower()
                coord_part = reply
        
        # Extract column and row
        if 'column=' in reply:
            col_start = reply.find('column=') + 7
            col_end = reply.find(',', col_start)
            if col_end == -1:
                col_end = reply.find(' ', col_start)
            if col_end == -1:
                col_end = len(reply)
            action_info['column'] = int(reply[col_start:col_end].strip())
        
        if 'row=' in reply:
            row_start = reply.find('row=') + 4
            row_end = reply.find(',', row_start)
            if row_end == -1:
                row_end = reply.find(' ', row_start)
            if row_end == -1:
                row_end = len(reply)
            action_info['row'] = int(reply[row_start:row_end].strip())
        
        # Extract element description (last quoted part)
        if '"' in reply:
            quotes = reply.split('"')
            if len(quotes) >= 2:
                action_info['element'] = quotes[-2]
        
        # Create summary
        if action_info['action'] == 'type':
            action_info['summary'] = f"type '{action_info['text']}' in {action_info['element']}"
        else:
            action_info['summary'] = f"{action_info['action']} on {action_info['element']}"
            
        return action_info
        
    except Exception as e:
        print(f"Could not parse action from reply: {e}")
        return {
            'action': 'click',
            'text': None,
            'column': 0,
            'row': 0,
            'element': 'unknown element',
            'summary': 'unknown action'
        }

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
    
    return screen_width, screen_height

def highlight_square(action_info, filename="shot.png", n_lines=50):
    """Highlight the chosen square and show action type."""
    try:
        col = action_info['column']
        row = action_info['row']

        img = cv2.imread(filename)
        screen_width, screen_height = pyautogui.size()
        x_step = screen_width // n_lines
        y_step = screen_height // n_lines

        x1, y1 = col * x_step, row * y_step
        x2, y2 = x1 + x_step, y1 + y_step

        # Choose color based on action type
        if action_info['action'] == 'click':
            color = (0, 0, 255)  # Red
        elif action_info['action'] == 'doubleclick':
            color = (0, 165, 255)  # Orange
        elif action_info['action'] == 'rightclick':
            color = (255, 0, 255)  # Magenta
        elif action_info['action'] == 'type':
            color = (255, 255, 0)  # Cyan
        else:
            color = (0, 0, 255)  # Default red

        # Draw filled rectangle (transparent overlay)
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        img = cv2.addWeighted(overlay, 0.5, img, 0.7, 0)
        
        # Add action text above the square
        text = f"{action_info['action'].upper()}"
        if action_info['action'] == 'type' and action_info['text']:
            text += f": {action_info['text'][:20]}"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        text_color = (255, 255, 255)
        
        # Position text above the square
        text_x = max(0, x1 - 10)
        text_y = max(30, y1 - 10)
        
        cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, font_thickness)

        cv2.imwrite("shot.png", img)

        # Show fullscreen briefly
        cv2.namedWindow("Action Preview", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Action Preview", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Action Preview", img)
        cv2.waitKey(800)  # Show a bit longer to read action type
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Could not highlight square: {e}")

def perform_action(action_info):
    """Execute the specified action at the given coordinates."""
    col = action_info['column']
    row = action_info['row']
    
    screen_width, screen_height = pyautogui.size()
    x_step = screen_width // 50
    y_step = screen_height // 50

    # Calculate center of the square
    center_x = col * x_step + x_step // 2
    center_y = row * y_step + y_step // 2

    # Move mouse to position
    pyautogui.moveTo(center_x, center_y)
    
    # Perform the specified action
    if action_info['action'] == 'click':
        pyautogui.click()
        print(f"Clicked at ({center_x}, {center_y})")
        
    elif action_info['action'] == 'doubleclick':
        pyautogui.doubleClick()
        print(f"Double-clicked at ({center_x}, {center_y})")
        
    elif action_info['action'] == 'rightclick':
        pyautogui.rightClick()
        print(f"Right-clicked at ({center_x}, {center_y})")
        
    elif action_info['action'] == 'type':
        # Click first to focus, then type
        pyautogui.click()
        if action_info['text']:
            pyautogui.typewrite(action_info['text'])
            print(f"Typed '{action_info['text']}' at ({center_x}, {center_y})")
        else:
            print("No text specified for type action")
    
    else:
        # Default to click
        pyautogui.click()
        print(f"Unknown action '{action_info['action']}', defaulted to click at ({center_x}, {center_y})")

def execute_action(goal, messages):
    """Combined function that gets action and coordinates, then executes it."""
    
    # 1. Take initial screenshot with grid
    take_screenshot_with_grid("shot_one.png")

    # 2. Ask GPT for the action type, element, and coordinates
    reply = get_element_and_first_guess(client, goal, messages)
    print("GPT response:", reply)
    
    # 3. Parse the action information
    action_info = parse_action_from_reply(reply)
    print(f"Parsed action: {action_info['summary']}")

    # 4. Highlight the chosen square with action indicator
    highlight_square(action_info, "shot_one.png")

    # 5. Verification retries
    for i in range(1):  # change 1 → any number of retries
        reply = get_guess_again(client, action_info, messages)
        print(f"GPT verification {i+1}:", reply)
        action_info = parse_action_from_reply(reply)
        highlight_square(action_info, "shot_one.png")
    
    # 6. Execute the final action
    perform_action(action_info)

def tell_gpt_the_goal(goal, messages):
    """Send an initial message to GPT explaining the goal and available actions."""
    messages.append({
        "role": "user",
        "content": f"GOAL: {goal} "
                   f"IMPORTANT RULES: "
                   f"1. YOU analyze screenshots and determine coordinates - do NOT ask me for them "
                   f"2. YOU look at the 50x50 grid overlay and choose column/row numbers "
                   f"3. Always respond in EXACT format: action, column=X, row=Y, \"element\" "
                   f"4. Available actions: click, doubleclick, rightclick, type "
                   f"5. For type action: type > \"text to type\", column=X, row=Y, \"element\" "
                   f"6. After each action, verify the red square position when shown "
                   f"7. NO explanations or questions - just give direct action commands "
                   f"You will see screenshots with grid overlays. Analyze them and provide coordinates."
    }) 
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    messages.append({
        "role": response.choices[0].message.role,
        "content": response.choices[0].message.content
    })

    print("GPT has been informed about the goal and available actions")

def main():
    initial_messages = []
    goal = input("Enter your goal to be achieved by GPT performing actions on your screen >> ")

    tell_gpt_the_goal(goal, initial_messages)

    num_actions = int(input("How many actions should GPT perform? (default 4) >> ") or "4")

    for i in range(num_actions):
        print(f"\n--- Action {i+1}/{num_actions} ---")
        messages = initial_messages.copy()
        execute_action(goal, messages)
        print(f"Total messages in conversation: {len(messages)}")
        
        # Small delay between actions
        pyautogui.sleep(0.5)

if __name__ == "__main__":
    client = OpenAI()
    main()