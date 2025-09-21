import os
import pyautogui
import base64
from openai import OpenAI
import numpy as np
import cv2
from easy import screenshot_screen, run_easyocr

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def clean_messages_for_new_image(messages):
    cleaned_messages = []
    for message in messages:
        if message["role"] in ["user", "assistant"]:
            if (message["role"] == "user" and 
                isinstance(message["content"], list) and 
                any(item.get("type") == "image_url" for item in message["content"])):
                text_parts = [item["text"] for item in message["content"] if item.get("type") == "text"]
                text_summary = " ".join(text_parts) if text_parts else "Previous screenshot analysis"
                cleaned_messages.append({
                    "role": "user", 
                    "content": f"[Previous screenshot analysis: {text_summary[:100]}...]"
                })
            else:
                cleaned_messages.append(message)
    return cleaned_messages

import pyautogui

def scan_icons(icons_folder="icons", confidence=0.7):
    """Scan the screen for known icons and return dict of {icon_name: (x, y)}"""
    present_icons = {}
    for folder in os.listdir(icons_folder):
        folder_path = os.path.join(icons_folder, folder)
        if not os.path.isdir(folder_path):
            continue
        for img_file in os.listdir(folder_path):
            if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img_path = os.path.join(folder_path, img_file)
            try:
                # print(f"Scanning for {img_path}")
                location = pyautogui.locateOnScreen(img_path, confidence=confidence)
                if location:
                    center = pyautogui.center(location)
                    # print(f"Found {folder} at {center}")
                    present_icons[folder] = (center.x, center.y)
                    break  # stop after first successful match in this folder
                else:
                    print(f"X {img_path} not found")
            except pyautogui.ImageNotFoundException:
                # This is normal if the icon is not visible
                # print(f"X {img_path} not found (ImageNotFoundException)")
                pass
            except Exception as e:
                # print(f"E: Error scanning {img_path}: {e}")
                pass
    return present_icons


def get_element_and_first_guess(client, goal, messages):
    # Plain screenshot
    screenshot = pyautogui.screenshot()
    screenshot.save("shot_one.png")
    base64_image = encode_image("shot_one.png")

    # Scan for icons
    icons_found = scan_icons("icons")
    print(f"Icons found: {list(icons_found.keys())}")

    # Run OCR (EasyOCR)
    img_np = screenshot_screen()
    text_elements = run_easyocr(img_np)
    print("Text elements found:", [t["text"] for t in text_elements])

    cleaned_messages = clean_messages_for_new_image(messages)
    cleaned_messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": (
                f"Here is the current screen. "
                f"Our end goal is: '{goal}'. "
                f"You may need to do some other actions first to reach the goal. "
                f"Here is a list of icons detected on screen: {list(icons_found.keys())}. "
                f"Here is a list of text elements detected on screen: "
                f"{[t['text'] for t in text_elements]}. "
                f"Please do TWO things in your response: "
                f"1. Identify which element I should click next to progress toward this goal "
                f"2. Choose an action for this step: click, double-click, right-click, or type."
                f"Respond STRICTLY in this format: action: <action>, \"<element description>\""
                f"If the action is type, provide the text like this: action: type(\"text to be typed\"), \"<element description>\""
            )},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        ]
    })

    print(cleaned_messages[-1]["content"][0]["text"])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=cleaned_messages
    )

    print("GPT response:", response.choices[0].message.content)

    messages.append({
        "role": "user",
        "content": f"[Screenshot analysis for goal: {goal}, detected icons: {list(icons_found.keys())}]"
    })
    messages.append({
        "role": response.choices[0].message.role,
        "content": response.choices[0].message.content
    })

    return response.choices[0].message.content.strip(), icons_found

def parse_element_from_reply(reply):
    """Extract action and element name/description"""
    try:
        action = reply.split(",")[0].split(":")[1].strip()
        if '"' in reply:
            element = reply.split('"')[1]
        else:
            element = "unknown"
        return action, element
    except Exception as e:
        print(f"Could not parse reply: {e}")
        return None, "unknown element"

def click_target(goal, messages):
    reply, icons_found = get_element_and_first_guess(client, goal, messages)
    print("GPT combined response:", reply)
    action, element = parse_element_from_reply(reply)

    # Match GPT's element to detected icons
    target_coords = None
    for icon_name, coords in icons_found.items():
        if icon_name.lower() in element.lower():
            target_coords = coords
            break

    if not target_coords:
        print(f"Element '{element}' not found among detected icons: {list(icons_found.keys())}")
        return

    center_x, center_y = target_coords

    # Execute action
    if "type" in action.lower():
        start = action.find('("') + 2
        end = action.rfind('")')
        text_to_type = action[start:end] if start != -1 and end != -1 else ""
        pyautogui.moveTo(center_x, center_y)
        pyautogui.click()
        pyautogui.write(text_to_type, interval=0.05)
    elif "double-click" in action.lower():
        pyautogui.moveTo(center_x, center_y)
        pyautogui.doubleClick()
    elif "right-click" in action.lower():
        pyautogui.moveTo(center_x, center_y)
        pyautogui.rightClick()
    else:
        pyautogui.moveTo(center_x, center_y)
        pyautogui.click()

def tell_gpt_the_goal(goal, messages):
    messages.append({
        "role": "user",
        "content": f"You will help achieve the following goal by clicking or typing on my screen: '{goal}'. "
                   f"To achieve the goal, which may be to click something, type something, or open something, "
                   f"you may have to click or close other things first. You will be allowed several actions, "
                   f"but first choose an action to perform. Each action will be determined by you."
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
    messages = []
    goal = input("Enter your goal to be achieved by gpt clicking on your screen >> ")
    tell_gpt_the_goal(goal, messages)
    print(messages[0]["content"])
    print(messages[1]["content"])

    for i in range(10):
        messages = messages[-20:] # keep last 20 messages to avoid context overflow
        click_target(goal, messages)
        print(len(messages), "messages in total")

if __name__ == "__main__":
    client = OpenAI()
    main()
