import os
import tkinter as tk
from tkinter import simpledialog
import pyautogui
from pynput import mouse, keyboard
from PIL import ImageGrab
import time

# --- Step 1: Ask for folder name ---
time.sleep(3)  # Small delay to ensure the dialog appears on top
root = tk.Tk()
root.withdraw()
folder_name = simpledialog.askstring("Folder Name", "Enter folder name:")
if not folder_name:
    print("No folder name given. Exiting.")
    exit()

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# --- Step 2: Setup variables ---
square_width, square_height = 200, 200
save_counter = 1
mouse_pos = (500, 500)
running = True
after_id = None  # store after() so we can cancel

# --- Step 3: Tkinter overlay window ---
overlay = tk.Tk()
overlay.attributes("-transparentcolor", "white")
overlay.attributes("-topmost", True)
overlay.overrideredirect(True)
canvas = tk.Canvas(overlay, bg="white", highlightthickness=0)
canvas.pack(fill="both", expand=True)

screen_w, screen_h = pyautogui.size()
overlay.geometry(f"{screen_w}x{screen_h}+0+0")

def draw_square():
    global after_id
    canvas.delete("all")
    x, y = mouse_pos
    x1, y1 = x - square_width // 2, y - square_height // 2
    x2, y2 = x + square_width // 2, y + square_height // 2
    canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)
    after_id = overlay.after(30, draw_square)

# --- Step 4: Mouse and keyboard controls ---
def on_click(x, y, button, pressed):
    global save_counter
    if pressed and button == mouse.Button.right:
        # Hide overlay before capture
        overlay.withdraw()

        left = x - square_width // 2
        top = y - square_height // 2
        bbox = (left, top, left + square_width, top + square_height)
        screenshot = ImageGrab.grab(bbox)

        filepath = os.path.join(folder_name, f"{folder_name}_{save_counter}.png")
        screenshot.save(filepath)
        print(f" Saved {filepath}")
        save_counter += 1

        # Show overlay again
        overlay.deiconify()

def on_move(x, y):
    global mouse_pos
    mouse_pos = (x, y)

def on_press(key):
    global square_width, square_height, running, after_id
    try:
        if key.char == 'w':
            square_height += 10
        elif key.char == 's':
            square_height = max(20, square_height - 10)
        elif key.char == 'a':
            square_width += 10
        elif key.char == 'd':
            square_width = max(20, square_width - 10)
        elif key.char == 'x':
            running = False
            if after_id:
                overlay.after_cancel(after_id)
            overlay.destroy()
            os._exit(0)
            return False
    except AttributeError:
        pass

# --- Step 5: Start listeners ---
mouse_listener = mouse.Listener(on_click=on_click, on_move=on_move)
mouse_listener.start()

keyboard_listener = keyboard.Listener(on_press=on_press)
keyboard_listener.start()

# --- Step 6: Start overlay loop ---
draw_square()
overlay.mainloop()

mouse_listener.stop()
keyboard_listener.stop()
print("Program stopped.")
