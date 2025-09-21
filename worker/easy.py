import time
import mss
import numpy as np
import cv2
import easyocr

def screenshot_screen():
    """Takes a screenshot of the entire screen and returns it as a NumPy array (BGR)."""
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # full screen
        img = np.array(sct.grab(monitor))
        # Convert BGRA to BGR
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

def run_easyocr(image):
    """Run EasyOCR on the screenshot and return detected text with bounding boxes."""
    reader = easyocr.Reader(['en'], gpu=False)  # set gpu=True if you have CUDA
    results = reader.readtext(image)

    detected_items = []
    for bbox, text, confidence in results:
        detected_items.append({
            "text": text,
            "confidence": confidence,
            "bbox": bbox  # 4 points (x,y) for the bounding box
        })
    return detected_items

def main():
    print("You have 3 seconds to open the window...")
    time.sleep(3)

    img = screenshot_screen()
    detected = run_easyocr(img)

    print("\nDetected items:")
    for item in detected:
        print(f"Text: {item['text']} | Confidence: {item['confidence']:.2f} | BBox: {item['bbox']}")

if __name__ == "__main__":
    main()
