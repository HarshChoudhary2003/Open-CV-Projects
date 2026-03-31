"""
add_known_face.py
CLI utility to enrol a person's face into the known-faces library.

Usage:
    python add_known_face.py --name "Harsh Choudhary" --image path/to/photo.jpg
    python add_known_face.py --capture --name "Harsh Choudhary"
"""

import argparse
import os
import sys
import shutil
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.settings import KNOWN_FACES_DIR


def enrol_from_file(name: str, src: str) -> None:
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    safe_name = name.strip().replace(" ", "_")
    ext       = os.path.splitext(src)[-1].lower() or ".jpg"
    dest      = os.path.join(KNOWN_FACES_DIR, f"{safe_name}{ext}")
    shutil.copy2(src, dest)
    print(f"✅ Enrolled '{name}' → {dest}")


def enrol_from_camera(name: str) -> None:
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera.")
        return

    print("📷 Camera open. Press SPACE to capture, ESC to cancel.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        display = frame.copy()
        cv2.putText(display, "SPACE = capture   ESC = cancel",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 130), 2)
        cv2.imshow("Enrol Face", display)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:   # SPACE
            safe_name = name.strip().replace(" ", "_")
            dest      = os.path.join(KNOWN_FACES_DIR, f"{safe_name}.jpg")
            cv2.imwrite(dest, frame)
            print(f"✅ Captured and enrolled '{name}' → {dest}")
            break
        elif key == 27:  # ESC
            print("Cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrol a known face")
    parser.add_argument("--name",    required=True, help="Person's full name")
    parser.add_argument("--image",   help="Path to an existing photo")
    parser.add_argument("--capture", action="store_true",
                        help="Capture from webcam instead of supplying a file")
    args = parser.parse_args()

    if args.capture:
        enrol_from_camera(args.name)
    elif args.image:
        enrol_from_file(args.name, args.image)
    else:
        parser.error("Supply either --image or --capture")
