import cv2
import os
import time

def main():
    # Define save directory
    save_dir = os.path.join("data", "temp_captured")
    os.makedirs(save_dir, exist_ok=True)

    # Initialize webcam
    # We try index 0, which is typically the default built-in webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Try changing the index in cv2.VideoCapture(0) to 1 or 2.")
        return

    # Set resolution to 640x480 (standard YOLO training input size)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("=== Image Capture Tool ===")
    print("Instructions:")
    print("  - Press [SPACE] to capture and save a frame.")
    print("  - Press [Q] to quit the tool.")
    print(f"Images will be saved to: {os.path.abspath(save_dir)}")
    print("==========================")

    # Count existing images in directory to resume numbering
    existing_files = [f for f in os.listdir(save_dir) if f.startswith("img_") and f.endswith(".jpg")]
    count = len(existing_files)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Clone the frame for display to overlay text
        display_frame = frame.copy()
        
        # Add visual guide text
        cv2.putText(display_frame, f"Captured: {count} images", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "SPACE: Capture  |  Q: Quit", (10, 460), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show frame
        cv2.imshow("Webcam Capture", display_frame)

        key = cv2.waitKey(1) & 0xFF
        
        # SPACE pressed to save frame
        if key == 32:
            # Generate a zero-padded filename based on count
            count += 1
            filename = f"img_{count:04d}.jpg"
            filepath = os.path.join(save_dir, filename)
            
            # Save the clean frame (without overlays)
            cv2.imwrite(filepath, frame)
            print(f"[{count:04d}] Saved: {filepath}")

        # 'Q' or Esc pressed to exit
        elif key == ord('q') or key == 27:
            print(f"Exiting. Total images captured in this session: {count - len(existing_files)}")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
