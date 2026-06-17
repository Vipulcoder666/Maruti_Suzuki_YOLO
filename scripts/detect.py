import cv2
import os
import time
from ultralytics import YOLO

def main():
    model_path = os.path.join("models", "best.pt")
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {os.path.abspath(model_path)}.")
        print("Please train your model using train.py first!")
        return

    # Load custom trained model
    print(f"Loading trained YOLOv8 model: {model_path}...")
    model = YOLO(model_path)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam. Verify that camera index (0) is correct.")
        return

    # Set camera resolution (standard matches YOLO training input)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Variables for FPS calculation
    prev_time = 0
    fps = 0.0

    # Default confidence threshold
    conf_threshold = 0.50

    print("\n=== Real-Time Bottle Detection Demo ===")
    print("Commands:")
    print("  - Press [Q] to quit the webcam demo.")
    print("  - Press [ ] ] (Right bracket) to INCREASE confidence threshold.")
    print("  - Press [ [ ] (Left bracket) to DECREASE confidence threshold.")
    print("=======================================")

    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Run model inference on frame
        # verbose=False keeps the console clean
        results = model(frame, conf=conf_threshold, device="cpu", verbose=False)
        
        # Draw detections
        annotated_frame = frame.copy()
        
        # YOLOv8 returns list of Results objects
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]

                # Draw bounding box (Green)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Label text
                label = f"{class_name} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                
                # Label background
                cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
                # Label text
                cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # FPS calculation using simple moving average
        end_time = time.time()
        time_diff = end_time - start_time
        current_fps = 1.0 / time_diff if time_diff > 0 else 0
        fps = 0.9 * fps + 0.1 * current_fps  # smooth value

        # Draw HUD background banner at top
        hud_bg = annotated_frame.copy()
        cv2.rectangle(hud_bg, (0, 0), (640, 40), (0, 0, 0), -1)
        cv2.addWeighted(hud_bg, 0.6, annotated_frame, 0.4, 0, annotated_frame)

        # Draw HUD texts
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"Conf Threshold: {conf_threshold:.2f}", (180, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated_frame, "[ / ]: Adjust Conf | Q: Quit", (400, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Show the video stream window
        cv2.imshow("Real-Time Bottle Detection", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        
        # Quit script
        if key == ord('q') or key == 27:
            print("Exiting demo.")
            break
            
        # Increase confidence threshold
        elif key == ord(']') or key == ord('}'):
            conf_threshold = min(0.95, conf_threshold + 0.05)
            print(f"Confidence threshold set to: {conf_threshold:.2f}")
            
        # Decrease confidence threshold
        elif key == ord('[') or key == ord('{'):
            conf_threshold = max(0.05, conf_threshold - 0.05)
            print(f"Confidence threshold set to: {conf_threshold:.2f}")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
