import cv2
import os
import glob

# Global variables for mouse callback
drawing = False
ix, iy = -1, -1
cx, cy = -1, -1
temp_boxes = []

def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, cx, cy, temp_boxes
    
    # Left mouse button click down - start drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        cx, cy = x, y

    # Mouse drag
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cx, cy = x, y

    # Left mouse button release - finish drawing box
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Ensure box has some size
        if abs(x - ix) > 5 and abs(y - iy) > 5:
            # Save raw pixel coordinates: [x_min, y_min, x_max, y_max]
            x_min = min(ix, x)
            y_min = min(iy, y)
            x_max = max(ix, x)
            y_max = max(iy, y)
            temp_boxes.append((x_min, y_min, x_max, y_max))

def load_annotations(txt_path, img_w, img_h):
    boxes = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, x_c, y_c, w, h = map(float, parts)
                    # Convert normalized YOLO to pixel coords
                    x_min = int((x_c - w/2) * img_w)
                    y_min = int((y_c - h/2) * img_h)
                    x_max = int((x_c + w/2) * img_w)
                    y_max = int((y_c + h/2) * img_h)
                    boxes.append((x_min, y_min, x_max, y_max))
    return boxes

def save_annotations(txt_path, boxes, img_w, img_h):
    with open(txt_path, 'w') as f:
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            # Calculate normalized YOLO format: class_id x_center y_center width height
            x_center = ((x_min + x_max) / 2) / img_w
            y_center = ((y_min + y_max) / 2) / img_h
            w = (x_max - x_min) / img_w
            h = (y_max - y_min) / img_h
            
            # Clamp values between 0.0 and 1.0 to avoid YOLO validation errors
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
            
            # Single class bottle (index 0)
            f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
    print(f"Saved: {txt_path} with {len(boxes)} boxes.")

def main():
    global temp_boxes, drawing, ix, iy, cx, cy
    
    img_dir = os.path.join("data", "temp_captured")
    if not os.path.exists(img_dir):
        print(f"Error: Directory {img_dir} does not exist. Run capture_images.py first!")
        return

    # Get sorted list of images
    img_extensions = ["*.jpg", "*.jpeg", "*.png"]
    img_paths = []
    for ext in img_extensions:
        img_paths.extend(glob.glob(os.path.join(img_dir, ext)))
    img_paths = sorted(img_paths)

    if not img_paths:
        print(f"No images found in {img_dir}. Capturing images first using capture_images.py is recommended.")
        return

    print("=== Custom Bounding Box Annotation Tool ===")
    print("Commands:")
    print("  - [Click & Drag] with mouse to draw a box.")
    print("  - [S]: Save current annotations (creates/updates YOLO label file).")
    print("  - [D] / [Right Arrow]: Move to NEXT image (auto-saves).")
    print("  - [A] / [Left Arrow]: Move to PREVIOUS image (auto-saves).")
    print("  - [C]: Clear all boxes on the current image.")
    print("  - [Q] or [ESC]: Save current image and quit tool.")
    print("Note: To mark a negative sample (no bottle), draw no boxes and press 'S' or navigate away.")
    print("===========================================")

    cv2.namedWindow("Annotator")
    cv2.setMouseCallback("Annotator", mouse_callback)

    idx = 0
    img_w, img_h = 640, 480  # Defaults
    
    # Load first image's annotations
    current_img_path = img_paths[idx]
    current_txt_path = os.path.splitext(current_img_path)[0] + ".txt"
    img = cv2.imread(current_img_path)
    if img is not None:
        img_h, img_w, _ = img.shape
    temp_boxes = load_annotations(current_txt_path, img_w, img_h)
    
    prev_idx = idx

    while True:
        # If image index changed, load new image and its annotations
        if idx != prev_idx:
            # Auto-save previous image annotations before switching
            prev_img_path = img_paths[prev_idx]
            prev_txt_path = os.path.splitext(prev_img_path)[0] + ".txt"
            save_annotations(prev_txt_path, temp_boxes, img_w, img_h)
            
            # Load new image
            current_img_path = img_paths[idx]
            current_txt_path = os.path.splitext(current_img_path)[0] + ".txt"
            img = cv2.imread(current_img_path)
            if img is not None:
                img_h, img_w, _ = img.shape
            temp_boxes = load_annotations(current_txt_path, img_w, img_h)
            prev_idx = idx

        if img is None:
            print(f"Error reading image: {img_paths[idx]}")
            idx = (idx + 1) % len(img_paths)
            continue

        # Create copy for rendering
        display_img = img.copy()

        # Draw already saved boxes in Green
        for box in temp_boxes:
            cv2.rectangle(display_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(display_img, "bottle", (box[0], box[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw box currently being dragged in Red
        if drawing:
            cv2.rectangle(display_img, (ix, iy), (cx, cy), (0, 0, 255), 2)

        # Draw HUD/Instructions
        hud_bg = display_img.copy()
        cv2.rectangle(hud_bg, (0, 0), (640, 45), (0, 0, 0), -1)
        cv2.addWeighted(hud_bg, 0.6, display_img, 0.4, 0, display_img)
        
        status = f"[{idx+1}/{len(img_paths)}] {os.path.basename(current_img_path)}"
        annotated_status = "ANNOTATED" if os.path.exists(current_txt_path) else "UNANNOTATED"
        status_color = (0, 255, 0) if os.path.exists(current_txt_path) else (0, 0, 255)
        
        cv2.putText(display_img, status, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_img, annotated_status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        cv2.putText(display_img, "S:Save | D:Next | A:Prev | C:Clear | Q:Quit", (250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Annotator", display_img)
        
        # Capture keystrokes
        # Wait 10ms
        key = cv2.waitKey(10) & 0xFF
        
        # Save annotations
        if key == ord('s'):
            save_annotations(current_txt_path, temp_boxes, img_w, img_h)
            
        # Move NEXT (d or right arrow - waitKey arrow codes can vary, so 'd' is primary)
        elif key == ord('d') or key == 83:  # 83 is typical VK_RIGHT in OpenCV
            idx = (idx + 1) % len(img_paths)
            
        # Move PREVIOUS (a or left arrow - waitKey arrow codes can vary, so 'a' is primary)
        elif key == ord('a') or key == 81:  # 81 is typical VK_LEFT in OpenCV
            idx = (idx - 1) % len(img_paths)
            
        # Clear current boxes
        elif key == ord('c'):
            temp_boxes = []
            print("Cleared boxes for current image.")
            
        # Quit
        elif key == ord('q') or key == 27:
            # Auto-save current annotations before exit
            save_annotations(current_txt_path, temp_boxes, img_w, img_h)
            print("Exiting annotator.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
