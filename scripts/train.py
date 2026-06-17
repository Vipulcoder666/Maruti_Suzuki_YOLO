import os
import shutil
from ultralytics import YOLO

def main():
    # Verify data.yaml exists
    yaml_path = os.path.abspath("data.yaml")
    if not os.path.exists(yaml_path):
        print(f"Error: {yaml_path} not found. Please run prepare_dataset.py first!")
        return

    print("=== YOLOv8 CPU Training Script ===")
    print("Loading pre-trained YOLOv8n model...")
    model = YOLO("yolov8n.pt")  # Load base weights

    # Start training
    # Ryzen 5 5500 has 6 cores / 12 threads. workers=4 is optimized for speed and stability.
    print("Starting training on CPU. This will take ~10-15 minutes for 30 epochs...")
    results = model.train(
        data=yaml_path,
        epochs=30,
        batch=8,
        imgsz=640,
        device="cpu",
        workers=4,
        project="models",
        name="bottle_detector",
        exist_ok=True
    )
    
    print("\nTraining completed successfully!")
    
    # Locate best weights and copy to models/best.pt
    best_weights_src = os.path.join("models", "bottle_detector", "weights", "best.pt")
    best_weights_dst = os.path.join("models", "best.pt")
    
    if os.path.exists(best_weights_src):
        os.makedirs("models", exist_ok=True)
        shutil.copy(best_weights_src, best_weights_dst)
        print(f"Copied best model weights to target destination: {os.path.abspath(best_weights_dst)}")
    else:
        print("Warning: Could not find trained weights at", best_weights_src)

if __name__ == "__main__":
    main()
