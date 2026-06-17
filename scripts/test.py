import os
from ultralytics import YOLO

def main():
    # Model path
    model_path = os.path.join("models", "best.pt")
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {os.path.abspath(model_path)}.")
        print("Please run train.py first to train the model!")
        return

    # Dataset path
    yaml_path = os.path.abspath("data.yaml")
    if not os.path.exists(yaml_path):
        print(f"Error: {yaml_path} not found. Please run prepare_dataset.py first!")
        return

    print("=== YOLOv8 Model Evaluation ===")
    print(f"Loading trained model: {model_path}")
    model = YOLO(model_path)

    # Run validation
    print("Evaluating model performance on the validation split...")
    metrics = model.val(
        data=yaml_path,
        split="val",
        device="cpu",
        project="models",
        name="evaluation",
        exist_ok=True
    )

    # Print summary of key metrics
    print("\n" + "="*40)
    print("           METRIC SUMMARY")
    print("="*40)
    # Class-wise metrics (we only have class 0: bottle)
    # Mean Average Precision @ IoU 0.50
    map50 = metrics.results_dict.get("metrics/mAP50(B)", 0.0)
    # Mean Average Precision @ IoU 0.50:0.95
    map50_95 = metrics.results_dict.get("metrics/mAP50-95(B)", 0.0)
    # Precision
    precision = metrics.results_dict.get("metrics/precision(B)", 0.0)
    # Recall
    recall = metrics.results_dict.get("metrics/recall(B)", 0.0)

    print(f"Precision (P):       {precision:.4f} (proportion of correct positive detections)")
    print(f"Recall (R):          {recall:.4f} (proportion of actual bottles detected)")
    print(f"mAP50:               {map50:.4f} (accuracy at 50% overlap threshold)")
    print(f"mAP50-95:            {map50_95:.4f} (average accuracy across overlap thresholds)")
    print("="*40)
    print(f"All plots, confusion matrices, and prediction samples are saved in:")
    print(f"  {os.path.abspath(os.path.join('models', 'evaluation'))}")
    print("="*40)

if __name__ == "__main__":
    main()
