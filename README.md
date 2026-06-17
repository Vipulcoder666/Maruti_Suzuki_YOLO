# Real-Time Bottle Detection System

A complete, end-to-end computer vision pipeline to detect plastic and metal bottles in real-time using **YOLOv8 Nano** and **OpenCV**. Optimized to run efficiently on standard laptop CPUs (specifically tested on AMD Ryzen 5 5500) and structured to demonstrate standard MLOps best practices to a Tech Lead.

---

## Key Features
- **Multi-Orientation Detection:** Detects bottles from all angles (upright, tilted, sideways, upside down, or partially blocked).
- **Multi-Material Support:** Robust detection for both transparent plastic and metallic/painted metal bottles.
- **Zero False-Positives Design:** Utilizes active negative background sampling (training on empty hands, keyboards, monitors, chairs) to prevent false alerts.
- **CPU-Optimized Training & Inference:** Uses YOLOv8 Nano (`3.2M` parameters), achieving 30+ FPS on laptop CPUs and fast training cycles (~15 minutes for 30 epochs).
- **Interactive Webcam Demo:** Interactive confidence threshold adjustments (`[` and `]`) directly in the live camera feed window.

---

## Project Structure
```
Bottle_Detection/
│
├── data/                       # Local dataset (Ignored by Git)
│   ├── temp_captured/          # Raw images from capture script
│   ├── train/                  # 80% train split (images + labels)
│   └── valid/                  # 20% validation split (images + labels)
│
├── models/                     # Trained models & metrics
│   ├── best.pt                 # Final trained model weights (Ignored by Git)
│   └── evaluation/             # Confusion matrices, PR curves, validation plots (Ignored by Git)
│
├── scripts/                    # Automation and pipeline scripts
│   ├── capture_images.py       # Capture frames from webcam
│   ├── annotate_images.py      # Draw bounding boxes and auto-save YOLO labels
│   ├── prepare_dataset.py      # Shuffle, split dataset, and generate data.yaml
│   ├── train.py                # Local CPU training pipeline
│   ├── test.py                 # Compute precision, recall, and validation metrics
│   └── detect.py               # Real-time webcam inference demo
│
├── .gitignore                  # Prevents committing large weights/datasets
├── data.yaml                   # YOLO dataset absolute paths (Ignored by Git)
└── requirements.txt            # Python dependencies
```

---

## Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone <your-repository-url>
   cd Bottle_Detection
   ```

2. **Install Dependencies:**
   Ensure Python 3.8+ is installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

---

## Workflow Steps

### Step 1: Capture Dataset Images
Capture images of your target bottles in your environment:
```bash
python scripts/capture_images.py
```
- **Instructions:** Hold your bottles in various orientations (upright, sideways, upside down, close, far) and press **[SPACE]** to capture frames. Capture ~120 positive frames.
- **Negative Samples:** Remove the bottles from view and capture ~30 frames of empty backgrounds, hands, monitors, and keyboard sweeps. Press **[Q]** to quit.

### Step 2: Local Annotation
Annotate the captured images locally without needing third-party cloud accounts or web tokens:
```bash
python scripts/annotate_images.py
```
- **Instructions:** Click and drag to draw a box around the bottle. Press **[S]** to save the bounding boxes.
- **Controls:** Press **[D]** for next image, **[A]** for previous image, **[C]** to clear current boxes, and **[Q]** to quit.
- **Negative Samples:** Do not draw any boxes on background-only images; simply press **[D]** to save empty labels.

### Step 3: Format & Split Dataset
Run the splitting script to divide the annotated images into training (80%) and validation (20%) sets and auto-generate `data.yaml`:
```bash
python scripts/prepare_dataset.py
```

### Step 4: Model Training
Fine-tune the pre-trained YOLOv8 Nano weights on your local CPU:
```bash
python scripts/train.py
```
- Optimizations: Set to train for 30 epochs with a batch size of 8 using 4 CPU worker threads. Results and weights are saved in `models/`.

### Step 5: Compute Evaluation Metrics
Run evaluation to verify final precision, recall, and mAP metrics:
```bash
python scripts/test.py
```

### Step 6: Deploy Real-Time webcam Demo
Run the interactive live detector:
```bash
python scripts/detect.py
```
- **Live Controls:**
  - Press **`]`** to increase the confidence threshold (removes false positives).
  - Press **`[`** to decrease the confidence threshold (increases sensitivity).
  - Press **`Q`** to quit.
