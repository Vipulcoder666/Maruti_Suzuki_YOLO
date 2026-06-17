import os
import shutil
import random
import glob

def main():
    # Define directories
    temp_dir = os.path.join("data", "temp_captured")
    
    # Destination directories
    train_img_dir = os.path.join("data", "train", "images")
    train_lbl_dir = os.path.join("data", "train", "labels")
    val_img_dir = os.path.join("data", "valid", "images")
    val_lbl_dir = os.path.join("data", "valid", "labels")

    # Recreate destination directories to start fresh
    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    if not os.path.exists(temp_dir):
        print(f"Error: Temporary directory '{temp_dir}' not found.")
        return

    # Find all images
    img_extensions = ["*.jpg", "*.jpeg", "*.png"]
    img_paths = []
    for ext in img_extensions:
        img_paths.extend(glob.glob(os.path.join(temp_dir, ext)))
    img_paths = sorted(img_paths)

    if not img_paths:
        print(f"Error: No images found in {temp_dir}!")
        return

    print(f"Found {len(img_paths)} total images in {temp_dir}.")

    # Gather image-label pairs, generating empty label files if missing
    pairs = []
    for img_path in img_paths:
        base, _ = os.path.splitext(img_path)
        lbl_path = base + ".txt"
        
        # If no label file exists, treat it as a negative sample (empty file)
        if not os.path.exists(lbl_path):
            with open(lbl_path, 'w') as f:
                pass  # create empty file
            print(f"Notice: Created empty label for negative sample: {os.path.basename(img_path)}")
            
        pairs.append((img_path, lbl_path))

    # Shuffle the dataset randomly
    random.seed(42)
    random.shuffle(pairs)

    # Split: 80% train, 20% validation
    split_index = int(len(pairs) * 0.8)
    train_pairs = pairs[:split_index]
    val_pairs = pairs[split_index:]

    print(f"Split distribution: {len(train_pairs)} training, {len(val_pairs)} validation.")

    # Copy files helper
    def copy_pairs(pairs_list, img_dest, lbl_dest):
        for img_p, lbl_p in pairs_list:
            # Copy image
            shutil.copy(img_p, os.path.join(img_dest, os.path.basename(img_p)))
            # Copy label
            shutil.copy(lbl_p, os.path.join(lbl_dest, os.path.basename(lbl_p)))

    print("Copying files to training and validation directories...")
    copy_pairs(train_pairs, train_img_dir, train_lbl_dir)
    copy_pairs(val_pairs, val_img_dir, val_lbl_dir)
    print("Files copied successfully.")

    # Create data.yaml
    workspace_dir = os.path.abspath(".")
    
    # We use forward slashes for data.yaml paths as recommended for cross-platform compatibility, even on Windows
    train_path = os.path.join(workspace_dir, train_img_dir).replace("\\", "/")
    val_path = os.path.join(workspace_dir, val_img_dir).replace("\\", "/")

    yaml_content = f"""train: {train_path}
val: {val_path}

nc: 1
names:
  0: bottle
"""
    
    yaml_path = os.path.join(workspace_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
        
    print(f"Created YOLO dataset configuration file at: {yaml_path}")
    print("\n=== Dataset Ready for YOLO Training ===")

if __name__ == "__main__":
    main()
