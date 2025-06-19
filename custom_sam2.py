import os
import sys

# Add the local sam2 directory to Python path to use the local build_sam.py
sys.path.insert(0, '/home/sai/Desktop/SAM2/segment-anything-2')

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Use float32 instead of bfloat16 for compatibility
torch.set_default_dtype(torch.float32)

if torch.cuda.get_device_properties(0).major >= 8:
    # Turn on tfloat32 for Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Import original SAM instead of SAM2 (since checkpoint is SAM-based)
from segment_anything import sam_model_registry, SamPredictor

# Load the original SAM model with your custom checkpoint
sam_checkpoint = "/home/sai/Desktop/SAM2/segment-anything-2/checkpoints/sam2_multimask_epoch2.pth"
sam_model_type = "vit_h"  # Use ViT-H since your checkpoint has 1280-dimensional features

print("Loading SAM model...")
sam = sam_model_registry[sam_model_type](checkpoint=None)  # Don't load default checkpoint
sam.to(device="cuda")

# Load your custom checkpoint
print("Loading custom checkpoint...")
checkpoint = torch.load(sam_checkpoint, map_location="cpu")
print("Checkpoint keys:", checkpoint.keys())

# Handle the checkpoint structure
if "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
    if "gpu0" in state_dict:
        # Use gpu0 state dict for single GPU inference
        sd = state_dict["gpu0"]
        print("Using gpu0 state dict from distributed checkpoint")
    else:
        sd = state_dict
else:
    sd = checkpoint

# Load the state dict
missing_keys, unexpected_keys = sam.load_state_dict(sd, strict=False)
print(f"Successfully loaded {len(sd)} parameters from checkpoint")
if missing_keys:
    print(f"Missing keys: {len(missing_keys)}")
if unexpected_keys:
    print(f"Unexpected keys: {len(unexpected_keys)}")

sam.eval()

# Create SAM predictor
predictor = SamPredictor(sam)

# Helper function to extract contours from mask
def extract_contours(mask):
    mask_2d = mask.squeeze()
    threshold = 0.02  # Lower threshold for weak masks
    mask_binary = (mask_2d > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
    mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cell_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 5 < area < 5000:  # Lowered min area
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.05:  # Lowered circularity
                    cell_contours.append(cnt)
    if len(cell_contours) == 0 and len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        cell_contours = [largest_contour]
    print(f"Found {len(contours)} total contours, filtered to {len(cell_contours)} cell contours (relaxed)")
    return cell_contours

# Helper function to overlay contours on original image
def overlay_contours_on_image(original_image_path, contours, obj_id, output_path):
    """Overlay contours on the original image and save the result"""
    # Read the original image
    original_image = cv2.imread(original_image_path)
    if original_image is None:
        print(f"Could not read image: {original_image_path}")
        return
    
    # Convert BGR to RGB for consistent color handling
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Create a copy for drawing
    result_image = original_image_rgb.copy()
    
    # Define colors for different objects (you can customize these)
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]
    
    # Get color for this object
    color = colors[(obj_id - 1) % len(colors)]
    
    # Draw contours on the image
    cv2.drawContours(result_image, contours, -1, color, 1)
    
    # Convert back to BGR for saving
    result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    
    # Save the result
    cv2.imwrite(output_path, result_image_bgr)
    print(f"Saved contour overlay: {output_path}")

# Helper function to process contours for a frame
def process_contours_for_frame(frame_idx, masks_dict, original_image_path):
    """Process all masks for a frame and overlay contours on the original image"""
    # Create output directory
    out_dir = f"/home/sai/Desktop/SAM2/ContourOverlays/imgNo4CH0/frame_{frame_idx}"
    os.makedirs(out_dir, exist_ok=True)
    
    # Process each object's mask
    for obj_id, mask in masks_dict.items():
        # Extract contours from the mask
        contours = extract_contours(mask)
        
        if contours:
            # Create output path for this object
            output_path = os.path.join(out_dir, f"obj_{obj_id}_contours.jpg")
            
            # Overlay contours on the original image
            overlay_contours_on_image(original_image_path, contours, obj_id, output_path)
        else:
            print(f"No contours found for object {obj_id} in frame {frame_idx}")

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.1])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.1])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], pos_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = "/home/sai/Desktop/SAM2/Videos/imgNo4CH0/"

# Scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# Take a look at the first video frame
ann_frame_idx = 0
plt.figure(figsize=(8, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))

# Get original image dimensions to calculate padding
original_img_path = os.path.join(video_dir, frame_names[ann_frame_idx])
original_img = cv2.imread(original_img_path)
original_h, original_w = original_img.shape[:2]
print(f"Original image dimensions: {original_w}x{original_h}")

# Calculate padding for 1024x1024
target_size = 1024
pad_h = max(0, target_size - original_h)
pad_w = max(0, target_size - original_w)
pad_top = pad_h // 2
pad_left = pad_w // 2
print(f"Padding: top={pad_top}, left={pad_left}")

# Define initial bounding boxes for objects (in original image coordinates)
# Let's use smaller, more precise boxes around actual cells
initial_boxes = [
    np.array([140, 120, 170, 175], dtype=np.float32),  # [x1, y1, x2, y2] - tighter around cell 1
    np.array([160, 170, 200, 215], dtype=np.float32)   # [x1, y1, x2, y2] - tighter around cell 2
]

# Visualize the initial bounding boxes
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image with Initial Bounding Boxes")
img_orig = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
plt.imshow(img_orig)

for i, box in enumerate(initial_boxes):
    x1, y1, x2, y2 = box
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                        linewidth=2, edgecolor=['red', 'green'][i], facecolor='none')
    plt.gca().add_patch(rect)
    plt.text(x1, y1-5, f'Box {i+1}', color=['red', 'green'][i], fontsize=10)

plt.tight_layout()
plt.show()

# Function to update bounding box based on mask prediction
def update_bbox_from_mask(mask, original_bbox, padding=(10, 10)):
    """Update bounding box position based on predicted mask centroid and extent"""
    # Find mask centroid and extent
    mask_binary = (mask > 0.05).astype(np.uint8)
    
    if np.sum(mask_binary) == 0:
        # No significant prediction, keep original bbox
        return original_bbox
    
    # Find contours to get the main object
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return original_bbox
    
    # Get largest contour (main object)
    largest_contour = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(largest_contour) < 20:  # Reduced minimum area
        return original_bbox
    
    # Get bounding rectangle of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Original bbox dimensions
    orig_x1, orig_y1, orig_x2, orig_y2 = original_bbox
    orig_w = orig_x2 - orig_x1
    orig_h = orig_y2 - orig_y1
    
    # Calculate centroid of new detection
    new_center_x = x + w // 2
    new_center_y = y + h // 2
    
    # More responsive tracking: use more of the detected size
    adaptive_w = int(0.5 * orig_w + 0.5 * w)  # 50% original, 50% detected
    adaptive_h = int(0.5 * orig_h + 0.5 * h)
    
    # Add some padding for better tracking
    adaptive_w = min(adaptive_w + 10, 100)  # Add padding but cap size
    adaptive_h = min(adaptive_h + 10, 100)
    
    # Create new bbox centered on detection but with adaptive size
    new_x1 = max(0, new_center_x - adaptive_w // 2)
    new_y1 = max(0, new_center_y - adaptive_h // 2)
    new_x2 = new_x1 + adaptive_w
    new_y2 = new_y1 + adaptive_h
    
    # Ensure bbox doesn't go outside image boundaries
    new_x2 = min(new_x2, 281)  # Assuming 281x281 image
    new_y2 = min(new_y2, 281)
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    
    return np.array([new_x1, new_y1, new_x2, new_y2], dtype=np.float32)

# Process each frame with temporal tracking
current_boxes = initial_boxes.copy()  # Start with initial boxes

for frame_idx in range(len(frame_names)):
    print(f"\nProcessing frame {frame_idx}...")
    
    # Load the frame
    frame_path = os.path.join(video_dir, frame_names[frame_idx])
    frame_img = cv2.imread(frame_path)
    frame_img_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
    
    # Pad the image to 1024x1024
    padded_frame = cv2.copyMakeBorder(frame_img_rgb, pad_top, pad_h - pad_top, 
                                     pad_left, pad_w - pad_left, 
                                     cv2.BORDER_CONSTANT, value=0)
    
    # Set the image in the predictor
    predictor.set_image(padded_frame)
    
    # Process each bounding box
    frame_masks = {}
    updated_boxes = []
    
    for i, box in enumerate(current_boxes):
        obj_id = i + 1
        
        # Adjust bounding box for padding
        padded_box = np.array([
            box[0] + pad_left,  # x1
            box[1] + pad_top,   # y1
            box[2] + pad_left,  # x2
            box[3] + pad_top    # y2
        ], dtype=np.float32)
        
        print(f"Frame {frame_idx}, Object {obj_id}: Using bbox {box} -> Padded bbox {padded_box}")
        
        # Convert box to SAM format [x1, y1, x2, y2]
        sam_box = padded_box.astype(np.int32)
        
        # Get mask prediction
        masks, scores, logits = predictor.predict(
            box=sam_box,
            multimask_output=False  # Get single best mask
        )
        
        # Get the best mask
        best_mask = masks[0]  # Single mask output
        
        # Crop back to original image size
        original_mask = best_mask[pad_top:pad_top+original_h, pad_left:pad_left+original_w]
        frame_masks[obj_id] = original_mask
        
        # Debug: Print mask information
        print(f"Object {obj_id} mask shape: {original_mask.shape}")
        print(f"Object {obj_id} mask unique values: {np.unique(original_mask)}")
        print(f"Object {obj_id} mask sum: {np.sum(original_mask)}")
        print(f"Object {obj_id} mask min/max: {np.min(original_mask)}/{np.max(original_mask)}")
        
        # Update bounding box for next frame based on mask prediction
        updated_box = update_bbox_from_mask(original_mask, box)
        updated_boxes.append(updated_box)
        
        print(f"Frame {frame_idx}, Object {obj_id}: Updated bbox {updated_box}")
        
        # Debug: Visualize the raw mask (only if mask has content)
        if np.sum(original_mask) > 0:
            plt.figure(figsize=(15, 5))
            
            # Original image with bounding box
            plt.subplot(1, 3, 1)
            plt.title(f"Frame {frame_idx} - Original Image with Box {obj_id}")
            plt.imshow(frame_img_rgb)
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                linewidth=2, edgecolor='red', facecolor='none')
            plt.gca().add_patch(rect)
            
            # Raw mask prediction
            plt.subplot(1, 3, 2)
            plt.title(f"Frame {frame_idx} - Raw Mask Prediction - Object {obj_id}")
            plt.imshow(original_mask, cmap='gray')
            plt.colorbar()
            
            # Mask overlaid on original image
            plt.subplot(1, 3, 3)
            plt.title(f"Frame {frame_idx} - Mask Overlay - Object {obj_id}")
            plt.imshow(frame_img_rgb)
            # Overlay mask in red
            mask_overlay = np.zeros_like(frame_img_rgb)
            mask_overlay[original_mask > 0] = [255, 0, 0]  # Red for mask
            plt.imshow(mask_overlay, alpha=0.5)
            
            plt.tight_layout()
            plt.show()
        else:
            print(f"Object {obj_id} has no mask content (all zeros)")
        
        print("---")
    
    # Update current boxes for next frame
    current_boxes = updated_boxes
    
    # Process contours for this frame
    process_contours_for_frame(frame_idx, frame_masks, frame_path)

print("Temporal tracking and contour overlay processing completed!")

# Helper functions for GUI integration (if needed)
def on_predict(self):
    print("Box:", self.boxes)
    # Call your SAM2 pipeline here, e.g.:
    run_sam2_pipeline(self.boxes)

def get_padding(h, w, target=1024):
    ph, pw = target - h, target - w
    top, bottom = ph // 2, ph - ph // 2
    left, right = pw // 2, pw - pw // 2
    return top, left

def pad_to_1024(img):
    h, w = img.shape[:2]
    ph, pw = 1024 - h, 1024 - w
    top, bottom = ph // 2, ph - ph // 2
    left, right = pw // 2, pw - pw // 2
    padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded, (top, left)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, (top, left) = pad_to_1024(img)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # (3, 1024, 1024)
    return img_tensor

def save_mask_debug(mask, frame_idx, obj_id):
    plt.imshow(mask.squeeze(), cmap='gray')
    plt.title(f'Mask for object {obj_id} frame {frame_idx}')
    plt.show()
