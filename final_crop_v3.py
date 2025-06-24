import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import re
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Set
from tqdm import tqdm
import argparse
import subprocess
from sklearn.cluster import KMeans
import json

# ----------------------------------------
# Script Information
# ----------------------------------------
print("=" * 80)
print("?? NANOWELL DETECTION AND CROPPING TOOL")
print("=" * 80)
print("?? Created by: Saikiran")
print("?? Version: 1.0")
print("?? Date: 2024")
print("?? Purpose: Automated nanowell detection and cropping using YOLO")
print("=" * 80)

# ----------------------------------------
# Logging setup (file + console)
# ----------------------------------------
# Global variable to store the log filename
log_filename = None

def setup_logging(output_dir: Path):
    """Setup logging with a single log file in the results/logs directory"""
    global log_filename
    
    # Create logs directory under the output directory
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a single log file for this run
    log_filename = logs_dir / f'nanowell_cropping_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # Configure logging to write to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    print_and_log(f"Log file created: {log_filename}")
    return log_filename

def print_and_log(msg: str, level: str = 'info'):
    print(msg)
    if level == 'info':
        logging.info(msg)
    elif level == 'warning':
        logging.warning(msg)
    else:
        logging.error(msg)

# ----------------------------------------
# Configuration
# ----------------------------------------
MODEL_PATH       = "/cellchorus/src/multiversion_testing/nanowell_cropping/runs/detect/nanowell_yolo_v7/weights/best.pt"

# # Default values - can be overridden by command line arguments
# DEFAULT_INPUT_ROOT = "/cellchorus/data/d/clients/20250530/20250530_Chip_CPS_1_Macrowell_A_Nalm6Viability-01"
# DEFAULT_BASE_OUT_DIR = "/cellchorus/data/d/clients/20250530/Results"

def generate_pattern_from_folder(folder_path: Path) -> re.Pattern:
    """Generate regex pattern from the folder name"""
    folder_name = folder_path.name
    
    # Extract the base name from the folder path
    # Example: "20250530_Chip_CPS_1_Macrowell_A_Nalm6Viability-01" -> "20250530_Chip_CPS_1_Macrowell_A_Nalm6Viability-01"
    base_name = folder_name
    
    # Create the pattern with the base name
    pattern_str = f"{re.escape(base_name)}_s(\\d{{3}})t(\\d{{1,2}})c1_ORG\\.tif"
    pattern = re.compile(pattern_str, re.IGNORECASE)
    
    print_and_log(f"Generated pattern from folder '{folder_name}': {pattern_str}")
    return pattern

# Expected number of nanowells (can be adjusted based on your data)
EXPECTED_WELLS = 36  # Default expectation, but not a hard limit
MIN_WELLS = 20      # Minimum number of wells to expect (warn if less)

MAX_WELLS = 45      # Maximum number of wells to expect (warn if more)
# Scene to bucket mapping function
def get_bucket_name(scene_id: int) -> str:
    return f"B{scene_id:03d}"

# New folder structure - will be created per bucket
def get_bucket_dirs(bucket: str, base_out_dir: Path) -> tuple:
    images_8bit = base_out_dir / bucket / "images" / "crops_8bit_s"
    images_16bit = base_out_dir / bucket / "images" / "crops_16bit_s"
    crops_vis = base_out_dir / bucket / "images" / "crops_vis"
    debug_dir = base_out_dir / bucket / "debug"
    return images_8bit, images_16bit, crops_vis, debug_dir

CROP_SIZE        = (196, 196)
CONF_THRESH      = 0.1  # Lowered from 0.25 to catch more potential wells
OVERLAP_THRESH   = 0.6  # Increased from 0.4 to allow slightly more overlap
CHANNELS         = [1, 2, 3, 4]  # Original channel numbers
SAVE_DEBUG_IMAGE = True
BATCH_SIZE       = 1
NUM_GPUS         = max(1, torch.cuda.device_count())

# Class definitions
CLASS_LABELS = {"Square": 0, "Diamond": 1, "Empty": 2, "Others": 3, "Invalid": 4}
INVALID_CLASS = CLASS_LABELS["Invalid"]
ID_TO_LABEL = {v: k for k, v in CLASS_LABELS.items()}
MIN_SIZE         = 140  # Lowered from 160 to catch smaller wells
MAX_SIZE         = 300  # Added maximum size check

# Default channel mapping (will be overridden by command line argument)
DEFAULT_CHANNEL_MAP = {
    1: "CH0",
    2: "CH3", 
    3: "CH2",
    4: "CH1"
}

# Global variable to store the dynamic channel mapping
CHANNEL_MAP = DEFAULT_CHANNEL_MAP.copy()

def parse_channel_mapping(channel_mapping_json: str) -> dict:
    """Parse channel mapping from JSON string and convert to internal format"""
    try:
        # Parse the JSON string
        channel_index_dict = json.loads(channel_mapping_json)
        
        # Convert from {"c1_ORG":"CH0", "c2_ORG":"CH3", ...} format to {1:"CH0", 2:"CH3", ...} format
        converted_map = {}
        for key, value in channel_index_dict.items():
            # Extract channel number from key like "c1_ORG" -> 1
            if key.startswith('c') and '_ORG' in key:
                channel_num = int(key[1:key.index('_ORG')])
                converted_map[channel_num] = value
        
        print_and_log(f"Parsed channel mapping: {channel_index_dict}")
        print_and_log(f"Converted to internal format: {converted_map}")
        
        return converted_map
    except Exception as e:
        print_and_log(f"Error parsing channel mapping: {e}", 'error')
        print_and_log(f"Using default channel mapping: {DEFAULT_CHANNEL_MAP}", 'warning')
        return DEFAULT_CHANNEL_MAP.copy()

@dataclass
class ImageBatch:
    paths:    List[Path]
    images:   List[np.ndarray]
    metadata: List[Dict]

def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate IoU (Intersection over Union) between two boxes [x1, y1, x2, y2]"""
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Check if boxes overlap
    if x2 < x1 or y2 < y1:
        return 0.0
        
    # Calculate intersection area
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    # Return IoU
    return intersection / union if union > 0 else 0.0

def filter_overlapping_boxes(boxes: np.ndarray, confidences: np.ndarray = None, classes: np.ndarray = None, iou_threshold: float = OVERLAP_THRESH) -> List[int]:
    """Filter out boxes that overlap more than iou_threshold"""
    if len(boxes) == 0:
        return []
        
    # Initialize list to keep track of boxes to remove
    to_remove = set()
    
    # Compare each box with every other box
    for i in range(len(boxes)):
        if i in to_remove:
            continue
            
        box1 = boxes[i]
        for j in range(i + 1, len(boxes)):
            if j in to_remove:
                continue
                
            box2 = boxes[j]
            iou = calculate_iou(box1, box2)
            
            # If boxes overlap significantly
            if iou > iou_threshold:
                # Log overlap detection
                print_and_log(f"Found overlapping wells {i+1} and {j+1} (IoU={iou:.3f}):")
                if confidences is not None:
                    print_and_log(f"  Well {i+1}: conf={confidences[i]:.3f}")
                    print_and_log(f"  Well {j+1}: conf={confidences[j]:.3f}")
                
                # If we have confidence scores, remove the lower confidence one
                if confidences is not None:
                    if confidences[i] >= confidences[j]:
                        to_remove.add(j)
                        print_and_log(f"  Removing well {j+1} (lower confidence)")
                    else:
                        to_remove.add(i)
                        print_and_log(f"  Removing well {i+1} (lower confidence)")
                        break  # Break since this box is being removed
                else:
                    # If no confidence scores, remove the second box
                    to_remove.add(j)
                    print_and_log(f"  Removing well {j+1} (second box)")
    
    # Create keep list (boxes to keep)
    keep = [i for i in range(len(boxes)) if i not in to_remove]
    
    # Filter out Invalid class if classes are provided
    #if classes is not None:
    #    keep = [i for i in keep if classes[i] != INVALID_CLASS]
    
    return keep

def load_image_batch(paths: List[Path], batch_size: int, pattern: re.Pattern) -> ImageBatch:
    batch = ImageBatch([], [], [])
    for p in paths[:batch_size]:
        try:
            print_and_log(f"Loading image: {p.name}")
            img = cv2.imread(str(p), -1)
            if img is None:
                raise ValueError("cv2.imread returned None")
            m = pattern.search(p.name)
            scene_id = int(m.group(1)) if m else None
            time_id = int(m.group(2)) if m else None
            batch.paths.append(p)
            batch.images.append(img)
            batch.metadata.append({'scene_id': scene_id, 'time_id': time_id})
            print_and_log(f"Parsed scene_id={scene_id}, time_id={time_id}")
        except Exception as e:
            print_and_log(f"Error loading {p.name}: {e}", 'error')
    return batch

def process_batch(model, batch: ImageBatch, pattern: re.Pattern) -> List[Dict]:
    rgb_imgs = []
    for img in batch.images:
        ch1_8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        rgb_imgs.append(cv2.merge([ch1_8]*3))

    try:
        print_and_log(f"Running YOLO on {len(rgb_imgs)} image(s)")
        results = model(rgb_imgs, verbose=False)
    except Exception as e:
        print_and_log(f"YOLO inference error: {e}", 'error')
        return []

    detections = []
    for res, meta in zip(results, batch.metadata):
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)
        
        # Log initial detections
        print_and_log(f"Initial YOLO detections: {len(boxes)} wells")
        for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            print_and_log(f"Well {i+1}: size={width:.1f}x{height:.1f}, conf={conf:.3f}, class={ID_TO_LABEL.get(cls, 'Unknown')}")
        
        # First filter by confidence
        mask = confs >= CONF_THRESH
        removed_conf = np.where(~mask)[0]
        if len(removed_conf) > 0:
            print_and_log(f"Removed {len(removed_conf)} wells due to low confidence (<{CONF_THRESH}):")
            for i in removed_conf:
                print_and_log(f"  Well {i+1}: conf={confs[i]:.3f}")
        
        boxes = boxes[mask]
        confs = confs[mask]
        classes = classes[mask]
        
        # Filter overlapping boxes and invalid class
        keep_indices = filter_overlapping_boxes(boxes, confs, classes, OVERLAP_THRESH)
        removed_overlap = [i for i in range(len(boxes)) if i not in keep_indices]
        if len(removed_overlap) > 0:
            print_and_log(f"Removed {len(removed_overlap)} wells due to overlap (>{OVERLAP_THRESH}):")
            for i in removed_overlap:
                print_and_log(f"  Well {i+1}: overlaps with another well")
        
        boxes = boxes[keep_indices]
        classes = classes[keep_indices]
        confs = confs[keep_indices]  # Keep confidence scores for valid detections
        
        image_name = batch.paths[0].name
        valid_boxes = len([c for c in classes if c != INVALID_CLASS])
        print_and_log(f"whole grid image {image_name} detected wells: {valid_boxes} (total: {len(boxes)})")
        
        if valid_boxes > 0:
            detections.append({'boxes': boxes, 'classes': classes, 'confidences': confs, 'metadata': meta})
        else:
            print_and_log(f"{image_name} ? no valid boxes after filtering", 'warning')
    return detections

# Add a dictionary to store well positions per scene
scene_well_positions = {}

# Add after scene_well_positions dictionary
temporal_cache = {}  # {scene_id: {time_id: (boxes, classes)}}

def get_well_key(center_x: float, center_y: float, tolerance: float = 20) -> tuple:
    """Generate a position key for a well that's consistent across timepoints"""
    # Round to the nearest tolerance to handle slight position variations
    x_key = round(center_x / tolerance) * tolerance
    y_key = round(center_y / tolerance) * tolerance
    return (x_key, y_key)

def load_nanowells(a_center_fname):
    """Load nanowell data from file"""
    f = open(a_center_fname)
    lines = f.readlines()
    f.close()

    nanowells = []
    for line in lines:
        line = line.rstrip().split('\t')
        line = [float(i) for i in line]
        nanowells.append(line)

    return nanowells

def write_nanowells_new(a_center_fname_new, nanowells):
    """Write nanowell data to file in the required format"""
    f = open(a_center_fname_new, 'w')

    for nanowell in nanowells:
        temp = nanowell[0:7]
        temp = [int(temp[0]), int(temp[1]), int(temp[2]), float(temp[3]), int(temp[4]), int(temp[5]), int(temp[6])]
        temp = [str(i) for i in temp]
        temp = '\t'.join(temp) + '\n'
        f.writelines(temp)

    f.close()
    return 1

def write_nanowell_info(fname, info_array):
    """Write nanowell info to file"""
    f = open(fname, 'w')
    for info in info_array:
        line = str(info[0]) + '\t' + str(info[1]) + '\t' + str(info[2]) + '\t' + format(info[3], '.4f') + '\n'
        f.writelines(line)
    f.close()

def sort_nanowell(a_center_fname, a_center_clean_fname, a_center_clean_sorted_fname, RC):
    """
    This function will load the full nanowell file a_center.txt, create an approximate grid using k-means;
    and sort nanowells in a_center_clean based on real Row and Column coordinates.
    """
    # Step 1: load a_center.txt and a_center_clean.txt
    org_nanowells = load_nanowells(a_center_fname)
    clean_nanowells = load_nanowells(a_center_clean_fname)

    # Step 2: get row_coords, col_coords using k-means
    row_margin = np.array([[nanowell[0], 0] for nanowell in org_nanowells])
    col_margin = np.array([[0, nanowell[1]] for nanowell in org_nanowells])

    kmeans_row = KMeans(n_clusters=RC, random_state=0).fit(row_margin)
    kmeans_col = KMeans(n_clusters=RC, random_state=0).fit(col_margin)

    row_coords_flat = sorted([int(i[0]) for i in kmeans_row.cluster_centers_])
    col_coords_flat = sorted([int(j[1]) for j in kmeans_col.cluster_centers_])

    # Step 3: update clean_nanowells --> clean_nanowells_RC with calculated R and C
    index_nanowells = []
    for nanowell in clean_nanowells:
        R = 1
        d0 = 2048
        for i in range(RC):
            d = abs(nanowell[0] - row_coords_flat[i])
            if d < d0:
                d0 = d
                R = i + 1

        C = 1
        d0 = 2048
        for j in range(RC):
            d = abs(nanowell[1] - col_coords_flat[j])
            if d < d0:
                d0 = d
                C = j + 1

        index = RC * (R - 1) + C
        # Directly appending R, C, and index to the nanowell data list
        index_nanowells.append(nanowell + [R, C, index])

    sorted_nanowells = sorted(index_nanowells, key=lambda l: l[-1])  # Sort by index

    # Step 4: write clean_nanowells_RC to file a_center_clean_sorted_fname
    write_nanowells_new(a_center_clean_sorted_fname, sorted_nanowells)

    return sorted_nanowells

def create_selected_nanowells_file(scene_id: int, sorted_nanowells: List, base_out_dir: Path):
    """
    Create selected_nanowells.txt file that the downstream pipeline expects
    This file contains the well indices from the sorted meta file
    """
    bucket = get_bucket_name(scene_id)
    
    # Create the labels directory structure that the downstream pipeline expects
    labels_dir = base_out_dir / bucket / "labels" / "DET" / "FRCNN-Fast" / "raw"
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Create selected_nanowells.txt with well indices
    selected_nanowells_file = labels_dir / "selected_nanowells.txt"
    
    with open(selected_nanowells_file, 'w') as f:
        for nanowell in sorted_nanowells:
            # nanowell format: [x_center, y_center, class, confidence, R, C, index]
            well_index = int(nanowell[6])  # index is the 7th element
            f.write(f"{well_index}\n")
    
    print_and_log(f"Created {selected_nanowells_file} with {len(sorted_nanowells)} well indices")

def generate_meta_files(scene_id: int, valid_detections: List, base_out_dir: Path, time_id: int = 1, confidences: List[float] = None):
    """
    Generate meta files (a_centers.txt, a_centers_clean.txt, a_centers_clean_sorted.txt) for a scene
    """
    bucket = get_bucket_name(scene_id)
    meta_dir = base_out_dir / bucket / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate a_centers.txt (all detections with confidence scores)
    a_centers_fname = meta_dir / "a_centers.txt"
    a_centers_clean_fname = meta_dir / "a_centers_clean.txt"
    a_centers_clean_sorted_fname = meta_dir / "a_centers_clean_sorted.txt"
    
    # For now, we'll use the valid detections from the current timepoint
    # In a full implementation, you might want to aggregate across all timepoints
    all_detections = []
    clean_detections = []
    
    for i, (_, box, center_x, center_y) in enumerate(valid_detections):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Use actual confidence score if available, otherwise default to 0.8
        confidence = confidences[i] if confidences and i < len(confidences) else 0.8
        
        # Class 1 for valid wells (you might want to adjust based on your class mapping)
        nanowell_class = 1
        
        all_detections.append([center_x, center_y, nanowell_class, confidence])
        
        # Only include wells that meet size criteria
        if width >= MIN_SIZE and height >= MIN_SIZE and width <= MAX_SIZE and height <= MAX_SIZE:
            clean_detections.append([center_x, center_y, nanowell_class, confidence])
    
    # Write a_centers.txt (all detections)
    write_nanowell_info(a_centers_fname, all_detections)
    print_and_log(f"Generated {a_centers_fname} with {len(all_detections)} detections")
    
    # Write a_centers_clean.txt (filtered detections)
    write_nanowell_info(a_centers_clean_fname, clean_detections)
    print_and_log(f"Generated {a_centers_clean_fname} with {len(clean_detections)} clean detections")
    
    # Generate a_centers_clean_sorted.txt using K-means clustering
    if len(clean_detections) > 0:
        RC = 6  # Assuming 6x6 grid
        sorted_nanowells = sort_nanowell(a_centers_fname, a_centers_clean_fname, a_centers_clean_sorted_fname, RC)
        print_and_log(f"Generated {a_centers_clean_sorted_fname} with {len(sorted_nanowells)} sorted nanowells")
        return sorted_nanowells
    else:
        print_and_log(f"No clean detections found for scene {scene_id}, skipping sorted file generation", 'warning')
        return []

def ensure_temporal_consistency(scene_id: int, time_id: int, boxes: np.ndarray, classes: np.ndarray) -> tuple:
    """Ensure temporal consistency of well classifications across timepoints.
    Uses IoU (Intersection over Union) to match wells between timepoints."""
    if len(boxes) == 0:
        return boxes, classes
        
    # Initialize scene in temporal cache if not exists
    if scene_id not in temporal_cache:
        temporal_cache[scene_id] = {}
    
    # Store current detections in cache
    temporal_cache[scene_id][time_id] = (boxes.copy(), classes.copy())
    
    # Get all timepoints in the scene
    timepoints = sorted(temporal_cache[scene_id].keys())
    if len(timepoints) <= 1:
        return boxes, classes

    print_and_log(f"\nTemporal Consistency Check for Scene {scene_id}, Time {time_id}:")
    print_and_log(f"Available timepoints: {timepoints}")
    
    # Special handling for t01 and t02
    t01_data = temporal_cache[scene_id].get(1)  # Get t01 data if available
    t02_data = temporal_cache[scene_id].get(2)  # Get t02 data if available
    
    if t01_data is None:
        print_and_log("Warning: t01 data not available for comparison")
        return boxes, classes
        
    t01_boxes, t01_classes = t01_data
    print_and_log(f"t01 has {len(t01_boxes)} wells ({sum(t01_classes != INVALID_CLASS)} valid)")
    
    if t02_data:
        t02_boxes, t02_classes = t02_data
        print_and_log(f"t02 has {len(t02_boxes)} wells ({sum(t02_classes != INVALID_CLASS)} valid)")
    
    # For each well in current timepoint
    for i, (curr_box, curr_cls) in enumerate(zip(boxes, classes)):
        curr_center_x = round((curr_box[0] + curr_box[2]) / 2)
        curr_center_y = round((curr_box[1] + curr_box[3]) / 2)
        print_and_log(f"\nChecking well {i+1} at ({curr_center_x}, {curr_center_y}):")
        print_and_log(f"  Current class: {ID_TO_LABEL.get(curr_cls, 'Unknown')}")
        
        # First check t01
        t01_match = False
        best_t01_iou = 0
        best_t01_cls = None
        best_t01_idx = None
        
        for idx, (t01_box, t01_cls) in enumerate(zip(t01_boxes, t01_classes)):
            # Calculate IoU between current well and t01 well
            iou = calculate_iou(curr_box, t01_box)
            
            # If IoU is better than previous best
            if iou > best_t01_iou:
                best_t01_iou = iou
                best_t01_cls = t01_cls
                best_t01_idx = idx
        
        # If we found a good match in t01 (IoU > 0.5)
        if best_t01_iou > 0.5:
            t01_match = True
            print_and_log(f"  Found matching well in t01: class={ID_TO_LABEL.get(best_t01_cls, 'Unknown')}, IoU={best_t01_iou:.3f}")
            
            # If current well is invalid but t01 is valid, make it valid
            if curr_cls == INVALID_CLASS and best_t01_cls != INVALID_CLASS:
                print_and_log(f"  Making well valid based on t01 (class={ID_TO_LABEL.get(best_t01_cls, 'Unknown')})")
                classes[i] = best_t01_cls
        else:
            print_and_log(f"  No good match in t01 (best IoU={best_t01_iou:.3f})")
        
        # If still invalid and t02 is available, check t02
        if curr_cls == INVALID_CLASS and t02_data:
            best_t02_iou = 0
            best_t02_cls = None
            best_t02_idx = None
            
            for idx, (t02_box, t02_cls) in enumerate(zip(t02_boxes, t02_classes)):
                # Calculate IoU between current well and t02 well
                iou = calculate_iou(curr_box, t02_box)
                
                # If IoU is better than previous best
                if iou > best_t02_iou:
                    best_t02_iou = iou
                    best_t02_cls = t02_cls
                    best_t02_idx = idx
            
            # If we found a good match in t02 (IoU > 0.5)
            if best_t02_iou > 0.5:
                print_and_log(f"  Found matching well in t02: class={ID_TO_LABEL.get(best_t02_cls, 'Unknown')}, IoU={best_t02_iou:.3f}")
                
                # If t02 well is valid, make current well valid
                if best_t02_cls != INVALID_CLASS:
                    print_and_log(f"  Making well valid based on t02 (class={ID_TO_LABEL.get(best_t02_cls, 'Unknown')})")
                    classes[i] = best_t02_cls
            else:
                print_and_log(f"  No good match in t02 (best IoU={best_t02_iou:.3f})")
    
    # Log final validation results
    valid_count = sum(1 for c in classes if c != INVALID_CLASS)
    print_and_log(f"\nAfter temporal consistency check:")
    print_and_log(f"Total wells: {len(boxes)}")
    print_and_log(f"Valid wells: {valid_count}")
    print_and_log(f"Invalid wells: {len(boxes) - valid_count}")
    
    return boxes, classes

def save_crops_bucketed(result: Dict, images: Dict[int, np.ndarray], input_root: Path, base_out_dir: Path, pattern: re.Pattern):
    scene_id    = result['metadata']['scene_id']
    time_id     = result['metadata']['time_id']
    boxes       = result['boxes']
    classes     = result['classes']
    confidences = result.get('confidences', [])

    # 1) Temporal consistency
    print_and_log(f"\nProcessing {get_bucket_name(scene_id)}, t{time_id:02d}:")
    boxes, classes = ensure_temporal_consistency(scene_id, time_id, boxes, classes)

    # 2) Build grid-based sorted order over all boxes
    centers = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in boxes])
    n_grid  = int(np.sqrt(EXPECTED_WELLS))
    kx = KMeans(n_clusters=n_grid, random_state=0).fit(centers[:,0].reshape(-1,1))
    ky = KMeans(n_clusters=n_grid, random_state=0).fit(centers[:,1].reshape(-1,1))
    x_centers = np.sort(kx.cluster_centers_.flatten())
    y_centers = np.sort(ky.cluster_centers_.flatten())

    grid_positions = []
    for idx, (cx, cy) in enumerate(centers):
        col = np.argmin(np.abs(x_centers - cx))
        row = np.argmin(np.abs(y_centers - cy))
        grid_positions.append((row, col, idx))
    sorted_positions = sorted(grid_positions, key=lambda x: (x[1], x[0]))
    all_sorted = [(idx, boxes[idx]) for (_,_,idx) in sorted_positions]
    valid_sorted = [(idx, box) for idx, box in all_sorted if classes[idx] != INVALID_CLASS]

    # 3) Decide whether to include invalid wells
    if len(valid_sorted) == EXPECTED_WELLS:
        to_crop = valid_sorted
        print_and_log(f"Found exactly {EXPECTED_WELLS} valid wells → cropping only those.")
    else:
        to_crop = all_sorted
        print_and_log(f"Valid wells ({len(valid_sorted)}) ≠ {EXPECTED_WELLS} → including invalid wells, total {len(to_crop)}.")

    # 4) Prepare output directories
    bucket    = get_bucket_name(scene_id)
    dir_8bit, dir_16bit, dir_vis, debug_dir = get_bucket_dirs(bucket, base_out_dir)
    for d in (dir_8bit.parent, dir_16bit.parent, dir_vis, debug_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 5) Crop in exact sequence, per-well subfolders
    for well_num, (idx, (x1,y1,x2,y2)) in enumerate(to_crop, start=1):
        for ch in CHANNELS:
            img = images.get(ch)
            if img is None:
                print_and_log(f"  [WARN] missing channel {ch} for well #{well_num}", 'warning')
                continue

            x1i, y1i, x2i, y2i = map(int, [x1,y1,x2,y2])
            crop = img[y1i:y2i, x1i:x2i]
            if crop.size == 0:
                print_and_log(f"  [WARN] zero-size crop at well #{well_num}", 'warning')
                continue

            fname = f"imgNo{well_num}{CHANNEL_MAP[ch]}_t{time_id}.tif"

            # 16-bit subfolder & save
            well_dir_16 = dir_16bit / f"imgNo{well_num}{CHANNEL_MAP[ch]}"
            well_dir_16.mkdir(parents=True, exist_ok=True)
            out16 = well_dir_16 / fname
            resized16 = cv2.resize(crop, CROP_SIZE, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out16), resized16, [cv2.IMWRITE_TIFF_COMPRESSION,1])
            print_and_log(f"Saved 16-bit crop: {out16}")

            # 8-bit subfolder & save
            well_dir_8 = dir_8bit / f"imgNo{well_num}{CHANNEL_MAP[ch]}"
            well_dir_8.mkdir(parents=True, exist_ok=True)
            norm8 = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            resized8 = cv2.resize(norm8, CROP_SIZE, interpolation=cv2.INTER_AREA)
            out8 = well_dir_8 / fname
            cv2.imwrite(str(out8), resized8, [cv2.IMWRITE_TIFF_COMPRESSION,1])
            print_and_log(f"Saved 8-bit crop: {out8}")

            # vis subfolder & save
            well_vis = dir_vis / f"imgNo{well_num}{CHANNEL_MAP[ch]}"
            well_vis.mkdir(parents=True, exist_ok=True)
            outvis = well_vis / fname
            cv2.imwrite(str(outvis), resized16, [cv2.IMWRITE_TIFF_COMPRESSION,1])
            print_and_log(f"Saved vis crop: {outvis}")

def process_gpu_chunk(gpu_id: int, files: List[Path], input_root: Path, base_out_dir: Path, pattern: re.Pattern) -> List[str]:
    print_and_log(f"[GPU{gpu_id}] Processing {len(files)} file(s)")
    # Sort by scene ID first, then time ID
    files = sorted(files, key=lambda f: tuple(map(int, pattern.search(f.name).groups())))

    try:
        model = YOLO(MODEL_PATH, task='detect')
        model.to(f"cuda:{gpu_id}")
    except Exception as e:
        print_and_log(f"Model load error on GPU{gpu_id}: {e}", 'error')
        return []

    processed = []
    for i in range(0, len(files), BATCH_SIZE):
        batch = load_image_batch(files[i:i+BATCH_SIZE], BATCH_SIZE, pattern)
        if not batch.paths: continue
        detections = process_batch(model, batch, pattern)
        for det in detections:
            sid = det['metadata']['scene_id']
            tid = det['metadata']['time_id']
            
            # Get the base name from the pattern for constructing other channel paths
            base_name = input_root.name
            
            # Update the image path format for other channels
            images = {}
            for ch in CHANNELS:
                ch_path = input_root / f"{base_name}_s{sid:03d}t{tid:02d}c{ch}_ORG.tif"
                if ch_path.exists():
                    images[ch] = cv2.imread(str(ch_path), -1)
                else:
                    print_and_log(f"Warning: Channel {ch} file not found: {ch_path}", 'warning')
                    images[ch] = None
            
            save_crops_bucketed(det, images, input_root, base_out_dir, pattern)
            processed.append(f"s{sid:03d}_t{tid:02d}: {len(det['boxes'])} wells")
        torch.cuda.empty_cache()

    print_and_log(f"[GPU{gpu_id}] Done, processed {len(processed)} time-points")
    return processed

def run_parallel_on_gpus(input_root: Path, base_out_dir: Path, pattern: re.Pattern):
    files = sorted(input_root.glob(f"{input_root.name}_s*c1_ORG.tif"))
    total = len(files)
    if total == 0:
        print_and_log("No matching files found!", 'error')
        return

    print_and_log(f"Found {total} file(s)")
    chunk_size = max(1, total // NUM_GPUS)
    chunks = [files[i:i+chunk_size] for i in range(0, total, chunk_size)]
    # ensure we don't create more chunks than GPUs
    if len(chunks) > NUM_GPUS:
        last = chunks.pop()
        chunks[-1].extend(last)
    print_and_log(f"Split into {len(chunks)} chunk(s)")

    pbar = tqdm(total=total, desc="Processing images")
    processed_count = 0

    with ThreadPoolExecutor(max_workers=NUM_GPUS) as executor:
        futures = [executor.submit(process_gpu_chunk, gid, chunk, input_root, base_out_dir, pattern)
                   for gid, chunk in enumerate(chunks)]
        for fut in as_completed(futures):
            try:
                results = fut.result()
                for msg in results:
                    print_and_log(msg)
                    processed_count += 1
                    pbar.update(1)
            except Exception as e:
                print_and_log(f"Error in chunk: {e}", 'error')

    pbar.close()
    print_and_log(f"Completed: {processed_count}/{total} time-points")

def NANOWELL_CROP_IMAGES(RAW_INPUT_PATH, OUTPUT_PATH):
    """
    Main function to process nanowell detection and cropping.
    
    Args:
        RAW_INPUT_PATH (str or Path): Path to the input directory containing the image files
        OUTPUT_PATH (str or Path): Path to the output directory where results will be saved
    
    Returns:
        bool: True if processing completed successfully, False otherwise
    """
    print("\n" + "=" * 60)
    print("?? STARTING NANOWELL CROPPING PROCESS")
    print("=" * 60)
    print(f"?? Input Directory: {RAW_INPUT_PATH}")
    print(f"?? Output Directory: {OUTPUT_PATH}")
    print(f"?? Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # Convert to Path objects
        input_root = Path(RAW_INPUT_PATH)
        base_out_dir = Path(OUTPUT_PATH)
        
        # Setup logging in the results directory
        setup_logging(base_out_dir)
        
        # Validate input directory exists
        if not input_root.exists():
            print_and_log(f"?? Error: Input directory does not exist: {input_root}", 'error')
            return False
        
        print(f"?? Input directory validated successfully")
        
        # Generate pattern from input folder name
        pattern = generate_pattern_from_folder(input_root)
        
        print(f"?? Generated file pattern: {pattern.pattern}")
        print(f"?? YOLO Model Path: {MODEL_PATH}")
        print(f"????  Available GPUs: {NUM_GPUS}")
        
        print_and_log(f"Input root: {input_root}")
        print_and_log(f"Base output directory: {base_out_dir}")
        print_and_log(f"Generated pattern: {pattern.pattern}")
        
        print_and_log(f"Starting with {NUM_GPUS} GPU(s)")
        # Base directory will be created as needed for each bucket
        base_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"?? Created output directory structure")
        
        run_parallel_on_gpus(input_root, base_out_dir, pattern)
        
        print("\n" + "=" * 60)
        print("?? NANOWELL CROPPING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"?? End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"?? Results saved to: {base_out_dir}")
        print(f"?? Log file: {log_filename}")
        print("=" * 60)
        print("All done.")
        return True
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("?? NANOWELL CROPPING FAILED!")
        print("=" * 60)
        print(f"?? Error Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"?? Error: {e}")
        print("=" * 60)
        print_and_log(f"Error in NANOWELL_CROP_IMAGES: {e}", 'error')
        return False

def NANOWELL_CROP_IMAGES_WITH_RANGE(RAW_INPUT_PATH, OUTPUT_PATH, BLOCKS, FRAMES):
    """
    Main function to process nanowell detection and cropping with specific block and frame ranges.
    
    Args:
        RAW_INPUT_PATH (str or Path): Path to the input directory containing the image files
        OUTPUT_PATH (str or Path): Path to the output directory where results will be saved
        BLOCKS (int, str, or list): Can be:
                                   - Integer: Number of blocks (scenes) to process (e.g., 27 means s001 to s027)
                                   - String: Single block like 'B001' to process only s001
                                   - List: Multiple blocks like ['B001', 'B002'] to process only s001, s002
        FRAMES (int): Number of frames (time points) to process (e.g., 45 means t01 to t45)
    
    Returns:
        bool: True if processing completed successfully, False otherwise
    """
    # Handle BLOCKS parameter conversion
    target_scenes = []
    
    if isinstance(BLOCKS, list):
        # Handle list of blocks like ['B001', 'B002']
        for block in BLOCKS:
            if isinstance(block, str) and block.startswith('B'):
                try:
                    scene_id = int(block[1:])  # Remove 'B' and convert to int
                    target_scenes.append(scene_id)
                except ValueError:
                    print_and_log(f"Error: Could not convert {block} to integer", 'error')
                    return False
            else:
                print_and_log(f"Error: Block {block} is not in expected format 'B001'", 'error')
                return False
    elif isinstance(BLOCKS, str):
        if BLOCKS.startswith('B'):
            # Single block like 'B001'
            try:
                scene_id = int(BLOCKS[1:])  # Remove 'B' and convert to int
                target_scenes = [scene_id]
                print_and_log(f"Processing single block: {BLOCKS} (scene s{scene_id:03d})")
            except ValueError:
                print_and_log(f"Error: Could not convert {BLOCKS} to integer", 'error')
                return False
        else:
            # Assume it's a number string like '27'
            try:
                max_scene = int(BLOCKS)
                target_scenes = list(range(1, max_scene + 1))  # s001 to s027
                print_and_log(f"Processing range: s001 to s{max_scene:03d}")
            except ValueError:
                print_and_log(f"Error: Could not convert {BLOCKS} to integer", 'error')
                return False
    elif isinstance(BLOCKS, int):
        # Integer like 27 means s001 to s027
        target_scenes = list(range(1, BLOCKS + 1))
        print_and_log(f"Processing range: s001 to s{BLOCKS:03d}")
    else:
        print_and_log(f"Error: BLOCKS must be int, str, or list, got {type(BLOCKS)}", 'error')
        return False
    
    # Validate target_scenes
    if not target_scenes:
        print_and_log("Error: No valid scenes to process", 'error')
        return False
    
    # Sort and remove duplicates
    target_scenes = sorted(list(set(target_scenes)))
    
    print("\n" + "=" * 60)
    print("?? STARTING NANOWELL CROPPING PROCESS WITH RANGE")
    print("=" * 60)
    print(f"?? Input Directory: {RAW_INPUT_PATH}")
    print(f"?? Output Directory: {OUTPUT_PATH}")
    print(f"?? Target Scenes: {[f's{scene:03d}' for scene in target_scenes]}")
    print(f"?? Frames (Time): 1 to {FRAMES} (t01 to t{FRAMES:02d})")
    print(f"?? Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # Convert to Path objects
        input_root = Path(RAW_INPUT_PATH)
        base_out_dir = Path(OUTPUT_PATH)
        
        # Setup logging in the results directory
        setup_logging(base_out_dir)
        
        # Validate input directory exists
        if not input_root.exists():
            print_and_log(f"?? Error: Input directory does not exist: {input_root}", 'error')
            return False
        
        print(f"?? Input directory validated successfully")
        
        # Generate pattern from input folder name
        pattern = generate_pattern_from_folder(input_root)
        
        print(f"?? Generated file pattern: {pattern.pattern}")
        print(f"?? YOLO Model Path: {MODEL_PATH}")
        print(f"????  Available GPUs: {NUM_GPUS}")
        
        print_and_log(f"Input root: {input_root}")
        print_and_log(f"Base output directory: {base_out_dir}")
        print_and_log(f"Generated pattern: {pattern.pattern}")
        
        print_and_log(f"Starting with {NUM_GPUS} GPU(s)")
        # Base directory will be created as needed for each bucket
        base_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"?? Created output directory structure")
        
        # Call the modified run function with specific target scenes
        run_parallel_on_gpus_with_specific_scenes(input_root, base_out_dir, pattern, target_scenes, FRAMES)
        
        print("\n" + "=" * 60)
        print("?? NANOWELL CROPPING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"?? End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"?? Results saved to: {base_out_dir}")
        print(f"?? Log file: {log_filename}")
        print("=" * 60)
        print("All done.")
        return True
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("?? NANOWELL CROPPING FAILED!")
        print("=" * 60)
        print(f"?? Error Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"?? Error: {e}")
        print("=" * 60)
        print_and_log(f"Error in NANOWELL_CROP_IMAGES_WITH_RANGE: {e}", 'error')
        return False

def run_parallel_on_gpus_with_specific_scenes(input_root: Path, base_out_dir: Path, pattern: re.Pattern, scenes: List[int], frames: int):
    """
    Run parallel processing with specific scenes and frames
    """
    # Find files that match the pattern and are within the specified ranges
    all_files = list(input_root.glob(f"{input_root.name}_s*c1_ORG.tif"))
    
    # Filter files based on scene and frame ranges
    filtered_files = []
    for file_path in all_files:
        match = pattern.search(file_path.name)
        if match:
            scene_id = int(match.group(1))
            time_id = int(match.group(2))
            
            # Check if scene and time are within specified ranges
            if scene_id in scenes and 1 <= time_id <= frames:
                filtered_files.append(file_path)
    
    total = len(filtered_files)
    if total == 0:
        print_and_log(f"No matching files found for scenes {scenes}, frames 1-{frames}!", 'error')
        return

    print_and_log(f"Found {total} file(s) for scenes {scenes}, frames 1-{frames}")
    chunk_size = max(1, total // NUM_GPUS)
    chunks = [filtered_files[i:i+chunk_size] for i in range(0, total, chunk_size)]
    # ensure we don't create more chunks than GPUs
    if len(chunks) > NUM_GPUS:
        last = chunks.pop()
        chunks[-1].extend(last)
    print_and_log(f"Split into {len(chunks)} chunk(s)")

    pbar = tqdm(total=total, desc="Processing images")
    processed_count = 0

    with ThreadPoolExecutor(max_workers=NUM_GPUS) as executor:
        futures = [executor.submit(process_gpu_chunk, gid, chunk, input_root, base_out_dir, pattern)
                   for gid, chunk in enumerate(chunks)]
        for fut in as_completed(futures):
            try:
                results = fut.result()
                for msg in results:
                    print_and_log(msg)
                    processed_count += 1
                    pbar.update(1)
            except Exception as e:
                print_and_log(f"Error in chunk: {e}", 'error')

    pbar.close()
    print_and_log(f"Completed: {processed_count}/{total} time-points")


def main():
    print("\n" + "=" * 60)
    print("?? RUNNING NANOWELL CROPPING TOOL (COMMAND LINE MODE)")
    print("=" * 60)
    print("?? Created by: Saikiran")
    print("?? Automated nanowell detection and cropping")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description='Nanowell detection and cropping tool')
    parser.add_argument('--input-root', type=str, 
                       default="/cellchorus/data/d/clients/20250530/20250530_Chip_CPS_1_Macrowell_A_Nalm6Viability-01",
                       help='Input root directory')
    parser.add_argument('--base-out-dir', type=str, 
                       default="/cellchorus/data/d/clients/20250530/Results",
                       help='Base output directory')
    parser.add_argument('--blocks', type=str, 
                       help='Blocks to process. Can be:\n'
                            '- Single block: "B001" (process only s001)\n'
                            '- Multiple blocks: "B001,B002" (process s001, s002)\n'
                            '- Range: "27" (process s001 to s027)')
    parser.add_argument('--frames', type=int, 
                       help='Number of frames (time points) to process (e.g., 45 means t01 to t45)')
    parser.add_argument('--channel-mapping', type=str,
                       help='Channel mapping in JSON format. Example: {"c1_ORG":"CH0", "c2_ORG":"CH3", "c3_ORG":"CH2", "c4_ORG":"CH1"}')
    
    args = parser.parse_args()
    
    # Parse and set channel mapping if provided
    global CHANNEL_MAP
    if args.channel_mapping:
        CHANNEL_MAP = parse_channel_mapping(args.channel_mapping)
        print(f"?? Channel Mapping: {CHANNEL_MAP}")
    else:
        print(f"?? Using Default Channel Mapping: {CHANNEL_MAP}")
    
    print(f"?? Input Directory: {args.input_root}")
    print(f"?? Output Directory: {args.base_out_dir}")
    if args.blocks and args.frames:
        print(f"?? Blocks: {args.blocks}")
        print(f"?? Frames (Time): 1 to {args.frames} (t01 to t{args.frames:02d})")
    print("=" * 60)
    
    # Use the appropriate function based on whether blocks and frames are provided
    if args.blocks and args.frames:
        # Parse blocks argument
        if ',' in args.blocks:
            # Multiple blocks like "B001,B002"
            blocks_list = [block.strip() for block in args.blocks.split(',')]
            success = NANOWELL_CROP_IMAGES_WITH_RANGE(args.input_root, args.base_out_dir, blocks_list, args.frames)
        else:
            # Single block or range
            success = NANOWELL_CROP_IMAGES_WITH_RANGE(args.input_root, args.base_out_dir, args.blocks, args.frames)
    else:
        # Use the original function (process all files)
        success = NANOWELL_CROP_IMAGES(args.input_root, args.base_out_dir)
    
    if success:
        print_and_log("?? Processing completed successfully!")
    else:
        print_and_log("?? Processing failed!", 'error')
        sys.exit(1)

if __name__ == "__main__":
    main()


