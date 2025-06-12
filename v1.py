import os
import cv2
import numpy as np
from glob import glob
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QColor
from concurrent.futures import ThreadPoolExecutor, as_completed


class ImageLoader(QThread):
    finished = pyqtSignal(list, list, list, str, int)

    def __init__(self, run_dirname, uid, max_time_points):
        super().__init__()
        self.run_dirname = run_dirname
        self.uid = uid
        self.stopped = False
        self.channels = ['CH0', 'CH1', 'CH2', 'CH3']
        self.max_time_points = max_time_points

    def stop_thread(self):
        self.stopped = True

    def run(self):
        images, boxes, channel_data = self.load_images(self.uid)
        # Emit finished signal with all data at once
        self.finished.emit(images, boxes, channel_data, self.uid, len(images))

    def load_16bit_image(self, img_path):
        """Load and normalize 16-bit image to 8-bit"""
        try:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
                return None, None

        # Normalize to full range
        img_normalized = cv2.normalize(
            img,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U
        )
            return img_normalized, img
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None

    def load_images(self, uid):
        """Load all images for a UID at once using parallel processing"""
        # Handle tab-separated or float uid
        if '\t' in uid:
            uid = uid.split('\t')[0]
        
        try:
            tot_uid = int(float(uid))
        except ValueError as e:
            print(f"Error converting uid '{uid}' to float: {e}")
            return [], [], []

        # Decode uid components
        col_val = tot_uid % 100
        tot_uid = tot_uid // 100
        row_val = tot_uid % 100
        tot_uid = tot_uid // 100
        block_val = str(tot_uid).zfill(3)  # Simplified zero padding
        img_no = str(6 * (col_val - 1) + row_val)

        # Base paths for images
        base_image_folder = os.path.join(self.run_dirname, f"B{block_val}/images")
        labels_folder = os.path.join(
            self.run_dirname, f"B{block_val}/labels/DET/FRCNN-Fast/clean/imgNo{img_no}")

        def process_time_point(time_point):
            """Process a single time point - load all channels and boxes"""
            if self.stopped:
                return None
                
            current_channel_images = {}

            # Load all channels for this time point
            for channel in self.channels:
                # Try 16-bit first, then 8-bit
                channel_paths = [
                    os.path.join(base_image_folder, f"crops_16bit_s/imgNo{img_no}{channel}/imgNo{img_no}{channel}_t{time_point}.tif"),
                    os.path.join(base_image_folder, f"crops_8bit_s/imgNo{img_no}{channel}/imgNo{img_no}{channel}_t{time_point}.tif")
                ]

                channel_img, channel_img_data = None, None
                for path in channel_paths:
                    if os.path.exists(path):
                        channel_img, channel_img_data = self.load_16bit_image(path)
                        if channel_img is not None:
                            break

                    if channel_img is not None:
                    current_channel_images[channel] = (channel_img, channel_img_data)

            # Must have CH0 to continue
            if 'CH0' not in current_channel_images:
                return None

            # Load bounding boxes
            cur_boxes = {}
            for box_type, letter in [('eff', 'E'), ('tar', 'T')]:
                box_file = os.path.join(labels_folder, f"imgNo{img_no}{letter}_t{time_point}.txt")
                if os.path.exists(box_file):
                    try:
                        with open(box_file, 'r') as f:
                    lines = f.readlines()
                            cur_boxes[box_type] = [
                                list(map(float, line.strip().split("\t")[1:5])) 
                                for line in lines if line.strip()
                            ]
                    except Exception as e:
                        print(f"Error reading box file {box_file}: {e}")
                        cur_boxes[box_type] = []
                else:
                    cur_boxes[box_type] = []

            return time_point, current_channel_images, cur_boxes

        # Use ThreadPoolExecutor for fast parallel loading
        images = []
        boxes = []
        channel_data = []
        
        # Use more workers for faster loading
        max_workers = min(16, os.cpu_count() * 2)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks at once (up to reasonable limit)
            max_frames = min(self.max_time_points, 2000)  # Safety limit
            futures = {
                executor.submit(process_time_point, tp): tp 
                for tp in range(1, max_frames + 1)
            }
            
            # Collect results as they complete
            results = {}
            for future in as_completed(futures):
                if self.stopped:
                    break

                try:
                    result = future.result(timeout=10)
                    if result is not None:
                        time_point, current_channel_images, cur_boxes = result
                        results[time_point] = (current_channel_images, cur_boxes)
                except Exception as e:
                    print(f"Error processing time point: {e}")
                    continue
            
            # Sort results by time point and build final lists
            sorted_time_points = sorted(results.keys())
            
            # Find the last consecutive time point (stop at first gap)
            last_time_point = 0
            for tp in sorted_time_points:
                if tp == last_time_point + 1:
                    last_time_point = tp
                else:
                    break

            # Build final arrays with only consecutive frames
            for tp in range(1, last_time_point + 1):
                if tp in results:
                    current_channel_images, cur_boxes = results[tp]
                    images.append(current_channel_images['CH0'])
                    boxes.append(cur_boxes)
                    channel_data.append(current_channel_images)

        print(f"Loaded {len(images)} frames for UID {uid}")
        return images, boxes, channel_data
