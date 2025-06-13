# import os
# import cv2
# import numpy as np
# from glob import glob
# from PyQt5.QtCore import QThread, pyqtSignal
# from PyQt5.QtGui import QColor
# from concurrent.futures import ThreadPoolExecutor, as_completed


# class ImageLoader(QThread):
#     finished = pyqtSignal(list, list, list, str)
#     progress = pyqtSignal(int, list, list, list, str)

#     def __init__(self, run_dirname, uid):
#         super().__init__()
#         self.run_dirname = run_dirname
#         self.uid = uid
#         self.stopped = False
#         self.channels = ['CH0', 'CH1', 'CH2', 'CH3']
#         self.cores_used = 4

#     def stop_thread(self):
#         self.stopped = True

#     def run(self):
#         images, boxes, channel_data = self.load_images(self.uid)
#         self.finished.emit(images, boxes, channel_data, self.uid)

#     def load_8bit_image(self, img_path):
#         # Read 16-bit image
#         img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
#         if img is None:
#             print(f"Could not read image: {img_path}")
#             return None
#         return img
#         # if img is None:
#         #     print(f"Could not read image: {img_path}")
#         #     return None

#         # # Normalize to full range
#         # img_normalized = cv2.normalize(
#         #     img,
#         #     None,
#         #     alpha=0,
#         #     beta=255,
#         #     norm_type=cv2.NORM_MINMAX,
#         #     dtype=cv2.CV_8U
#         # )

#         # return img_normalized  # , img

#     def load_16bit_image(self, img_path):
#         # Read 16-bit image
#         img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

#         if img is None:
#             print(f"Could not read image: {img_path}")
#             return None

#         # Normalize to full range
#         img_normalized = cv2.normalize(
#             img,
#             None,
#             alpha=0,
#             beta=255,
#             norm_type=cv2.NORM_MINMAX,
#             dtype=cv2.CV_8U
#         )

#         return img_normalized, img  # , img

#     def load_channel_image(self, channel_paths):
#         # Try to load 16-bit first, then fall back to 8-bit
#         img = None
#         img_data = None
#         if os.path.exists(channel_paths[16]):
#             img, img_data = self.load_16bit_image(channel_paths[16])
#         if img is None:
#             # TODO current bug where there are 16-bit images in 8-bit folder
#             if os.path.exists(channel_paths[8]):
#                 img, img_data = self.load_16bit_image(channel_paths[8])

#         return img, img_data

#     # def load_images(self, uid):
#     #     # Handle tab-separated or float uid
#     #     if '\t' in uid:
#     #         uid = uid.split('\t')[0]

#     #     try:
#     #         tot_uid = int(float(uid))
#     #     except ValueError as e:
#     #         print(f"Error converting uid '{uid}' to float: {e}")
#     #         return [], [], []

#     #     # Decode uid components
#     #     col_val = tot_uid % 100
#     #     tot_uid = int(tot_uid / 100)
#     #     row_val = tot_uid % 100
#     #     tot_uid = int(tot_uid / 100)
#     #     block_val = str(tot_uid)
#     #     block_val = "0" * (3 - len(block_val)) + block_val
#     #     img_no = str(10 * (col_val - 1) + row_val)

#     #     # Base paths for images
#     #     base_image_folder = os.path.join(
#     #         self.run_dirname, f"B{block_val}/images")
#     #     labels_folder = os.path.join(
#     #         self.run_dirname, f"B{block_val}/labels/DET/FRCNN-Fast/clean/imgNo{img_no}")

#     #     images = []
#     #     boxes = []
#     #     channel_data = []

#     #     time_point = 0
#     #     while not self.stopped:
#     #         next_time_point = time_point + 1

#     #         # Channel data for this time point (as a list to maintain sequence)
#     #         current_channel_images = {}

#     #         # Attempt to load each channel's image
#     #         for channel in self.channels:
#     #             # Prefer 16-bit, fall back to 8-bit
#     #             channel_paths = {
#     #                 16: os.path.join(
#     #                     base_image_folder, f"crops_16bit_s/imgNo{img_no}{channel}/imgNo{img_no}{channel}_t{next_time_point}.tif"),
#     #                 8: os.path.join(
#     #                     base_image_folder, f"crops_8bit_s/imgNo{img_no}{channel}/imgNo{img_no}{channel}_t{next_time_point}.tif")
#     #             }

#     #             channel_img = None
#     #             channel_img_data = None

#     #             if os.path.exists(channel_paths[16]):
#     #                 channel_img, channel_img_data = self.load_16bit_image(
#     #                     channel_paths[16])
#     #             if channel_img is None:
#     #                 # TODO current bug where there are 16 bit images in 8 bit folder
#     #                 if os.path.exists(channel_paths[8]):
#     #                     channel_img, channel_img_data = self.load_16bit_image(
#     #                         channel_paths[8])

#     #             if channel_img is not None:
#     #                 current_channel_images[channel] = (
#     #                     channel_img, channel_img_data)

#     #         # Check if we have at least CH0
#     #         if 'CH0' not in current_channel_images:
#     #             break

#     #         # Load bounding boxes
#     #         cur_boxes = {}
#     #         eff_file = os.path.join(
#     #             labels_folder, f"imgNo{img_no}E_t{next_time_point}.txt")
#     #         tar_file = os.path.join(
#     #             labels_folder, f"imgNo{img_no}T_t{next_time_point}.txt")

#     #         if os.path.exists(eff_file):
#     #             with open(eff_file, 'r') as f:
#     #                 lines = f.readlines()
#     #                 cur_boxes["eff"] = [
#     #                     list(map(float, line.strip().split("\t")[1:5])) for line in lines]

#     #         if os.path.exists(tar_file):
#     #             with open(tar_file, 'r') as f:
#     #                 lines = f.readlines()
#     #                 cur_boxes["tar"] = [
#     #                     list(map(float, line.strip().split("\t")[1:5])) for line in lines]

#     #         images.append(current_channel_images['CH0'])
#     #         boxes.append(cur_boxes)
#     #         channel_data.append(current_channel_images)

#     #         time_point += 1

#     #     return images, boxes, channel_data
#     def load_images(self, uid):
#         # Handle tab-separated or float uid
#         if '\t' in uid:
#             uid = uid.split('\t')[0]
#         try:
#             tot_uid = int(float(uid))
#         except ValueError as e:
#             print(f"Error converting uid '{uid}' to float: {e}")
#             return [], [], []

#         # Decode uid components
#         col_val = tot_uid % 100
#         tot_uid = tot_uid // 100  # Use integer division
#         row_val = tot_uid % 100
#         tot_uid = tot_uid // 100  # Use integer division
#         block_val = str(tot_uid)
#         block_val = "0" * (3 - len(block_val)) + block_val
#         img_no = str(10 * (col_val - 1) + row_val)

#         # Base paths for images
#         base_image_folder = os.path.join(
#             self.run_dirname, f"B{block_val}/images")
#         labels_folder = os.path.join(
#             self.run_dirname, f"B{block_val}/labels/DET/FRCNN-Fast/clean/imgNo{img_no}")

#         images = []
#         boxes = []
#         channel_data = []

#         max_workers = 8  # Adjust based on your system's capabilities
#         max_time_points = 1000  # Safety limit to prevent infinite loops
#         futures_dict = {}  # To keep track of time point ordering

#         # Define the worker function that processes a single time point
#         def process_time_point(time_point):
#             # Channel data for this time point
#             current_channel_images = {}

#             # Attempt to load each channel's image
#             for channel in self.channels:
#                 # Prefer 16-bit, fall back to 8-bit
#                 channel_paths = {
#                     16: os.path.join(
#                         base_image_folder, f"crops_16bit_s/imgNo{img_no}{channel}/imgNo{img_no}{channel}_t{time_point}.tif"),
#                     8: os.path.join(
#                         base_image_folder, f"crops_8bit_s/imgNo{img_no}{channel}/imgNo{img_no}{channel}_t{time_point}.tif")
#                 }

#                 channel_img = None
#                 channel_img_data = None

#                 try:
#                     if os.path.exists(channel_paths[16]):
#                         channel_img, channel_img_data = self.load_16bit_image(
#                             channel_paths[16])
#                     if channel_img is None and os.path.exists(channel_paths[8]):
#                         channel_img, channel_img_data = self.load_16bit_image(
#                             channel_paths[8])

#                     if channel_img is not None:
#                         current_channel_images[channel] = (
#                             channel_img, channel_img_data)
#                 except Exception as e:
#                     print(
#                         f"Error loading image for channel {channel}, time point {time_point}: {e}")

#             # Check if we have the required channel
#             if 'CH0' not in current_channel_images:
#                 return time_point, None, None

#             # Load bounding boxes
#             cur_boxes = {}
#             eff_file = os.path.join(
#                 labels_folder, f"imgNo{img_no}E_t{time_point}.txt")
#             tar_file = os.path.join(
#                 labels_folder, f"imgNo{img_no}T_t{time_point}.txt")

#             if os.path.exists(eff_file):
#                 with open(eff_file, 'r') as f:
#                     lines = f.readlines()
#                     cur_boxes["eff"] = [
#                         list(map(float, line.strip().split("\t")[1:5])) for line in lines]

#             if os.path.exists(tar_file):
#                 with open(tar_file, 'r') as f:
#                     lines = f.readlines()
#                     cur_boxes["tar"] = [
#                         list(map(float, line.strip().split("\t")[1:5])) for line in lines]

#             return time_point, current_channel_images, cur_boxes

#         # Use ThreadPoolExecutor for parallel processing
#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             # Submit tasks for a batch of time points
#             time_point = 1
#             batch_size = 20  # Process in batches to avoid memory issues

#             while time_point <= max_time_points and not self.stopped:
#                 # Submit a batch of tasks
#                 batch_end = min(time_point + batch_size, max_time_points + 1)
#                 for tp in range(time_point, batch_end):
#                     future = executor.submit(process_time_point, tp)
#                     futures_dict[future] = tp

#                 # Process completed futures from this batch
#                 for future in as_completed(list(futures_dict.keys())):
#                     tp = futures_dict.pop(future)  # Remove from tracking dict

#                     try:
#                         time_point_idx, current_channel_images, cur_boxes = future.result(
#                             timeout=30)

#                         # If no data for required channel, we're done
#                         if current_channel_images is None:
#                             self.stopped = True
#                             break

#                         # Calculate actual index for ordered insertion
#                         insert_idx = time_point_idx - 1

#                         # Ensure lists have enough slots
#                         while len(images) <= insert_idx:
#                             images.append(None)
#                             boxes.append(None)
#                             channel_data.append(None)

#                         # Insert data at the correct position
#                         images[insert_idx] = current_channel_images['CH0']
#                         boxes[insert_idx] = cur_boxes
#                         channel_data[insert_idx] = current_channel_images

#                     except Exception as e:
#                         print(f"Error processing time point {tp}: {e}")
#                         self.stopped = True
#                         break

#                 # If stopped, break out of the loop
#                 if self.stopped:
#                     break

#                 # Move to the next batch
#                 time_point = batch_end

#         # Filter out any None values (gaps) in our results
#         filtered_results = [(i, b, c) for i, b, c in zip(
#             images, boxes, channel_data) if i is not None]
#         if filtered_results:
#             images, boxes, channel_data = zip(*filtered_results)
#             return list(images), list(boxes), list(channel_data)
#         else:
#             return [], [], []


# ===================================

import os
import cv2
import numpy as np
from glob import glob
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QColor
from concurrent.futures import ThreadPoolExecutor, as_completed


class ImageLoader(QThread):
    finished = pyqtSignal(list, list, list, str, int)
    # Emit progress along with data
    progress = pyqtSignal(int, list, list, list, str)

    def __init__(self, run_dirname, uid, max_time_points):
        super().__init__()
        self.run_dirname = run_dirname
        self.uid = uid
        self.stopped = False
        self.channels = ['CH0', 'CH1', 'CH2', 'CH3']
        self.cores_used = 4
        # Safety limit to prevent infinite loops
        self.max_time_points = max_time_points

    def stop_thread(self):
        self.stopped = True

    def run(self):
        images, boxes, channel_data = self.load_images(self.uid)
        self.finished.emit(images, boxes, channel_data,
                           self.uid, self.max_time_points)

    def load_16bit_image(self, img_path):
        # Read 16-bit image
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"Could not read image: {img_path}")
            return None

        # Normalize to full range
        img_normalized = cv2.normalize(
            img,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U
        )

        return img_normalized, img  # , img

    def load_images(self, uid):
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
        tot_uid = tot_uid // 100  # Use integer division
        row_val = tot_uid % 100
        tot_uid = tot_uid // 100  # Use integer division
        block_val = str(tot_uid)
        block_val = "0" * (3 - len(block_val)) + block_val
        img_no = str(6 * (col_val - 1) + row_val)

        # Base paths for images
        base_image_folder = os.path.join(
            self.run_dirname, f"B{block_val}/images")
        labels_folder = os.path.join(
            self.run_dirname, f"B{block_val}/labels/DET/FRCNN-Fast/clean/imgNo{img_no}")

        images = []
        boxes = []
        channel_data = []

        max_workers = 8  # Adjust based on your system's capabilities

        futures_dict = {}  # To keep track of time point ordering

        # Define the worker function that processes a single time point
        def process_time_point(time_point):
            # Channel data for this time point
            current_channel_images = {}

            # Attempt to load each channel's image
            for channel in self.channels:
                # Prefer 16-bit, fall back to 8-bit
                channel_paths = {
                    16: os.path.join(
                        base_image_folder, f"crops_16bit_s/imgNo{img_no}{channel}/imgNo{img_no}{channel}_t{time_point}.tif"),
                    8: os.path.join(
                        base_image_folder, f"crops_8bit_s/imgNo{img_no}{channel}/imgNo{img_no}{channel}_t{time_point}.tif")
                }

                channel_img = None
                channel_img_data = None

                try:
                    if os.path.exists(channel_paths[16]):
                        channel_img, channel_img_data = self.load_16bit_image(
                            channel_paths[16])
                    if channel_img is None and os.path.exists(channel_paths[8]):
                        channel_img, channel_img_data = self.load_16bit_image(
                            channel_paths[8])

                    if channel_img is not None:
                        current_channel_images[channel] = (
                            channel_img, channel_img_data)
                except Exception as e:
                    print(
                        f"Error loading image for channel {channel}, time point {time_point}: {e}")

            # Check if we have the required channel
            if 'CH0' not in current_channel_images:
                return time_point, None, None

            # Load bounding boxes
            cur_boxes = {}
            eff_file = os.path.join(
                labels_folder, f"imgNo{img_no}E_t{time_point}.txt")
            tar_file = os.path.join(
                labels_folder, f"imgNo{img_no}T_t{time_point}.txt")

            if os.path.exists(eff_file):
                with open(eff_file, 'r') as f:
                    lines = f.readlines()
                    cur_boxes["eff"] = [
                        list(map(float, line.strip().split("\t")[1:5])) for line in lines]

            if os.path.exists(tar_file):
                with open(tar_file, 'r') as f:
                    lines = f.readlines()
                    cur_boxes["tar"] = [
                        list(map(float, line.strip().split("\t")[1:5])) for line in lines]

            return time_point, current_channel_images, cur_boxes

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks for the first 30% of time points (first thread)
            time_point = 1
            batch_size = 20  # Process in batches to avoid memory issues
            total_images = int(self.max_time_points * 1.5)
            loaded_images = 0
            # 30% of the total time points
            first_thread_end = int(0.15 * self.max_time_points)

            # First thread to handle 30% of the work
            while time_point <= first_thread_end and not self.stopped:
                # Submit tasks
                batch_end = min(time_point + batch_size, first_thread_end + 1)
                for tp in range(time_point, batch_end):
                    future = executor.submit(process_time_point, tp)
                    futures_dict[future] = tp

                # Process completed futures from this batch
                for future in as_completed(list(futures_dict.keys())):
                    tp = futures_dict.pop(future)  # Remove from tracking dict

                    try:
                        time_point_idx, current_channel_images, cur_boxes = future.result(
                            timeout=30)

                        # If no data for required channel, we're done
                        if current_channel_images is None:
                            self.stopped = True
                            break

                        # Calculate actual index for ordered insertion
                        insert_idx = time_point_idx - 1

                        # Ensure lists have enough slots
                        while len(images) <= insert_idx:
                            images.append(None)
                            boxes.append(None)
                            channel_data.append(None)

                        # Insert data at the correct position
                        images[insert_idx] = current_channel_images['CH0']
                        boxes[insert_idx] = cur_boxes
                        channel_data[insert_idx] = current_channel_images

                        # Track progress and emit progress at 30%
                        loaded_images += 1
                        # progress_percentage = int(
                        #     (loaded_images / total_images) * 100)
                        # if progress_percentage == 30:
                        #     self.progress.emit(30, list(images), list(
                        #         boxes), list(channel_data), self.uid)  # Emit at 30%

                    except Exception as e:
                        print(f"Error processing time point {tp}: {e}")
                        self.stopped = True
                        break

                if self.stopped:
                    break

                time_point = batch_end
            # Emit progress for 30% completion
            self.progress.emit(30, list(images), list(
                boxes), list(channel_data), self.uid)
            tot_img = len(images)
            images = []
            boxes = []
            channel_data = []

            # Second thread to handle the remaining 70% of the work
            time_point = first_thread_end + 1
            while time_point <= total_images and not self.stopped:
                # Submit tasks for the remaining 70%
                batch_end = min(time_point + batch_size, total_images + 1)
                for tp in range(time_point, batch_end):
                    future = executor.submit(process_time_point, tp)
                    futures_dict[future] = tp

                # Process completed futures from this batch
                for future in as_completed(list(futures_dict.keys())):
                    tp = futures_dict.pop(future)  # Remove from tracking dict

                    try:
                        time_point_idx, current_channel_images, cur_boxes = future.result(
                            timeout=30)

                        if current_channel_images is None:
                            self.stopped = True
                            break

                        insert_idx = time_point_idx - 1
                        while len(images) <= insert_idx:
                            images.append(None)
                            boxes.append(None)
                            channel_data.append(None)

                        images[insert_idx] = current_channel_images['CH0']
                        boxes[insert_idx] = cur_boxes
                        channel_data[insert_idx] = current_channel_images

                        loaded_images += 1
                        # progress_percentage = int(
                        #     (loaded_images / total_images) * 100)

                        # if progress_percentage == 100:
                        #     self.progress.emit(100, list(images), list(
                        #         boxes), list(channel_data))  # Emit at 100%

                    except Exception as e:
                        print(f"Error processing time point {tp}: {e}")
                        self.stopped = True
                        break

                if self.stopped:
                    break

                time_point = batch_end
        self.max_time_points = tot_img + len(images)
        print(self.max_time_points)

        # Filter out any None values (gaps) in our results
        filtered_results = [(i, b, c) for i, b, c in zip(
            images, boxes, channel_data) if i is not None and c is not None]
        if filtered_results:
            images, boxes, channel_data = zip(*filtered_results)
            return list(images), list(boxes), list(channel_data)
        else:
            return [], [], []
