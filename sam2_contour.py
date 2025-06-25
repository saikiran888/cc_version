import os
import math
import shutil
import numpy as np
import cv2
from scipy.interpolate import splprep, splev
import json
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath('./SEGMVC-DevCopy'))
if True:
    from fixClassAndTracks import FixClassTrackingDialog
from skimage.measure import regionprops, find_contours
from csbdeep.utils import normalize
from stardist.models import StarDist2D
import re
from PyQt5.QtCore import Qt, QPoint, QPointF, QRect, QTimer, QRectF, QSizeF
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import time
from itertools import chain
import sip  # Used to check if a widget has been deleted
import uuid
import torch
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QGraphicsOpacityEffect
sys.path.append("/cellchorus/data/shyam/projects/SA/segment-anything/")
from segment_anything import sam_model_registry, SamPredictor
from collections import Counter
from scipy.ndimage import gaussian_filter1d
import numpy as np
sam_checkpoint = "/cellchorus/data/shyam/projects/SA/segment-anything/sam_vit_h_4b8939.pth"
custom_sam_checkpoint = "/cellchorus/data/e/Sai_Desktop/SAM2/segment-anything-2/checkpoints/sam2_multimask_epoch2.pth"
model_type = "vit_h"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # Force use of GPU 4 as device 0
device = "cuda" if torch.cuda.is_available() else "cpu"

def safe_rename_in_folder(folder, renames):
    """
    renames: list of tuples (old_filename, final_filename)
    Renames each file in two phases:
      1. Rename each old file to a unique temporary name.
      2. Rename each temporary file to its final filename.
    Returns a dictionary mapping (old_filename, final_filename) -> count.
    """
    update_info = {}
    temp_mapping = {}
    # Phase 1: Rename each old file to a temporary name.
    for old_filename, final_filename in renames:
        old_path = os.path.join(folder, old_filename)
        temp_filename = f"TMP_{uuid.uuid4().hex}_{old_filename}"
        temp_path = os.path.join(folder, temp_filename)
        try:
            os.rename(old_path, temp_path)
            temp_mapping[temp_filename] = final_filename
            update_info[(old_filename, final_filename)] = update_info.get((old_filename, final_filename), 0) + 1
        except Exception as e:
            print(f"Phase 1 rename failed for {old_filename}: {e}")
    # Phase 2: Rename temporary files to final filenames.
    for temp_filename, final_filename in temp_mapping.items():
        temp_path = os.path.join(folder, temp_filename)
        final_path = os.path.join(folder, final_filename)
        try:
            os.rename(temp_path, final_path)
        except Exception as e:
            print(f"Phase 2 rename failed for {temp_filename}: {e}")
    return update_info
# Helper: Chaikin's corner-cutting smoothing
def chaikin_smoothing(points, iterations=2):
    """
    Apply Chaikin's algorithm to smooth a closed polygon.
    Args:
        points (List[Tuple[int, int]]): Original polygon vertices.
        iterations (int): Number of smoothing passes.
    Returns:
        List[Tuple[int, int]]: Smoothed polygon vertices.
    """
    smoothed = points[:]
    for _ in range(iterations):
        new_pts = []
        n = len(smoothed)
        for i in range(n):
            p0 = smoothed[i]
            p1 = smoothed[(i + 1) % n]
            # Compute Q and R points
            qx = 0.75 * p0[0] + 0.25 * p1[0]
            qy = 0.75 * p0[1] + 0.25 * p1[1]
            rx = 0.25 * p0[0] + 0.75 * p1[0]
            ry = 0.25 * p0[1] + 0.75 * p1[1]
            new_pts.append((int(qx), int(qy)))
            new_pts.append((int(rx), int(ry)))
        smoothed = new_pts
    return smoothed
class SamWorker(QObject):
    predictorReady = pyqtSignal()
    def __init__(self, checkpoint, model_type, device):
        super().__init__()
        self.checkpoint = checkpoint
        self.model_type  = model_type
        self.device      = device
        self.sam         = None
        self.predictor   = None

    @pyqtSlot()
    def loadModel(self):
        # load once, in background thread
        self.sam = sam_model_registry[self.model_type](checkpoint=None)  # Don't load default checkpoint
        self.sam.to(self.device)
        self.predictor = SamPredictor(self.sam)
        self.predictorReady.emit()

    def loadCustomCheckpoint(self, checkpoint_path):
        """Load a custom checkpoint with proper state dict handling"""
        try:
            print(f"Loading custom checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
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
            missing_keys, unexpected_keys = self.sam.load_state_dict(sd, strict=False)
            print(f"Successfully loaded {len(sd)} parameters from checkpoint")
            if missing_keys:
                print(f"Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"Unexpected keys: {len(unexpected_keys)}")

            self.sam.eval()
            print("Custom checkpoint loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading custom checkpoint: {e}")
            return False

    @pyqtSlot(object)
    def setImage(self, blended_rgb: np.ndarray):
        # runs in the same background thread
        if self.predictor is None:
            return
        # warm up predictor on this image
        self.predictor.set_image(blended_rgb)
# =============================================================================
# SwapSelectionDialog and ReorderTrackingDialog (kept for reference)
# =============================================================================
class SwapSelectionDialog(QDialog):
    def __init__(self, tracking_labels, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Swap(s)")
        self.resize(300, 200)
        self.selected_pairs = None
        
        layout = QVBoxLayout(self)
        self.list_widget = QListWidget(self)
        self.list_widget.setSelectionMode(QListWidget.MultiSelection)
        
        self.pairs = []  # Each element: (i, j, "Swap label_i and label_j")
        n = len(tracking_labels)
        for i in range(n):
            for j in range(i+1, n):
                text = f"Swap {tracking_labels[i]} with {tracking_labels[j]}"
                self.pairs.append((i, j, text))
                item = QListWidgetItem(text)
                self.list_widget.addItem(item)
        
        layout.addWidget(self.list_widget)
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK", self)
        cancel_button = QPushButton("Cancel", self)
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
    
    def get_selected_pairs(self):
        selected_items = self.list_widget.selectedItems()
        selected_indices = [self.list_widget.row(item) for item in selected_items]
        return [self.pairs[i][:2] for i in selected_indices]

class ReorderTrackingDialog(QDialog):
    def __init__(self, tracking_labels, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reorder Tracking Labels")
        self.resize(300, 400)
        layout = QVBoxLayout(self)
        
        self.list_widget = QListWidget(self)
        self.list_widget.setDragDropMode(QListWidget.InternalMove)
        for label in tracking_labels:
            item = QListWidgetItem(label)
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)
        
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK", self)
        cancel_button = QPushButton("Cancel", self)
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
    def get_new_order(self):
        return [self.list_widget.item(i).text() for i in range(self.list_widget.count())]

# =============================================================================
# Module-level utility functions
# =============================================================================
def numpy_to_qimage(array: np.ndarray) -> QImage:
    if array is None or len(array.shape) not in [2, 3]:
        raise ValueError(f"Invalid array for QImage conversion. Array shape: {array.shape if array is not None else None}")
    if len(array.shape) == 2:  # Grayscale
        height, width = array.shape
        qimage = QImage(array.data, width, height, width, QImage.Format_Grayscale8)
    elif len(array.shape) == 3:
        height, width, channels = array.shape
        if channels == 3:  # RGB
            qimage = QImage(array.data, width, height, width * 3, QImage.Format_RGB888)
        elif channels == 4:  # RGBA
            qimage = QImage(array.data, width, height, width * 4, QImage.Format_RGBA8888)
        else:
            raise ValueError(f"Unexpected channel count: {channels} in array. Shape: {array.shape}")
    else:
        raise ValueError(f"Unexpected array shape: {array.shape}")
    if qimage.isNull():
        raise ValueError("Conversion to QImage resulted in a null image!")
    return qimage.copy()

def qimage_to_numpy(qimage: QImage) -> np.ndarray:
    qimage = qimage.convertToFormat(QImage.Format_RGBA8888)
    width, height = qimage.width(), qimage.height()
    ptr = qimage.constBits()
    ptr.setsize(height * width * 4)
    return np.array(ptr, dtype=np.uint8).reshape((height, width, 4)).copy()

# =============================================================================
# Main Widget: ContourZoomWidget
# =============================================================================
class ContourZoomWidget(QWidget):
    imageForSam = pyqtSignal(object)
    def __init__(self, base_path: str, label_text: str, masks=None, mask_type: str = None,flagged_frames: list = None, parent: QWidget = None):
        super().__init__(parent)
        self.setWindowFlag(Qt.Window)
        self.setWindowTitle("Contour Zoom")
        self.setAttribute(Qt.WA_DeleteOnClose)
        
        # ???? add these defaults BEFORE you create the slider ????
        self.angular_sigma    = 5      # gaussian kernel s
        self.angular_strength = 1.0    # 0=no smoothing, 1=full smoothing
        self.angular_samples  = 100    # how many angle steps
        # In your ContourZoomWidget __init__ method, add:
        self.tracking_mapping = {}  # In-memory override mapping for tracking labels.
        self.last_saved_frame_label = None
        self.base_path = base_path
        self.label_text = label_text
        self.masks = masks
        self.mask_type = mask_type
        self.region_growth_radius = 80
        self.overlay_channels = {"CH1": True, "CH2": True, "CH3": True}
        self.dead_flags = {}
        self.frame_save_status = {}

        # Parse block, nanowell, frame
        block, nanowell, frame = self.label_text.split("-")
        frame = frame.lstrip("t")
        ch0_path = self.get_image_path(block, nanowell, frame, 'CH0')
        self.ch0_image = cv2.imread(ch0_path, cv2.IMREAD_GRAYSCALE)
        if self.ch0_image is None:
            raise FileNotFoundError(f"CH0 image not found at {ch0_path}")

        self.ch0_pixmap = QPixmap.fromImage(numpy_to_qimage(self.ch0_image))
        ##############SAM###########
        self.samLoaded = False

        # start the SAM thread & loader
        self._samThread = QThread()
        self._samWorker = SamWorker(sam_checkpoint, model_type, device)
        self._samWorker.moveToThread(self._samThread)
        self._samThread.started.connect(self._samWorker.loadModel)
        self._samWorker.predictorReady.connect(self._samThread.quit)
        self._samWorker.predictorReady.connect(self._onSamReady)
        self._samThread.start()
        self.imageForSam.connect(self._samWorker.setImage)
     
        #####################
        # Determine max_frame by scanning CH0 directory
        ch0_dir = os.path.join(self.base_path, f'{block}/images/crops_8bit_s/imgNo{nanowell}CH0')
        frame_files = [f for f in os.listdir(ch0_dir) if f.endswith('.tif')]
        frame_numbers = []
        for f in frame_files:
            m = re.search(r'_t(\d+)\.tif', f)
            if m:
                frame_numbers.append(int(m.group(1)))
        self.max_frame = max(frame_numbers) if frame_numbers else 1
        print(f"Max frame: {self.max_frame}")

        #self.setFixedWidth(600)

        # Timer for elapsed time in title
        self.start_time = time.time()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_title_with_elapsed_time)
        self.timer.start(1000)
        self.setWindowTitle(f"{self.label_text} - 0 sec")

        # Label for current mask type
        self.mask_type_label = QLabel(f"Current Mask Type: {self.mask_type if self.mask_type else 'None'}", self)
        self.mask_type_label.setStyleSheet("font-weight: bold; color: blue;")


        # ?? make ??S?? save *and* load FreeHand masks ??
        self.save_shortcut = QShortcut(QKeySequence("S"), self)
        self.save_shortcut.activated.connect(self._on_save_and_freehand)
        # ????????????????????????????????
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        main_layout.addWidget(self.mask_type_label, alignment=Qt.AlignCenter)

        # Image Display Area (600x600)
        self.image_label = ContourDrawingLabel(self.ch0_image, self.ch0_pixmap, self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)
        #self.image_label.setFixedSize(600, 600)
        main_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        # Enable tracking labels by default.
        self.image_label.show_tracking_labels = True
        self.compute_tracking_labels()
        self.apply_angular_smoothing_to_all()
        self.image_label.update()

        # Navigation Arrows (overlay on image)
        self.leftArrow = QPushButton("<", self.image_label)
        self.leftArrow.setStyleSheet("background-color: rgba(0,0,0,0.5); color: white; border: none; font-size: 20pt;")
        self.leftArrow.setFixedSize(40, 40)
        self.leftArrow.setFlat(True)
        self.leftArrow.clicked.connect(self.go_to_previous_frame)

        self.rightArrow = QPushButton(">", self.image_label)
        self.rightArrow.setStyleSheet("background-color: rgba(0,0,0,0.5); color: white; border: none; font-size: 20pt;")
        self.rightArrow.setFixedSize(40, 40)
        self.rightArrow.setFlat(True)
        self.rightArrow.clicked.connect(self.go_to_next_frame)
        # ?????? Flagged-frame arrows (smaller) ??????
        self.flaggedPrevArrow = QPushButton("<", self.image_label)
        self.flaggedPrevArrow.setToolTip("Go to previous flagged frame")
        self.flaggedPrevArrow.setFlat(True)
        self.flaggedPrevArrow.setFixedSize(30, 30)
        self.flaggedPrevArrow.setStyleSheet(
            "background-color: rgba(0,0,0,0.3); color: white; border: none; font-size: 16pt;"
        )
        self.flaggedPrevArrow.clicked.connect(self.go_to_previous_flagged_frame)

        self.flaggedNextArrow = QPushButton(">", self.image_label)
        self.flaggedNextArrow.setToolTip("Go to next flagged frame")
        self.flaggedNextArrow.setFlat(True)
        self.flaggedNextArrow.setFixedSize(30, 30)
        self.flaggedNextArrow.setStyleSheet(
            "background-color: rgba(0,0,0,0.3); color: white; border: none; font-size: 16pt;"
        )
        self.flaggedNextArrow.clicked.connect(self.go_to_next_flagged_frame)

        # Control Panel (grid layout)
        self.control_widget = QWidget(self)
        #control_widget.setFixedWidth(600)
        grid_layout = QGridLayout(self.control_widget)
        grid_layout.setContentsMargins(5, 5, 5, 5)
        grid_layout.setSpacing(10)
        self.image_label.setMinimumSize(600, 600)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # let the control panel size itself
        self.control_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        # Channels group
        channel_group = QGroupBox("Ch", self)
        channel_layout = QVBoxLayout(channel_group)

        # CH1 button
        self.btn_ch1 = QPushButton("CH1", self)
        self.btn_ch1.setCheckable(True)
        self.btn_ch1.setToolTip("Toggle CH1")
        # initialize from overlay_channels default (True)
        self.btn_ch1.setChecked(self.overlay_channels["CH1"])
        # on toggle, pass the actual state
        self.btn_ch1.toggled.connect(lambda checked: self.toggle_channel("CH1", checked))
        channel_layout.addWidget(self.btn_ch1)

        # CH2 button
        self.btn_ch2 = QPushButton("CH2", self)
        self.btn_ch2.setCheckable(True)
        self.btn_ch2.setToolTip("Toggle CH2")
        self.btn_ch2.setChecked(self.overlay_channels["CH2"])
        self.btn_ch2.toggled.connect(lambda checked: self.toggle_channel("CH2", checked))
        channel_layout.addWidget(self.btn_ch2)

        # CH3 button
        self.btn_ch3 = QPushButton("CH3", self)
        self.btn_ch3.setCheckable(True)
        self.btn_ch3.setToolTip("Toggle CH3")
        self.btn_ch3.setChecked(self.overlay_channels["CH3"])
        self.btn_ch3.toggled.connect(lambda checked: self.toggle_channel("CH3", checked))
        channel_layout.addWidget(self.btn_ch3)

        grid_layout.addWidget(channel_group, 0, 0)
        # Store flagged frames list (expected list of integers)
        self.flagged_frames = sorted(flagged_frames) if flagged_frames else []

        # Setup playback and navigation controls
        self.play_button = QPushButton("Play", self)
        self.play_button.setCheckable(True)
        self.play_button.clicked.connect(self.toggle_play)




        # Contours group
        contour_group = QGroupBox("Contours", self)
        contour_layout = QVBoxLayout(contour_group)

        btn_undo_contour = QPushButton("Undo Contour", self)
        btn_undo_contour.setToolTip("Undo last drawn contour")
        btn_undo_contour.clicked.connect(self.image_label.undo_last_contour)
        contour_layout.addWidget(btn_undo_contour)

        btn_save = QPushButton("Save", self)
        btn_save.setToolTip("Save (by track)")
        btn_save.clicked.connect(self.save_contours)
        contour_layout.addWidget(btn_save)

        btn_clear = QPushButton("Clear", self)
        btn_clear.setToolTip("Clear")
        btn_clear.clicked.connect(self.clear_all_contours)
        contour_layout.addWidget(btn_clear)

        btn_seg = QPushButton("SD", self)
        btn_seg.setToolTip("Segment")
        btn_seg.clicked.connect(self.segment_with_stardist)
        contour_layout.addWidget(btn_seg)

        btn_curate = QPushButton("Curate", self)
        btn_curate.setToolTip("Curate masks to GT")
        btn_curate.setStyleSheet("background-color: green; color: white;")
        btn_curate.clicked.connect(self.curate_masks_to_gt)
        contour_layout.addWidget(btn_curate)
        btn_angular = QPushButton("Smooth ?", self)
        btn_angular.setToolTip("Apply angular smoothing to all contours")
        btn_angular.clicked.connect(self.apply_angular_smoothing_to_all)
        contour_layout.addWidget(btn_angular)

        self.dead_checkbox = QCheckBox("Dead", self)
        self.dead_checkbox.setToolTip("Mark cell as dead. When checked, a 'D' is appended to mask filenames.")
        self.dead_checkbox.setChecked(False)
        self.dead_checkbox.stateChanged.connect(self.on_dead_checkbox_changed)
        contour_layout.addWidget(self.dead_checkbox)

        grid_layout.addWidget(contour_group, 0, 1)

        # Tracking group: contains Run Tracking + Confirm Tracks/Classes
        tracking_group = QGroupBox("Tracking", self)
        tracking_layout = QVBoxLayout(tracking_group)

        run_tracking_button = QPushButton("Run Tracking", self)
        run_tracking_button.setToolTip("Run cell tracking on current view")
        run_tracking_button.clicked.connect(self.run_tracking)
        tracking_layout.addWidget(run_tracking_button)

        #self.confirm_button = QPushButton("Confirm Tracks/Classes", self)
        #self.confirm_button.setToolTip("Commit in-memory track/class changes to disk")
        #self.confirm_button.clicked.connect(self.confirm_tracks_classes)
        #tracking_layout.addWidget(self.confirm_button)

        grid_layout.addWidget(tracking_group, 0, 2)

        # SAM Tools group
        sam_group = QGroupBox("SAM Tools", self)
        sam_layout = QVBoxLayout(sam_group)

        self.box_button = QPushButton("Box", self)
        self.box_button.setCheckable(True)
        self.box_button.toggled.connect(self.toggle_box_mode)

        sam_layout.addWidget(self.box_button)

        btn_run_sam = QPushButton("Run SAM", self)
        btn_run_sam.setToolTip("Run Segment-Anything on your boxes")
        btn_run_sam.clicked.connect(self.run_sam_on_boxes)
        sam_layout.addWidget(btn_run_sam)

        # Add new video prediction button
        btn_run_sam_video = QPushButton("Run SAM Video", self)
        btn_run_sam_video.setToolTip("Run SAM prediction for next 10 frames using drawn boxes")
        btn_run_sam_video.clicked.connect(self.run_sam_video_prediction)
        sam_layout.addWidget(btn_run_sam_video)

        # Add frame count control
        frame_count_layout = QHBoxLayout()
        frame_count_label = QLabel("Frames:", self)
        self.frame_count_spinbox = QSpinBox(self)
        self.frame_count_spinbox.setMinimum(1)
        self.frame_count_spinbox.setMaximum(50)
        self.frame_count_spinbox.setValue(10)
        self.frame_count_spinbox.setToolTip("Number of frames to predict")
        frame_count_layout.addWidget(frame_count_label)
        frame_count_layout.addWidget(self.frame_count_spinbox)
        sam_layout.addLayout(frame_count_layout)

        # Add custom model option
        self.use_custom_model_checkbox = QCheckBox("Use Custom Model", self)
        self.use_custom_model_checkbox.setToolTip("Use custom trained SAM model instead of default")
        self.use_custom_model_checkbox.setChecked(False)
        sam_layout.addWidget(self.use_custom_model_checkbox)

        # Add FreeHand export option
        self.save_as_freehand_checkbox = QCheckBox("Save as FreeHand", self)
        self.save_as_freehand_checkbox.setToolTip("Also save extracted contours as FreeHand masks")
        self.save_as_freehand_checkbox.setChecked(True)
        sam_layout.addWidget(self.save_as_freehand_checkbox)

        # Add Gaussian smoothing control for SAM contours
        gaussian_layout = QHBoxLayout()
        gaussian_label = QLabel("Smooth σ:", self)
        self.gaussian_sigma_spinbox = QDoubleSpinBox(self)
        self.gaussian_sigma_spinbox.setMinimum(0.1)
        self.gaussian_sigma_spinbox.setMaximum(10.0)
        self.gaussian_sigma_spinbox.setValue(2.0)
        self.gaussian_sigma_spinbox.setSingleStep(0.1)
        self.gaussian_sigma_spinbox.setToolTip("Gaussian smoothing strength for SAM contours (higher = smoother)")
        gaussian_layout.addWidget(gaussian_label)
        gaussian_layout.addWidget(self.gaussian_sigma_spinbox)
        sam_layout.addLayout(gaussian_layout)

        grid_layout.addWidget(sam_group, 0, 3)

        smooth_group = QGroupBox("Angular Smooth", self)
        sl = QSlider(Qt.Horizontal, self)
        sl.setRange(0,100)
        sl.setValue(int(self.angular_strength*100))
        sl.setToolTip("0=off ? 100=full smoothing")
        sl.valueChanged.connect(lambda v: self.on_smooth_strength_changed(v/100))
        layout = QVBoxLayout(smooth_group)
        layout.addWidget(sl)
        grid_layout.addWidget(smooth_group, 2, 0, 1, 4)

        # Resize group
        resize_group = QGroupBox("Res", self)
        resize_layout = QVBoxLayout(resize_group)

        btn_r800 = QPushButton("800", self)
        btn_r800.setToolTip("800x800")
        btn_r800.clicked.connect(lambda: self.resize_widget(800, 800))
        resize_layout.addWidget(btn_r800)

        btn_r600 = QPushButton("600", self)
        btn_r600.setToolTip("600x600")
        btn_r600.clicked.connect(lambda: self.resize_widget(600, 600))
        resize_layout.addWidget(btn_r600)

        grid_layout.addWidget(resize_group, 1, 0)

        # Edit group
        edit_group = QGroupBox("Edt", self)
        edit_layout = QVBoxLayout(edit_group)

        self.edit_mode_button = QPushButton("Nudge", self)
        self.edit_mode_button.setCheckable(True)
        self.edit_mode_button.setToolTip("Toggle nudge")
        self.edit_mode_button.toggled.connect(self.toggle_edit_mode)
        edit_layout.addWidget(self.edit_mode_button)

        self.rubber_band_button = QPushButton("Rubber", self)
        self.rubber_band_button.setCheckable(True)
        self.rubber_band_button.setToolTip("Toggle rubber")
        self.rubber_band_button.toggled.connect(self.toggle_rubber_band_mode)
        edit_layout.addWidget(self.rubber_band_button)

        self.region_growth_button = QPushButton("Grow", self)
        self.region_growth_button.setCheckable(True)
        self.region_growth_button.setToolTip("Toggle grow")
        self.region_growth_button.toggled.connect(self.toggle_region_growth_mode)
        edit_layout.addWidget(self.region_growth_button)

        btn_undo_edit = QPushButton("Undo Edit", self)
        btn_undo_edit.setToolTip("Undo last editing action")
        btn_undo_edit.clicked.connect(self.image_label.undo_last_action)
        edit_layout.addWidget(btn_undo_edit)

        grid_layout.addWidget(edit_group, 1, 1)

        # Adjust group
        adjust_group = QGroupBox("Adj", self)
        adjust_layout = QVBoxLayout(adjust_group)

        self.resize_label = QLabel("Res (100%):", self)
        adjust_layout.addWidget(self.resize_label)

        self.resize_slider = QSlider(Qt.Horizontal, self)
        self.resize_slider.setMinimum(70)
        self.resize_slider.setMaximum(125)
        self.resize_slider.setValue(100)
        self.resize_slider.setTickInterval(5)
        self.resize_slider.setToolTip("Size")
        self.resize_slider.setFixedWidth(100)
        self.resize_slider.valueChanged.connect(self.update_contour_scale)
        adjust_layout.addWidget(self.resize_slider)

        self.density_label = QLabel("Dens (100%):", self)
        adjust_layout.addWidget(self.density_label)

        self.density_slider = QSlider(Qt.Horizontal, self)
        self.density_slider.setMinimum(10)
        self.density_slider.setMaximum(200)
        self.density_slider.setValue(100)
        self.density_slider.setTickInterval(10)
        self.density_slider.setToolTip("Density")
        self.density_slider.setFixedWidth(100)
        self.density_slider.valueChanged.connect(self.update_point_density)
        adjust_layout.addWidget(self.density_slider)

        # Region growth radius slider
        # Region growth radius label + slider (both go in the Adjust group)
        radius_label = QLabel("Grow Radius", self)
        adjust_layout.addWidget(radius_label)

        radius_slider = QSlider(Qt.Horizontal, self)
        radius_slider.setRange(10, 200)
        radius_slider.setValue(self.region_growth_radius)
        radius_slider.valueChanged.connect(lambda v: setattr(self, 'region_growth_radius', v))
        adjust_layout.addWidget(radius_slider)

        grid_layout.addWidget(adjust_group, 1, 2)

        # Bulk Update button
        self.bulk_update_button = QPushButton("Bulk Update", self)
        self.bulk_update_button.setToolTip("Change the entire cell label across consecutive frames")
        self.bulk_update_button.clicked.connect(self.propagate_bulk_cell_update)
        grid_layout.addWidget(self.bulk_update_button, 1, 3)

        # Columns' stretch factors
        grid_layout.setColumnStretch(0, 1)
        grid_layout.setColumnStretch(1, 2)
        grid_layout.setColumnStretch(2, 1)
        grid_layout.setColumnStretch(3, 1)

        main_layout.addWidget(self.control_widget)
        self.setLayout(main_layout)



        self.frame_slider = QSlider(Qt.Horizontal, self)
        self.frame_slider.setMinimum(1)
        self.frame_slider.setMaximum(self.max_frame)
        self.frame_slider.setValue(int(self.label_text.split("-")[-1].lstrip("t")))
        self.frame_slider.valueChanged.connect(self.slider_value_changed)

        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.play_button)
        nav_layout.addWidget(self.frame_slider)
        main_layout.addLayout(nav_layout)

        # Styling
        self.setStyleSheet("""
            QWidget {
                background-color: #f8f8f8;
                font-family: Arial, sans-serif;
            }
            QGroupBox {
                border: 1px solid #aaa;
                border-radius: 4px;
                margin-top: 5px;
                background-color: #fff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 3px;
                font-weight: bold;
                color: #444;
            }
            QPushButton {
                background-color: #e6e6e6;
                color: #333;
                border: 1px solid #bbb;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 9pt;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
            QPushButton:pressed {
                background-color: #bcbcbc;
            }
            QPushButton:checked {
                background-color: #66bb6a;
                color: white;
                border: 1px solid #388e3c;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999;
                height: 6px;
                background: #ddd;
                margin: 2px 0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #aaa;
                border: 1px solid #888;
                width: 12px;
                margin: -2px 0;
                border-radius: 6px;
            }
        """)

        # If we have default "Track" masks, apply them
        if self.masks and "Track" in self.masks:
            self.mask_type = "Track"
            self.image_label.current_mask_type = "Track"
            self.image_label.apply_masks_to_pixmap(self.masks["Track"], "Track")

        # Apply fluorescence overlay
        self.apply_fluorescence_overlay()

        # Recompute tracking labels
        self.compute_tracking_labels()
        self.apply_angular_smoothing_to_all()
        self.image_label.update()

        # Default Mask Type combo
        default_mask_group = QGroupBox("Default Mask Type", self)
        default_mask_layout = QHBoxLayout(default_mask_group)
        self.mask_combo = QComboBox(self)
        # Add all valid mask options
        self.mask_combo.addItems(["MRCNN", "FRCNN", "SAM", "FreeHand", "GT", "CT", "Track"])
        # Initial default selection
        self.mask_combo.setCurrentText("MRCNN")
        # Update current mask type
        self.mask_combo.currentTextChanged.connect(self.update_default_mask_type)
        default_mask_layout.addWidget(self.mask_combo)
        main_layout.addWidget(default_mask_group, alignment=Qt.AlignCenter)

    def jump_to_frame(self, new_frame_num: int):
        """
        Helper to jump directly to a specific frame number.
        """
        block, nanowell, _ = self.label_text.split("-")
        # Update label_text and title
        self.label_text = f"{block}-{nanowell}-t{new_frame_num}"
        self.setWindowTitle(self.label_text)

        # Load and display the new frame (mirrors slider behavior)
        ch0_path = self.get_image_path(block, nanowell, str(new_frame_num), 'CH0')
        self.ch0_image = cv2.imread(ch0_path, cv2.IMREAD_GRAYSCALE)
        if self.ch0_image is None:
            print(f"CH0 image not found at {ch0_path}")
            return
        self.ch0_pixmap = QPixmap.fromImage(numpy_to_qimage(self.ch0_image))
        self.image_label.update_pixmap(self.ch0_pixmap)
        self.image_label.pixmap_original = self.ch0_pixmap
        self.image_label.clear_contours()
        self.apply_fluorescence_overlay()
        self.switch_mask_type(self.mask_type)
        self.image_label.current_mask_type = self.mask_type

        # Reset resolution slider and smoothing baseline
        if hasattr(self, 'original_contours'):
            del self.original_contours
        self.resize_slider.setValue(100)

        # Warm up SAM and reset timer
        self.imageForSam.emit(self.fluorescence_overlay)
        self.start_time = time.time()

        # Update the main frame slider without triggering its signal
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(new_frame_num)
        self.frame_slider.blockSignals(False)

    def go_to_next_flagged_frame(self):
        """
        Navigate to the next flagged frame in the list.
        """
        current = int(self.label_text.split("-")[-1].lstrip("t"))
        for f in self.flagged_frames:
            if f > current:
                self.jump_to_frame(f)
                return
        QMessageBox.information(self, "Navigation", "No next flagged frame.")

    def go_to_previous_flagged_frame(self):
        """
        Navigate to the previous flagged frame in the list.
        """
        current = int(self.label_text.split("-")[-1].lstrip("t"))
        for f in reversed(self.flagged_frames):
            if f < current:
                self.jump_to_frame(f)
                return
        QMessageBox.information(self, "Navigation", "No previous flagged frame.")
    def _on_save_and_freehand(self):
        # 1) save raw contours
        self.save_contours()
        # 2) switch to FreeHand
        self.switch_mask_type("FreeHand")
        self.mask_type = "FreeHand"
        self.image_label.current_mask_type = "FreeHand"
        print("Saved + loaded FreeHand masks")

    def scaled_boxes(self):
        """Convert widget-coords QRect ? full-res [x1,y1,x2,y2] list."""
        W, H = self.ch0_image.shape[1], self.ch0_image.shape[0]
        pw, ph = self.image_label.pixmap().width(), self.image_label.pixmap().height()
        off_x = (self.image_label.width()  - pw) / 2
        off_y = (self.image_label.height() - ph) / 2
        sx, sy = W / pw, H / ph

        boxes = []
        for rect in self.image_label.sam_boxes:
            x1 = (rect.left()   - off_x) * sx
            y1 = (rect.top()    - off_y) * sy
            x2 = (rect.right()  - off_x) * sx
            y2 = (rect.bottom() - off_y) * sy
            boxes.append([x1, y1, x2, y2])
        return boxes
    def _onSamReady(self):
        self.samLoaded = True
        print("? SAM predictor loaded and ready to warm-up per frame.")

    def run_sam_on_boxes(self):
        if not self.samLoaded:
            QMessageBox.warning(self, "SAM", "Still loading SAM model ?? please wait.")
            return

        boxes = self.scaled_boxes()
        if not boxes:
            QMessageBox.warning(self, "SAM", "Draw at least one box first.")
            return

        # 0) make sure the predictor has the image
        self._samWorker.predictor.set_image(self.fluorescence_overlay)

        # 1) convert to torch boxes
        input_boxes = torch.tensor(boxes, device=device).float()
        H, W = self.fluorescence_overlay.shape[:2]
        transformed = self._samWorker.predictor.transform.apply_boxes_torch(input_boxes, (H, W))

        # 2) predict
        masks, _, _ = self._samWorker.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed,
            multimask_output=False
        )

        # 3) save out exactly like you did before
        block, nanowell, frame = self.label_text.split("-")
        sam_dir = os.path.join(
            self.base_path, block, "Masks", "SAM",
            f"imgNo{nanowell}CH0", f"t{frame.lstrip('t')}"
        )
        os.makedirs(sam_dir, exist_ok=True)
        for idx, mask in enumerate(masks):
            mask_file = os.path.join(sam_dir, f"{self.label_text}_mask{idx}.npy")
            np.save(mask_file, mask.cpu().numpy())
            print(f"Saved SAM mask to {mask_file}")

        # 4) reload and display
        self.switch_mask_type("SAM")
        QMessageBox.information(self, "SAM", "Finished ?? SAM masks loaded.")

        # 5) clear all drawn boxes now that we're done
        self.image_label.sam_boxes.clear()
        # optionally turn off Box-mode so you??t immediately draw more boxes
        if self.box_button.isChecked():
            self.box_button.setChecked(False)
        self.image_label.update()


    def on_sam_finished(self, exitCode, exitStatus):
        QApplication.restoreOverrideCursor()
        if exitCode != 0:
            QMessageBox.critical(self, "SAM", f"SAM failed (exit {exitCode}).")
            return
        # reload and display the new SAM masks
        self.switch_mask_type("SAM")
        QMessageBox.information(self, "SAM", "Finished ?? SAM masks loaded.")

    def on_smooth_strength_changed(self, strength):
        self.angular_strength = strength
        self.apply_angular_smoothing_to_all()

    # ------------------------- Playback and Navigation -------------------------
    def toggle_play(self):
        if self.play_button.isChecked():
            self.play_button.setText("Pause")
            self.start_playback()
        else:
            self.play_button.setText("Play")
            self.stop_playback()

    def start_playback(self):
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.advance_frame)
        self.play_timer.start(1000)

    def stop_playback(self):
        if hasattr(self, 'play_timer'):
            self.play_timer.stop()

    def advance_frame(self):
        block, nanowell, frame = self.label_text.split("-")
        frame_num = int(frame.lstrip("t"))
        if frame_num < self.max_frame:
            self.go_to_next_frame()
            self.frame_slider.setValue(frame_num + 1)
        else:
            self.stop_playback()
            self.play_button.setChecked(False)
            self.play_button.setText("Play")

    def slider_value_changed(self, value):
        block, nanowell, _ = self.label_text.split("-")
        new_label_text = f"{block}-{nanowell}-t{value}"
        self.label_text = new_label_text
        self.setWindowTitle(new_label_text)
        ch0_path = self.get_image_path(block, nanowell, str(value), 'CH0')
        self.ch0_image = cv2.imread(ch0_path, cv2.IMREAD_GRAYSCALE)
        if self.ch0_image is None:
            print(f"CH0 image not found at {ch0_path}")
            return
        self.ch0_pixmap = QPixmap.fromImage(numpy_to_qimage(self.ch0_image))
        self.image_label.update_pixmap(self.ch0_pixmap)
        self.image_label.pixmap_original = self.ch0_pixmap
        self.image_label.clear_contours()
        self.imageForSam.emit(self.fluorescence_overlay)

        self.switch_mask_type(self.mask_type)

    def update_title_with_elapsed_time(self):
        elapsed = time.time() - self.start_time
        self.setWindowTitle(f"{self.label_text} - {elapsed:.0f} sec")

    def on_dead_checkbox_changed(self, state):
        current_index = self.image_label.selected_contour_index
        if current_index != -1:
            self.dead_flags[current_index] = (state == Qt.Checked)


    def closeEvent(self, event):
        # 1) stop the title-update timer
        if hasattr(self, 'timer'):
            self.timer.stop()

        # 2) cleanly quit & wait for the SAM thread
        if hasattr(self, '_samThread') and self._samThread.isRunning():
            self._samThread.quit()
            # give it up to 2 s to finish
            if not self._samThread.wait(2000):
                print("Warning: SAM thread did not quit in time")
        # now call the base class
        super().closeEvent(event)
    def toggle_edit_mode(self, enabled):
        """
        Enable or disable nudge/edit mode. When turning on, automatically
        turn off box-drawing, rubber-band edit, and region-growth modes.
        """
        # 1) Set the flag on the drawing label
        self.image_label.edit_mode = enabled
        # 2) Keep the button labeled ??Nudge??
        self.edit_mode_button.setText("Nudge")
        # 3) If enabling edit, disable the other modes on the label
        if enabled:
            # clear any partial box-draw
            if hasattr(self, 'box_button') and self.box_button.isChecked():
                self.box_button.setChecked(False)
            # clear any rubber-band selection
            if self.rubber_band_button.isChecked():
                self.rubber_band_button.setChecked(False)
            # clear any region-growth
            if self.region_growth_button.isChecked():
                self.region_growth_button.setChecked(False)
            # clear any selected points
            self.image_label.selected_points.clear()
            # clear any selected contour index
            self.image_label.selected_contour_index = -1

        # 4) refresh focus & redraw
        self.image_label.setFocus()
        self.image_label.update()


    def toggle_box_mode(self, enabled):
        """
        Enable or disable SAM-box draw mode. When turning on, automatically
        turn off nudge/edit, rubber-band edit, and region-growth modes.
        """
        # 1) Set the flag on the drawing label
        self.image_label.box_mode = enabled
        # 2) Keep the button labeled ??Box??
        self.box_button.setText("Box")
        # 3) If enabling box-draw, disable the other modes
        if enabled:
            # turn off nudge/edit
            if self.edit_mode_button.isChecked():
                self.edit_mode_button.setChecked(False)
            # turn off rubber-band
            if self.rubber_band_button.isChecked():
                self.rubber_band_button.setChecked(False)
            # turn off region-growth
            if self.region_growth_button.isChecked():
                self.region_growth_button.setChecked(False)

        # 4) redraw so any unfinished box is cleared
        self.image_label.update()


    def get_image_path(self, block, nanowell, frame, channel):
        return os.path.join(
            self.base_path,
            f'{block}/images/crops_8bit_s/imgNo{nanowell}{channel}/imgNo{nanowell}{channel}_t{frame}.tif'
        )
    # And update your toggle_channel method to:

    def toggle_channel(self, channel: str, enabled: bool):
        """
        Turn the given fluorescence channel overlay on or off
        according to the button's checked state.
        """
        self.overlay_channels[channel] = enabled
        print(f"{channel} overlay {'enabled' if enabled else 'disabled'}.")
        self.apply_fluorescence_overlay()

    def apply_fluorescence_overlay(self, include_masks=False):
        try:
            if not any(self.overlay_channels.values()):
                rgb = cv2.cvtColor(self.ch0_image, cv2.COLOR_GRAY2RGB)
                qim = QImage(rgb.data,
                            rgb.shape[1],
                            rgb.shape[0],
                            rgb.shape[1]*3,
                            QImage.Format_RGB888)
                self.image_label.update_pixmap(QPixmap.fromImage(qim.copy()))
                return
            block, nanowell, frame = self.label_text.split("-")
            frame_numeric = frame.lstrip("t")
            ch0_path = self.get_image_path(block, nanowell, frame_numeric, 'CH0')
            ch1_path = ch0_path.replace('CH0', 'CH1')
            ch2_path = ch0_path.replace('CH0', 'CH2')
            ch3_path = ch0_path.replace('CH0', 'CH3')

            ch0_image = cv2.imread(ch0_path, cv2.IMREAD_GRAYSCALE)
            ch1_image = cv2.imread(ch1_path, cv2.IMREAD_GRAYSCALE) if self.overlay_channels.get("CH1", True) else None
            ch2_image = cv2.imread(ch2_path, cv2.IMREAD_GRAYSCALE) if self.overlay_channels.get("CH2", True) else None
            ch3_image = cv2.imread(ch3_path, cv2.IMREAD_GRAYSCALE) if self.overlay_channels.get("CH3", True) else None

            if ch0_image is None:
                print("CH0 image not found.")
                return

            ch1_image = ch1_image if ch1_image is not None else np.zeros_like(ch0_image)
            ch2_image = ch2_image if ch2_image is not None else np.zeros_like(ch0_image)
            ch3_image = ch3_image if ch3_image is not None else np.zeros_like(ch0_image)

            ch0_norm = ch0_image / 255.0
            ch1_norm = ch1_image / 255.0
            ch2_norm = ch2_image / 255.0
            ch3_norm = ch3_image / 255.0      # <?? normalize CH3
            blended = np.stack([ch0_norm, ch0_norm, ch0_norm], axis=-1)
            blended[..., 1] += 0.5 * ch1_norm
            blended[..., 0] += 0.5 * ch2_norm
            blended[..., 2] += 0.8 * ch3_norm  # blue (new!)
            blended = np.clip(blended, 0, 1)
            blended_overlay = (blended * 255).astype(np.uint8)

            self.fluorescence_overlay = blended_overlay
            if sip.isdeleted(self.image_label):
                print("image_label has been deleted, skipping mask application.")
                return
            self.image_label.update_pixmap(QPixmap.fromImage(numpy_to_qimage(blended_overlay)))
            if include_masks and self.mask_type in self.masks:
                self.image_label.apply_masks_to_pixmap(self.masks[self.mask_type], self.mask_type)
        except Exception as e:
            print(f"Error in fluorescence overlay: {e}")

    def segment_with_stardist(self):
        import os, tensorflow as tf
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        tf.config.set_visible_devices([], 'GPU')
        print("Segmenting using StarDist...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            block, nanowell, frame = self.label_text.split("-")
            frame_numeric = frame.lstrip("t")
            ch1_path = os.path.join(
                self.base_path,
                f'{block}/images/crops_8bit_s/imgNo{nanowell}CH1/imgNo{nanowell}CH1_t{frame_numeric}.tif')
            ch2_path = os.path.join(
                self.base_path,
                f'{block}/images/crops_8bit_s/imgNo{nanowell}CH2/imgNo{nanowell}CH2_t{frame_numeric}.tif')
            ch1_frame = cv2.imread(ch1_path, cv2.IMREAD_GRAYSCALE)
            ch2_frame = cv2.imread(ch2_path, cv2.IMREAD_GRAYSCALE)
            if ch1_frame is None or ch2_frame is None:
                print("CH1 or CH2 image not found for segmentation.")
                return

            if not hasattr(self, 'stardist_model'):
                print("Loading StarDist model...")
                self.stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')

            ch1_norm = normalize(ch1_frame.astype(np.float32), 1, 99.8)
            ch2_norm = normalize(ch2_frame.astype(np.float32), 1, 99.8)

            labels_ch1 = self.stardist_model.predict_instances(ch1_norm, prob_thresh=0.8, nms_thresh=0.4)[0]
            labels_ch2 = self.stardist_model.predict_instances(ch2_norm, prob_thresh=0.8, nms_thresh=0.4)[0]
            labels_dict = {"ch1": labels_ch1, "ch2": labels_ch2}

            label_width = self.image_label.width()
            label_height = self.image_label.height()

            contours = []
            for key, label_img in labels_dict.items():
                for region in regionprops(label_img):
                    if region.area < 10:
                        continue
                    mask = label_img == region.label
                    region_contours = find_contours(mask, 0.5)
                    for cnt in region_contours:
                        qpoints = [QPoint(
                            int(pt[1] * label_width / mask.shape[1]),
                            int(pt[0] * label_height / mask.shape[0])
                        ) for pt in cnt]
                        if len(qpoints) > 1:
                            contours.append(qpoints)
            self.image_label.contours = contours
            if not sip.isdeleted(self.image_label):
                self.image_label.update_pixmap(self.image_label.pixmap())
            print(f"StarDist segmentation complete with {len(contours)} contours.")
        finally:
            # restore the normal cursor
            QApplication.restoreOverrideCursor()


    def clear_all_contours(self):
        self.image_label.clear_contours()

    def update_contour_scale(self, value):
        scale_factor = value / 100.0
        self.image_label.scale_selected_contour(scale_factor)
        self.resize_label.setText("Res (100%):" if value == 100 else f"Res ({value}%):")

    def update_point_density(self, value):
        self.density_label.setText("Dens (100%):" if value == 100 else f"Dens ({value}%):")

    # inside ContourZoomWidget:
    def update_contour_scale(self, value):
        """
        Called by the slider??s valueChanged.
        Pass the raw slider value (70??125) to the label,
        then update the label text.
        """
        # pass the integer percentage, not value/100.0
        self.image_label.scale_selected_contour(value)
        self.resize_label.setText(
            "Res (100%):" if value == 100 else f"Res ({value}%):"
        )


    def toggle_rubber_band_mode(self, enabled):
        """
        Enable or disable rubber-band edit mode. If turning on, automatically
        turn off region-growth mode.
        """
        # Set the flag in the drawing label
        self.image_label.rubber_band_mode = enabled
        # Update button text
        self.rubber_band_button.setText("Rubber")
        if enabled:
            # If edit-nudge mode is on, turn it off
            if self.edit_mode_button.isChecked():
                self.edit_mode_button.setChecked(False)
            # If region-growth mode is on, turn it off
            if self.region_growth_button.isChecked():
                self.region_growth_button.setChecked(False)

    def toggle_region_growth_mode(self, enabled):
        """
        Enable or disable region-growth mode. If turning on, automatically
        turn off rubber-band edit mode.
        """
        # Set the flag in the drawing label
        self.image_label.region_growth_mode = enabled
        # Update button text
        self.region_growth_button.setText("Grow")
        if enabled:
            # If edit-nudge mode is on, turn it off
            if self.edit_mode_button.isChecked():
                self.edit_mode_button.setChecked(False)
            # If rubber-band mode is on, turn it off
            if self.rubber_band_button.isChecked():
                self.rubber_band_button.setChecked(False)

    def resize_widget(self, width, height):
        # 1) Lock the image display area to exactly width??height
        self.image_label.setFixedSize(width, height)

        # 2) Re-draw the current pixmap into the new label size
        #    (this triggers the same scaling logic you have in resizeEvent)
        scaled = self.ch0_pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.update_pixmap(scaled)

        # 3) Finally, resize the window to fit its contents
        self.adjustSize()



    def should_warn_about_inconsistent_masks(self, block, nanowell):
        """
        Checks if the number of masks across frames varies and if all frames have consistent filenames.
        Returns True if warning should be shown, False otherwise.
        """
        masks_dir_base = os.path.join(self.base_path, block, "Masks", "FreeHand", f"imgNo{nanowell}CH0")
        if not os.path.isdir(masks_dir_base):
            return False  # no folder to scan

        frame_dirs = [d for d in os.listdir(masks_dir_base) if d.startswith("t") and os.path.isdir(os.path.join(masks_dir_base, d))]
        all_counts = []
        filename_sets = []

        for fd in frame_dirs:
            path = os.path.join(masks_dir_base, fd)
            npy_files = [f for f in os.listdir(path) if f.endswith(".npy")]
            all_counts.append(len(npy_files))
            filename_sets.append(set(npy_files))

        if not all_counts:
            return False

        mode_count = Counter(all_counts).most_common(1)[0][0]
        has_count_inconsistency = any(c != mode_count for c in all_counts)
        has_filename_inconsistency = len(set(map(tuple, filename_sets))) > 1

        return has_count_inconsistency or has_filename_inconsistency


    def run_tracking(self):
        """
        Launches the tracking script by passing base_path, block, nanowell, and mask_type.
        Checks for FreeHand mask inconsistencies (file count and names) before proceeding.
        """
        try:
            block, nanowell, _ = self.label_text.split("-")
        except Exception as e:
            print("Error extracting block and nanowell:", e)
            return

        # ----------------- Sanity Check: FreeHand mask consistency -----------------
        if self.mask_type == "FreeHand":
            from collections import Counter

            mask_base = os.path.join(self.base_path, block, "Masks", "FreeHand", f"imgNo{nanowell}CH0")
            frame_dirs = sorted([d for d in os.listdir(mask_base) if d.startswith("t") and os.path.isdir(os.path.join(mask_base, d))])

            file_counts = []
            filename_sets = []
            frame_nums = []

            for fd in frame_dirs:
                fnum = int(fd.lstrip("t"))
                path = os.path.join(mask_base, fd)
                npy_files = sorted([f for f in os.listdir(path) if f.endswith(".npy")])
                file_counts.append(len(npy_files))
                filename_sets.append(tuple(sorted(npy_files)))
                frame_nums.append(fnum)

            mode_count = Counter(file_counts).most_common(1)[0][0]
            most_common_filenames = Counter(filename_sets).most_common(1)[0][0]

            inconsistent_count_frames = [f for f, c in zip(frame_nums, file_counts) if c != mode_count]
            inconsistent_name_frames = [f for f, names in zip(frame_nums, filename_sets) if names != most_common_filenames]

            if inconsistent_count_frames or inconsistent_name_frames:
                msg = "?? **Mask Inconsistency Detected**\n\n"
                if inconsistent_count_frames:
                    msg += f"- Inconsistent number of `.npy` files (expected {mode_count}) in frames:\n  " \
                        f"{', '.join(['t' + str(f) for f in inconsistent_count_frames])}\n"
                if inconsistent_name_frames:
                    msg += f"- Mismatch in file *names* in frames:\n  " \
                        f"{', '.join(['t' + str(f) for f in inconsistent_name_frames])}\n"

                msg += "\nTracking expects the **same number and names** of `.npy` files across frames.\n\nProceed anyway?"

                reply = QMessageBox.question(
                    self,
                    "FreeHand Mask Warning",
                    msg,
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    print("Tracking aborted due to mask inconsistencies.")
                    return

        # ---------------------------------------------------------------------------

        tracking_script = "/cellchorus/src/multiversion_testing/DEEP-TIMING/sandbox/DT5-viewer/AnnotationMasterDev/Tracking_ann.py"
        viewer_python = "/home/shyam/.conda/envs/viewer/bin/python"
        arguments = [tracking_script, self.base_path, block, nanowell, self.mask_type if self.mask_type else "None"]

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.tracking_process = QProcess(self)
        self.tracking_process.finished.connect(self.tracking_finished)

        print(f"Run Tracking command executed: {viewer_python} {arguments}")
        self.tracking_process.start(viewer_python, arguments)


    def tracking_finished(self, exitCode, exitStatus):
        QApplication.restoreOverrideCursor()
        print("Tracking process finished with exit code:", exitCode)

        # Create a label for the status update.
        done_label = QLabel("Tracking Done", self)
        done_label.setStyleSheet("font-size: 16pt; color: red; background-color: rgba(255, 255, 255, 0.5);")
        done_label.setAlignment(Qt.AlignCenter)
        done_label.setGeometry(self.rect())
        done_label.show()

        # Start an opacity animation (non-blocking)
        opacity_effect = QGraphicsOpacityEffect(done_label)
        done_label.setGraphicsEffect(opacity_effect)
        animation = QPropertyAnimation(opacity_effect, b"opacity")
        animation.setDuration(3000)  # 3 seconds
        animation.setStartValue(1.0)
        animation.setEndValue(0.0)
        animation.setEasingCurve(QEasingCurve.OutQuad)
        animation.start()

        # Schedule label removal after 3 seconds using QTimer.singleShot.
        QTimer.singleShot(3000, done_label.deleteLater)

 


    def resizeEvent(self, event):
        super().resizeEvent(event)
        new_size = self.image_label.size()
        self.image_label.scaled_pixmap = self.ch0_pixmap.scaled(new_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        if not sip.isdeleted(self.image_label):
            self.image_label.update_pixmap(self.image_label.scaled_pixmap)
        arrow_y = (self.image_label.height() - self.leftArrow.height()) // 2
        self.leftArrow.move(10, arrow_y)
        self.rightArrow.move(self.image_label.width() - self.rightArrow.width() - 10, arrow_y)
        # flagged-frame arrows (just inside of those)
        self.flaggedPrevArrow.move(60, arrow_y)
        self.flaggedNextArrow.move(
            self.image_label.width() - self.flaggedNextArrow.width() - 60,
            arrow_y
        )
    # ------------------------- Hotkey-Driven Actions -------------------------
    # In ContourZoomWidget:
    def keyPressEvent(self, event):
        from PyQt5.QtCore import Qt
        key = event.key()
                # ??0?? = turn off all masks (reset to just the fluorescence overlay)
        if key == Qt.Key_0:
            # clear any mask-type overlay:
            self.mask_type = None
            self.image_label.current_mask_type = None
            self.mask_type_label.setText("Current Mask Type: None")
            # clear all drawn or loaded contours:
            self.image_label.contours = []
            self.image_label.raw_contours = []
            # reset to the base pixmap + fluorescence overlay
            self.image_label.update_pixmap(self.ch0_pixmap)
            self.apply_fluorescence_overlay()
            event.accept()
            return
        # ? / ? for next/previous frame
        if event.key() == Qt.Key_Left:
            self.go_to_previous_frame()
            event.accept()
            return
        if event.key() == Qt.Key_Right:
            self.go_to_next_frame()
            event.accept()
            return

        # ? / ? for next/previous flagged frame
        if event.key() == Qt.Key_Up:
            self.go_to_previous_flagged_frame()
            event.accept()
            return
        if event.key() == Qt.Key_Down:
            self.go_to_next_flagged_frame()
            event.accept()
            return
        # ????????????????????????????????????????????????????????????
        # Pressing ??S?? now saves contours
        if key == Qt.Key_S:
            self.save_contours()
            event.accept()
            return
        # ????????????????????????????????????????????????????????????

        # Ctrl+I toggles multi-select
        if event.modifiers() & Qt.ControlModifier and key == Qt.Key_I:
            self.image_label.setMultiSelectionMode(not self.image_label.multi_selection_mode)
            print(f"Multi-selection mode {'ON' if self.image_label.multi_selection_mode else 'OFF'}.")
            event.accept()
            return

        # F swaps when multi-select has =2
        if key == Qt.Key_F:
            if self.image_label.multi_selection_mode and len(self.image_label.multi_selected_indices) >= 2:
                self.propagate_tracking_swap()
            else:
                print("Need at least two contours selected to swap.")
            event.accept()
            return

        # C = fix cell type, only if a contour is selected
        if key == Qt.Key_C:
            if self.image_label.selected_contour_index == -1:
                print("No contour selected; ignoring C.")
            else:
                self.propagate_celltype_fix()
            event.accept()
            return

        # 1??6 or G switch mask types
        mask_mapping = {
            Qt.Key_1: "MRCNN",
            Qt.Key_2: "FRCNN",
            Qt.Key_3: "SAM",
            Qt.Key_4: "FreeHand",
            Qt.Key_G: "GT",
            Qt.Key_5: "CT",
            Qt.Key_6: "Track"
        }
        if key in mask_mapping:
            new_mask = mask_mapping[key]
            if self.mask_type == new_mask:
                self.reset_to_original_image()
                self.mask_type = None
                self.image_label.current_mask_type = None
                print(f"Toggled off {new_mask}")
            else:
                self.switch_mask_type(new_mask)
                self.mask_type = new_mask
                self.image_label.current_mask_type = new_mask
                print(f"Switched to {new_mask}")
            event.accept()
            return

        # X = delete selected contour, only if one is selected
        if key == Qt.Key_X:
            if self.image_label.selected_contour_index == -1:
                print("No contour selected; ignoring X.")
            else:
                idx = self.image_label.selected_contour_index
                self.image_label.contours.pop(idx)
                if idx < len(self.image_label.raw_contours):
                    self.image_label.raw_contours.pop(idx)
                self.image_label.selected_contour_index = -1
                self.image_label.update()
            event.accept()
            return

        # fallback
        super().keyPressEvent(event)




    def reset_to_original_image(self):
        print("Resetting to original image...")
        self.overlay_channels = {ch: True for ch in self.overlay_channels}
        self.image_label.contours = []
        self.image_label.update_pixmap(self.ch0_pixmap)
        self.apply_fluorescence_overlay()
        print("Reset complete.")





    

    def get_mask_paths(self, block, nanowell, frame, mask_type: str):
        mask_dirs = {
            'SAM': 'SAM',
            'MRCNN': 'MRCNN',
            'FRCNN': 'FRCNN',
            'FreeHand': 'FreeHand',
            'GT': 'GT',
            'CT': 'CT',
            'Track': 'Track',
        }
        mask_dir = mask_dirs.get(mask_type)
        if not mask_dir:
            print(f"Unsupported mask type: {mask_type}")
            return []
        folder_path = os.path.join(
            self.base_path,
            block,
            "Masks",
            mask_dir,
            f"imgNo{nanowell}CH0",
            f"t{frame}"
        )
        if not os.path.exists(folder_path):
            print(f"Mask folder not found: {folder_path}")
            return []
        return [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if f.endswith('.npy')]

    def load_previous_freehand_masks(self):
        """
        Load the FreeHand .npy masks from the prior frame
        and hand them through switch_mask_type??so they get interpolated & smoothed.
        """
        block, nanowell, frame = self.label_text.split("-")
        prev = int(frame.lstrip("t")) - 1
        if prev < 1:
            return

        mask_paths = self.get_mask_paths(block, nanowell, str(prev), "FreeHand")
        if not mask_paths:
            return

        # stash them temporarily and switch view
        masks = []
        for p in mask_paths:
            try:
                masks.append(np.load(p).astype(np.uint8))
            except:
                pass

        # use the same pipeline
        self.masks = getattr(self, "masks", {})
        self.masks["FreeHand"] = masks
        self.switch_mask_type("FreeHand")

    def save_masks_without_resmoothing(self, block, nanowell, frame):
        save_dir = os.path.join(self.base_path, block, "Masks", "FreeHand", f"imgNo{nanowell}CH0", f"t{frame}")
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        H, W = self.ch0_image.shape

        pix = self.image_label.pixmap()
        pw, ph = pix.width(), pix.height()
        lw, lh = self.image_label.width(), self.image_label.height()
        off_x = (lw - pw) / 2.0
        off_y = (lh - ph) / 2.0
        sx, sy = W / pw, H / ph

        raw_list = []
        for contour in self.image_label.contours:
            pts = []
            for pt in contour:
                x_img = (pt.x() - off_x) * sx
                y_img = (pt.y() - off_y) * sy
                pts.append((x_img, y_img))
            raw_list.append(pts)

        np.save(os.path.join(save_dir, "raw_contours.npy"), np.array(raw_list, dtype=object), allow_pickle=True)

        for i, pts in enumerate(raw_list):
            arr = np.array([(int(round(x)), int(round(y))) for x, y in pts], np.int32)
            mask = np.zeros((H, W), np.uint8)
            cv2.fillPoly(mask, [arr], 255)
            suffix = "_D" if self.dead_flags.get(i, False) else ""
            lab = self.image_label.tracking_labels[i] if i < len(self.image_label.tracking_labels) else None
            fname = f"{lab[0]}_{lab}{suffix}.npy" if lab else f"mask{i+1}{suffix}.npy"
            np.save(os.path.join(save_dir, fname), mask)

    # Updated `save_contours` method for `ContourZoomWidget`
    def save_contours(self):
        """
        Save each contour's raw coordinates to its own .npy file,
        using the same <Class>_<ID>[_D].npy naming convention.
        """
        from PyQt5.QtWidgets import QMessageBox

        # Parse block, nanowell, frame
        block, nanowell, frame = self.label_text.split("-")
        frame = frame.lstrip("t")

        # Compute widget-to-image transform
        H, W = self.ch0_image.shape
        pix = self.image_label.pixmap()
        pw, ph = pix.width(), pix.height()
        lw, lh = self.image_label.width(), self.image_label.height()
        off_x = (lw - pw) / 2.0
        off_y = (lh - ph) / 2.0
        sx, sy = W / pw, H / ph

        # Prepare save directory (create intermediate folders if needed)
        save_dir = os.path.join(
            self.base_path, block, "Masks", "FreeHand",
            f"imgNo{nanowell}CH0", f"t{frame}"
        )
        try:
            # Remove any existing contents
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            # Create the full directory tree
            os.makedirs(save_dir, exist_ok=True)
        except PermissionError as e:
            print(f"[save_contours] Permission denied for {save_dir}: {e}")
            QMessageBox.warning(
                self,
                "Save Failed",
                f"Could not create directory:\n{save_dir}\n\nPermission denied."
            )
            return

        # Iterate over contours and save raw coordinate lists
        for i, contour in enumerate(self.image_label.contours):
            # Map widget points back to image coordinates
            pts = []
            for pt in contour:
                x_img = (pt.x() - off_x) * sx
                y_img = (pt.y() - off_y) * sy
                pts.append((x_img, y_img))

            # Build filename using tracking label and dead-flag
            lbl = None
            if i < len(self.image_label.tracking_labels):
                lbl = self.image_label.tracking_labels[i]  # e.g. "A12"
            suffix = "_D" if self.dead_flags.get(i, False) else ""
            if lbl:
                cls, idx = lbl[0], lbl[1:]
                fname = f"{cls}_{idx}{suffix}.npy"
            else:
                fname = f"cell{i+1}{suffix}.npy"

            # Save the raw points array
            raw_path = os.path.join(save_dir, fname)
            np.save(raw_path, np.array(pts, dtype=object), allow_pickle=True)

        # Mark this frame as saved
        key = f"{block}-{nanowell}-t{frame}"
        self.frame_save_status[key] = True

        # Clear selection and refresh display
        self.image_label.selected_points.clear()
        self.image_label.update()


    def switch_mask_type(self, mask_type: str):
        """
        Load contours for the given mask type by inspecting the contents of each .npy file:
        - If shape is (N,2) treat as raw coordinate arrays.
        - If shape is (H, W) or (1, H, W) treat as binary mask(s).
        """
        # clear any highlighting/selections
        self.image_label.selected_contour_index = -1
        self.image_label.clearMultiSelection()
        self.image_label.selected_points.clear()

        block, nanowell, frame = self.label_text.split("-")
        frame_num = frame.lstrip("t")
        self.mask_type = mask_type
        self.mask_type_label.setText(f"Current Mask Type: {mask_type}")
        self.apply_fluorescence_overlay()

        # clear any existing contours
        self.image_label.raw_contours = []
        self.image_label.contours     = []
        # reset the smoothing baseline to whatever we just loaded
        self.image_label.angular_baseline = [list(c) for c in self.image_label.contours]

        # compute widget?image transform
        H, W = self.ch0_image.shape
        pix = self.image_label.pixmap()
        pw, ph = pix.width(), pix.height()
        lw, lh = self.image_label.width(), self.image_label.height()
        off_x = (lw - pw) / 2.0
        off_y = (lh - ph) / 2.0
        scale_x = pw / W
        scale_y = ph / H
        
        base_dir = os.path.join(
            self.base_path, block, "Masks", mask_type,
            f"imgNo{nanowell}CH0", f"t{frame_num}"
        )

        print(f"[{mask_type}] Looking in {base_dir}")
        if not os.path.isdir(base_dir):
            print(f"[{mask_type}] Folder missing, clearing contours")
            self.compute_tracking_labels()
            self.apply_angular_smoothing_to_all()
            self.image_label.update()
            return

        # look for a single raw_contours.npy first
        list_file = os.path.join(base_dir, "raw_contours.npy")
        if os.path.isfile(list_file):
            pts_list = np.load(list_file, allow_pickle=True)
            loaded = []
            for pts in pts_list:
                qpts = [
                    QPoint(
                        int(round(x * scale_x + off_x)),
                        int(round(y * scale_y + off_y))
                    )
                    for x, y in pts
                ]
                loaded.append(qpts)
            self.image_label.raw_contours = loaded.copy()
            self.image_label.contours     = loaded.copy()
            self.compute_tracking_labels()
            self.apply_angular_smoothing_to_all()
            self.image_label.update()
            print(f"[{mask_type}] Loaded raw_contours.npy with {len(loaded)} contours")
            return

        # otherwise inspect every .npy
        for fname in sorted(os.listdir(base_dir)):
            if not fname.endswith(".npy"):
                continue
            path = os.path.join(base_dir, fname)
            try:
                arr = np.load(path, allow_pickle=True)
            except Exception as e:
                print(f"[{mask_type}] skip {fname}: cannot load ({e})")
                continue

            # 1) raw coordinate arrays
            if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 2:
                qpts = [
                    QPoint(
                        int(round(off_x + x * scale_x)),
                        int(round(off_y + y * scale_y))
                    )
                    for x, y in arr
                ]
                self.image_label.raw_contours.append(qpts)
                continue

            # 2) binary mask arrays, including SAM??s (1, H, W)
            mask_arr = None
            if isinstance(arr, np.ndarray):
                # if it??s (1, H, W), squeeze down to (H, W)
                if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[1] == H and arr.shape[2] == W:
                    mask_arr = arr.squeeze(0)
                # if it??s already (H, W)
                elif arr.ndim == 2 and arr.shape == (H, W):
                    mask_arr = arr

            if mask_arr is not None:
                mask = (mask_arr > 0).astype(np.uint8)
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in cnts:
                    if cv2.contourArea(cnt) > 5000:
                        continue
                    pts = [
                        QPoint(
                            int(round(off_x + pt[0][0] * scale_x)),
                            int(round(off_y + pt[0][1] * scale_y))
                        )
                        for pt in cnt
                    ]
                    interp = self.image_label.interpolate_contour(pts, desired_gap=2)
                    smooth = self.image_label.smooth_drawn_contour(
                        interp,
                        s=20,
                        num_points=int(len(interp) * 10 * self.image_label.density_factor)
                    )
                    self.image_label.raw_contours.append(smooth)
                continue

            # 3) anything else, skip
            print(f"[{mask_type}] skip {fname}: unrecognized array shape {getattr(arr, 'shape', None)}")

        # commit what we found
        if self.image_label.raw_contours:
            self.image_label.contours = [list(c) for c in self.image_label.raw_contours]
            self.compute_tracking_labels()
            self.apply_angular_smoothing_to_all()
            self.image_label.update()
            print(f"[{mask_type}] Loaded {len(self.image_label.contours)} contours")
        else:
            print(f"[{mask_type}] No contours found, showing only tracking labels")
            self.compute_tracking_labels()
            self.apply_angular_smoothing_to_all()
            self.image_label.update()

        self.apply_angular_smoothing_to_all()
        # after contours have been loaded & painted??
        self.image_label.update()
        # ?? clear any leftover nudge-selection
        self.image_label.selected_points.clear()



    def load_previous_freehand_masks(self):
        """
        Load the FreeHand .npy masks from the prior frame
        and hand them through switch_mask_type??so they get interpolated & smoothed.
        """
        block, nanowell, frame = self.label_text.split("-")
        prev = int(frame.lstrip("t")) - 1
        if prev < 1:
            return

        mask_paths = self.get_mask_paths(block, nanowell, str(prev), "FreeHand")
        if not mask_paths:
            return

        # stash them temporarily and switch view
        masks = []
        for p in mask_paths:
            try:
                masks.append(np.load(p).astype(np.uint8))
            except:
                pass

        # use the same pipeline
        self.masks = getattr(self, "masks", {})
        self.masks["FreeHand"] = masks
        self.switch_mask_type("FreeHand")

    def update_default_mask_type(self, text):
        # Update the widget's default mask type with the chosen value.
        self.mask_type = text
        self.mask_type_label.setText(f"Current Mask Type: {self.mask_type}")
        # Optionally, update the displayed mask immediately
        self.switch_mask_type(text)

    def go_to_next_frame(self):
        print("Navigating to next frame...")
        try:
            elapsed = time.time() - self.start_time
            if self.parent() is not None and hasattr(self.parent(), 'log_zoom_time'):
                self.parent().log_zoom_time(self.label_text, elapsed)
            block, nanowell, frame = self.label_text.split("-")
            frame_num = int(frame.lstrip("t"))
            if frame_num >= self.max_frame:
                print("Already at the last frame.")
                return

            new_frame_num = frame_num + 1
            self.label_text = f"{block}-{nanowell}-t{new_frame_num}"
            self.setWindowTitle(self.label_text)
            ch0_path = self.get_image_path(block, nanowell, f"{new_frame_num}", 'CH0')
            self.ch0_image = cv2.imread(ch0_path, cv2.IMREAD_GRAYSCALE)
            if self.ch0_image is None:
                print(f"CH0 image not found at {ch0_path}")
                return
            self.ch0_pixmap = QPixmap.fromImage(numpy_to_qimage(self.ch0_image))
            self.image_label.update_pixmap(self.ch0_pixmap)
            self.image_label.pixmap_original = self.ch0_pixmap
            self.image_label.clear_contours()
            self.apply_fluorescence_overlay()
            self.switch_mask_type(self.mask_type)
            self.image_label.current_mask_type = self.mask_type

            # --- Reset resolution slider and baseline contours ---
            if hasattr(self, 'original_contours'):
                del self.original_contours
            self.resize_slider.setValue(100)

            self.imageForSam.emit(self.fluorescence_overlay)
            self.start_time = time.time()
        except Exception as e:
            print(f"Error navigating to next frame: {e}")

    def go_to_previous_frame(self):
        print("Navigating to previous frame...")
        try:
            elapsed = time.time() - self.start_time
            if self.parent() is not None and hasattr(self.parent(), 'log_zoom_time'):
                self.parent().log_zoom_time(self.label_text, elapsed)
            block, nanowell, frame = self.label_text.split("-")
            frame_num = int(frame.lstrip("t"))
            if frame_num > 1:
                new_frame_num = frame_num - 1
                self.label_text = f"{block}-{nanowell}-t{new_frame_num}"
                self.setWindowTitle(self.label_text)
                ch0_path = self.get_image_path(block, nanowell, f"{new_frame_num}", 'CH0')
                self.ch0_image = cv2.imread(ch0_path, cv2.IMREAD_GRAYSCALE)
                if self.ch0_image is None:
                    print(f"CH0 image not found at {ch0_path}")
                    return
                self.ch0_pixmap = QPixmap.fromImage(numpy_to_qimage(self.ch0_image))
                self.image_label.update_pixmap(self.ch0_pixmap)
                self.image_label.pixmap_original = self.ch0_pixmap
                self.apply_fluorescence_overlay()
                self.switch_mask_type(self.mask_type)

                # --- Reset resolution slider and baseline contours ---
                if hasattr(self, 'original_contours'):
                    del self.original_contours
                self.resize_slider.setValue(100)

                self.start_time = time.time()
                self.imageForSam.emit(self.fluorescence_overlay)
            else:
                print("Already at the first frame.")
        except Exception as e:
            print(f"Error navigating to previous frame: {e}")

    def curate_masks_to_gt(self):
        try:
            block, nanowell, frame = self.label_text.split("-")
            frame_numeric = frame.lstrip("t")
            if not self.mask_type:
                print("No current mask type set for curation.")
                return
            source_paths = self.get_mask_paths(block, nanowell, frame_numeric, self.mask_type)
            if not source_paths:
                print(f"No masks found for current mask type: {self.mask_type}")
                return
            dest_folder = os.path.join(
                self.base_path,
                block,
                "Masks",
                "GT",
                f"imgNo{nanowell}CH0",
                f"t{frame_numeric}"
            )
            if os.path.exists(dest_folder):
                shutil.rmtree(dest_folder)
            os.makedirs(dest_folder, exist_ok=True)
            tracking_labels = self.image_label.tracking_labels if hasattr(self.image_label, "tracking_labels") else []
            for i, path in enumerate(source_paths):
                if i < len(tracking_labels) and tracking_labels[i]:
                    label = tracking_labels[i]
                    cell_type = label[0]
                    file_name = f"{cell_type}_{label}.npy"
                else:
                    file_name = f"mask{i + 1}.npy"
                dest_path = os.path.join(dest_folder, file_name)
                shutil.move(path, dest_path)
                print(f"Moved {path} to {dest_path}")
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Success")
            msg_box.setText("Masks curated to GT successfully!")
            msg_box.setStyleSheet("background-color: green; color: white;")
            msg_box.setStandardButtons(QMessageBox.NoButton)
            msg_box.show()
            QTimer.singleShot(2000, lambda: msg_box.done(0))
        except Exception as e:
            print(f"Error during mask curation: {e}")

 
    def compute_tracking_labels(self):
        """
        Loads labels from disk for the current frame, then overrides them with any
        in-memory updates from self.tracking_mapping (or self.tracking_mapping_by_id)
        only if the current frame is >= self.override_start_frame.
        """
        self.image_label.tracking_labels = []
        block, nanowell, frame = self.label_text.split("-")
        frame_numeric = frame.lstrip("t")
        current_frame = int(frame_numeric)
        folder = os.path.join(
            self.base_path,
            block,
            "Masks",
            self.mask_type,
            f"imgNo{nanowell}CH0",
            f"t{frame_numeric}"
        )
        if not os.path.exists(folder):
            print(f"Mask folder not found: {folder}")
            return

        npy_files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])
        labels = []
        for fname in npy_files:
            m = re.match(r"([A-Za-z]+)_(\d+)\.npy", fname)
            if m:
                cell_class, tracking_id = m.groups()
                label = f"{cell_class}{tracking_id}"
                labels.append(label)
            else:
                labels.append("")

        # Apply in-memory overrides only if the current frame is at or beyond the override start.
        if hasattr(self, "override_start_frame") and self.override_start_frame is not None:
            if current_frame >= self.override_start_frame:
                if hasattr(self, "tracking_mapping") and self.tracking_mapping:
                    for idx, new_label in self.tracking_mapping.items():
                        if idx < len(labels):
                            labels[idx] = new_label
                if hasattr(self, "tracking_mapping_by_id") and self.tracking_mapping_by_id:
                    for i, lab in enumerate(labels):
                        m = re.match(r"([A-Za-z]+)(\d+)", lab)
                        if m:
                            tid = int(m.group(2))
                            if tid in self.tracking_mapping_by_id:
                                labels[i] = self.tracking_mapping_by_id[tid]
        else:
            # No override has been set; ensure override_start_frame remains None.
            self.override_start_frame = None

        self.image_label.tracking_labels = labels
        print(f"Tracking labels for frame {current_frame}: {self.image_label.tracking_labels}")
    def apply_angular_smoothing_to_all(self):
        #self.image_label.apply_angular_smoothing()
        #print("Angular smoothing applied to all contours.")
        self.image_label.apply_coordinate_gaussian_smoothing(self.angular_sigma)
        print("Coordinate Gaussian smoothing applied to all contours.")

    def propagate_bulk_cell_update(self):
        """
        For a single selected cell, update the entire label (both cell type and tracking number)
        as provided by the user. The override start frame is recorded so that the change applies from
        the current frame onward.
        """
        idx = self.image_label.selected_contour_index
        if idx == -1:
            QMessageBox.warning(self, "Bulk Update", "No contour is selected.")
            return

        current_label = self.image_label.tracking_labels[idx]
        new_label, ok = QInputDialog.getText(
            self,
            "Bulk Update Cell Label",
            f"Enter new full label for cell (currently {current_label}):",
            text=current_label
        )
        if not ok or not new_label:
            return

        if not hasattr(self, "tracking_mapping"):
            self.tracking_mapping = {}
        self.tracking_mapping[idx] = new_label

        current_frame = int(self.label_text.split("-")[-1].lstrip("t"))
        if not hasattr(self, "override_start_frame") or self.override_start_frame is None:
            self.override_start_frame = current_frame

        QMessageBox.information(self, "Bulk Update",
                                f"Bulk update complete for contour {idx}:\n{current_label} -> {new_label}\n"
                                "This update will be applied to all consecutive frames upon confirmation.")
        print(f"Bulk updated in-memory mapping for contour {idx}: {new_label}")
        self.compute_tracking_labels()
        self.image_label.update()

    def propagate_tracking_swap(self):
        """
        Swaps numeric IDs of selected contours on disk, persisting from current frame onward.
        """
        from PyQt5.QtWidgets import QApplication, QMessageBox
        import os, re

        # Gather selected indices and current labels
        selected = self.image_label.multi_selected_indices
        if len(selected) < 2:
            QMessageBox.warning(self, "Swap Error", "Select at least two contours for swapping.")
            return

        current_labels = [self.image_label.tracking_labels[i] for i in selected]
        letters = [lbl[0] for lbl in current_labels]
        nums = [lbl[1:] for lbl in current_labels]
        # Circular swap
        new_nums = [nums[-1]] + nums[:-1]
        new_labels = [ltr + num for ltr, num in zip(letters, new_nums)]

        # Map track ID (int) to new full label
        mapping_by_id = {}
        for idx, new_lbl in zip(selected, new_labels):
            match = re.match(r"[A-Za-z](\d+)", current_labels[selected.index(idx)])
            if match:
                tid = int(match.group(1))
                mapping_by_id[tid] = new_lbl

        # Build renames across frames
        block, nanowell, _ = self.label_text.split("-")
        start_frame = int(self.label_text.split("-")[-1].lstrip("t"))
        total = 0
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for fnum in range(start_frame, self.max_frame + 1):
                folder = os.path.join(
                    self.base_path, block, "Masks", "FreeHand",
                    f"imgNo{nanowell}CH0", f"t{fnum}"
                )
                if not os.path.isdir(folder):
                    continue
                renames = []
                for fname in sorted(os.listdir(folder)):
                    m = re.match(r"([A-Za-z])_(\d+)(_D)?\.npy", fname)
                    if not m:
                        continue
                    orig_type, id_str, dead = m.group(1), m.group(2), m.group(3) or ""
                    cid = int(id_str)
                    orig_lbl = f"{orig_type}{id_str}"
                    new_full = mapping_by_id.get(cid)
                    if new_full and new_full != orig_lbl:
                        new_fname = f"{new_full[0]}_{new_full[1:]}{dead}.npy"
                        renames.append((fname, new_fname))
                if renames:
                    safe_rename_in_folder(folder, renames)
                    total += len(renames)
            QMessageBox.information(
                self, "Swap Complete",
                f"{total} swap renames applied to disk."
            )
        finally:
            QApplication.restoreOverrideCursor()

        self.image_label.clearMultiSelection()
        # 2) reload & redraw the masks/contours you??re looking at
        self.switch_mask_type(self.mask_type)

    def propagate_celltype_fix(self):
        """
        For a single selected cell, updates its cell type letter immediately on disk
        (preserving its tracking number) from the current frame onward.
        """
        idx = self.image_label.selected_contour_index
        labels = self.image_label.tracking_labels
        if idx < 0 or idx >= len(labels):
            QMessageBox.warning(self, "Fix Cell Type", "No contour is selected.")
            return

        # Ask user for new letter
        current = labels[idx]
        new_class, ok = QInputDialog.getText(
            self, "Fix Cell Type",
            f"Enter new cell type for {current}:",
            text=current[0]
        )
        if not ok or not new_class: 
            return
        new_label = new_class[0] + current[1:]

        # Build map from old track-ID to new full label
        import re, os
        mapping_by_id = {}
        m = re.match(r"[A-Za-z](\d+)", current)
        if m:
            orig_id = int(m.group(1))
            mapping_by_id[orig_id] = new_label

        # Now rename across frames immediately
        block, nanowell, _ = self.label_text.split("-")
        start_frame = int(self.label_text.split("-")[-1].lstrip("t"))
        total = 0
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for fnum in range(start_frame, self.max_frame + 1):
                folder = os.path.join(
                    self.base_path, block, "Masks", "FreeHand",
                    f"imgNo{nanowell}CH0", f"t{fnum}"
                )
                if not os.path.isdir(folder):
                    continue

                renames = []
                for fname in sorted(os.listdir(folder)):
                    mm = re.match(r"([A-Za-z])_(\d+)(_D)?\.npy", fname)
                    if not mm:
                        continue
                    _, track_str, dead = mm.groups()
                    tid = int(track_str)
                    orig_label = mm.group(1) + track_str
                    new_full = mapping_by_id.get(tid)
                    if new_full and new_full != orig_label:
                        new_fname = f"{new_full[0]}_{new_full[1:]}{dead or ''}.npy"
                        renames.append((fname, new_fname))
                if renames:
                    safe_rename_in_folder(folder, renames)
                    total += len(renames)
        finally:
            QApplication.restoreOverrideCursor()

        QMessageBox.information(
            self, "Class Update",
            f"Cell type changed on disk for {total} files:\n"
            f"{current} ? {new_label}"
        )

        # Refresh in-memory labels right away
        self.compute_tracking_labels()
        self.apply_angular_smoothing_to_all()
        self.image_label.clearMultiSelection()
        # 2) reload & redraw the masks/contours you??re looking at
        self.switch_mask_type(self.mask_type)
        self.image_label.update()

    """
    def confirm_tracks_classes(self):
        from PyQt5.QtWidgets import QApplication, QMessageBox
        import os
        import re

        # Build id-to-new-label mapping from index-based overrides
        mapping_by_id = {}
        raw_mapping = getattr(self, 'tracking_mapping', {})
        labels = self.image_label.tracking_labels
        for idx, new_label in raw_mapping.items():
            if idx < len(labels):
                m = re.match(r"([A-Za-z])(\d+)", labels[idx])
                if m:
                    cell_id = int(m.group(2))
                    mapping_by_id[cell_id] = new_label

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            block, nanowell, _ = self.label_text.split("-")
            start_frame = int(self.label_text.split("-")[-1].lstrip("t"))
            total_renamed = 0

            for fnum in range(start_frame, self.max_frame + 1):
                folder = os.path.join(
                    self.base_path,
                    block,
                    "Masks",
                    "FreeHand",
                    f"imgNo{nanowell}CH0",
                    f"t{fnum}"
                )
                if not os.path.isdir(folder):
                    continue

                renames = []
                for fname in sorted(os.listdir(folder)):
                    m = re.match(r"([A-Za-z])_(\d+)(_D)?\.npy", fname)
                    if not m:
                        continue
                    orig_type, track_str, dead = m.group(1), m.group(2), m.group(3) or ""
                    track_id = int(track_str)
                    orig_label = f"{orig_type}{track_str}"
                    new_label = mapping_by_id.get(track_id, orig_label)
                    if new_label != orig_label:
                        new_fname = f"{new_label[0]}_{new_label[1:]}{dead}.npy"
                        renames.append((fname, new_fname))

                if renames:
                    safe_rename_in_folder(folder, renames)
                    total_renamed += len(renames)

            QMessageBox.information(
                self,
                "Labels Updated",
                f"{total_renamed} files renamed"
            )
            self.override_start_frame = 0
            self.compute_tracking_labels()
            self.image_label.update()
        finally:
            QApplication.restoreOverrideCursor()
    """
    def load_contours_for_frame(self, block, nanowell, fnum):
        """
        Loads the original contours and tracking labels for a given frame by reading saved .npy mask files.
        Returns a tuple: (list_of_contours, list_of_tracking_labels).
        """
        contours = []
        tracking_labels = []
        folder = os.path.join(
            self.base_path,
            block,
            "Masks",
            self.mask_type,
            f"imgNo{nanowell}CH0",
            f"t{fnum}"
        )
        if not os.path.exists(folder):
            print(f"Mask folder not found for frame t{fnum}: {folder}")
            return contours, tracking_labels
        
        mask_files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])
        for file_name in mask_files:
            file_path = os.path.join(folder, file_name)
            try:
                mask = np.load(file_path)
                mask = np.squeeze(mask)
                mask = (mask > 0).astype(np.uint8)
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in cnts:
                    if cv2.contourArea(cnt) > 5000:
                        continue
                    qpoints = []
                    for pt in cnt:
                        x = int(pt[0][0] * self.image_label.width() / mask.shape[1])
                        y = int(pt[0][1] * self.image_label.height() / mask.shape[0])
                        qpoints.append(QPoint(x, y))
                    if qpoints:
                        contours.append(qpoints)
                        import re
                        m = re.match(r"([A-Za-z])_(\d+)", file_name)
                        if m:
                            tracking_labels.append(m.group(1) + m.group(2))
                        else:
                            tracking_labels.append("")
            except Exception as e:
                print(f"Failed to load or process mask {file_path}: {e}")
        return contours, tracking_labels

    def run_sam_video_prediction(self):
        """
        Run SAM prediction for the next 10 frames using drawn bounding boxes.
        Enhanced version with robust tracking for sudden movements.
        """
        if not self.samLoaded:
            QMessageBox.warning(self, "SAM", "Still loading SAM model ?? please wait.")
            return

        boxes = self.scaled_boxes()
        if not boxes:
            QMessageBox.warning(self, "SAM", "Draw at least one box first.")
            return

        # Get current frame info
        block, nanowell, frame = self.label_text.split("-")
        current_frame = int(frame.lstrip("t"))
        
        # Calculate frames to process (use spinbox value, but don't exceed max_frame)
        requested_frames = self.frame_count_spinbox.value()
        frames_to_process = min(requested_frames, self.max_frame - current_frame)
        if frames_to_process <= 0:
            QMessageBox.information(self, "SAM Video", "No more frames to process.")
            return

        print(f"Starting video prediction for frames {current_frame + 1} to {current_frame + frames_to_process}")
        print(f"Processing {len(boxes)} bounding boxes")
        print(f"Box coordinates: {boxes}")

        # Load custom model if requested
        if self.use_custom_model_checkbox.isChecked():
            print("Loading custom SAM model...")
            if not self._samWorker.loadCustomCheckpoint(custom_sam_checkpoint):
                QMessageBox.critical(self, "SAM Video Error", "Failed to load custom checkpoint")
                return
        else:
            print("Using default SAM model")

        # Initialize tracking history for each box
        tracking_histories = [[] for _ in range(len(boxes))]

        # Show progress dialog
        progress = QProgressDialog("Processing video frames...", "Cancel", 0, frames_to_process, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(True)
        progress.show()

        try:
            # Process each frame
            for frame_offset in range(1, frames_to_process + 1):
                if progress.wasCanceled():
                    break

                target_frame = current_frame + frame_offset
                progress.setValue(frame_offset - 1)
                progress.setLabelText(f"Processing frame {target_frame}...")
                
                print(f"Processing frame {target_frame}")

                # Load the target frame image
                ch0_path = self.get_image_path(block, nanowell, str(target_frame), 'CH0')
                frame_image = cv2.imread(ch0_path, cv2.IMREAD_GRAYSCALE)
                if frame_image is None:
                    print(f"Frame {target_frame} image not found at {ch0_path}")
                    continue

                # Create fluorescence overlay for this frame
                frame_overlay = self.create_fluorescence_overlay_for_frame(block, nanowell, str(target_frame))
                if frame_overlay is None:
                    print(f"Could not create fluorescence overlay for frame {target_frame}")
                    continue

                # Set the image in the predictor
                try:
                    self._samWorker.predictor.set_image(frame_overlay)
                    print(f"Set image for frame {target_frame}, shape: {frame_overlay.shape}")
                except Exception as e:
                    print(f"Error setting image for frame {target_frame}: {e}")
                    continue

                # Process each bounding box
                frame_masks = []
                for box_idx, box in enumerate(boxes):
                    print(f"Processing box {box_idx + 1}/{len(boxes)}: {box}")
                    
                    # Ensure box coordinates are valid
                    x1, y1, x2, y2 = box
                    if x1 >= x2 or y1 >= y2:
                        print(f"Invalid box {box_idx}: {box}")
                        continue
                    
                    # Convert box to SAM format [x1, y1, x2, y2]
                    sam_box = np.array(box, dtype=np.int32)
                    
                    try:
                        # Get mask prediction
                        masks, scores, logits = self._samWorker.predictor.predict(
                            box=sam_box,
                            multimask_output=False  # Get single best mask
                        )
                        
                        # Get the best mask
                        best_mask = masks[0]
                        best_score = scores[0]
                        print(f"Box {box_idx + 1} prediction complete, mask shape: {best_mask.shape}, score: {best_score:.3f}")
                        
                        # Check if mask has significant content
                        mask_sum = np.sum(best_mask > 0.05)
                        if mask_sum < 50:  # Very small mask
                            print(f"Box {box_idx + 1} produced very small mask ({mask_sum} pixels), skipping")
                            continue
                            
                        frame_masks.append(best_mask)
                        
                    except Exception as e:
                        print(f"Error predicting mask for box {box_idx + 1}: {e}")
                        continue

                print(f"Saving {len(frame_masks)} masks for frame {target_frame}")

                # Save masks for this frame
                sam_dir = os.path.join(
                    self.base_path, block, "Masks", "SAM",
                    f"imgNo{nanowell}CH0", f"t{target_frame}"
                )
                os.makedirs(sam_dir, exist_ok=True)
                
                for mask_idx, mask in enumerate(frame_masks):
                    mask_file = os.path.join(sam_dir, f"{block}-{nanowell}-t{target_frame}_mask{mask_idx}.npy")
                    np.save(mask_file, mask)
                    print(f"Saved SAM mask {mask_idx} to {mask_file}")
                    
                    # Also extract contours and save as FreeHand masks
                    if self.save_as_freehand_checkbox.isChecked():
                        contours = self.extract_contours_from_mask(mask)
                        print(f"Extracted {len(contours)} contours from mask {mask_idx}")
                        if contours:
                            # Apply Gaussian smoothing to the contours
                            smoothing_sigma = self.gaussian_sigma_spinbox.value()
                            smoothed_contours = self.apply_gaussian_smoothing_to_contours(contours, sigma=smoothing_sigma)
                            print(f"Applied Gaussian smoothing (σ={smoothing_sigma}) to {len(smoothed_contours)} contours")
                            self.save_contours_as_freehand(smoothed_contours, block, nanowell, str(target_frame), mask_idx)

                # Update bounding boxes for next frame using robust tracking
                if len(frame_masks) > 0:
                    updated_boxes = []
                    for box_idx, (box, mask) in enumerate(zip(boxes, frame_masks)):
                        # Use robust tracking with motion prediction
                        updated_box = self.update_bbox_from_mask_robust(
                            mask, box, target_frame, tracking_histories[box_idx]
                        )
                        updated_boxes.append(updated_box)
                    
                    # Update boxes for next iteration
                    boxes = updated_boxes
                    print(f"Updated boxes for next frame: {boxes}")

            progress.setValue(frames_to_process)
            
            if not progress.wasCanceled():
                QMessageBox.information(self, "SAM Video", 
                    f"Successfully processed {frames_to_process} frames!\n"
                    f"Masks saved to SAM directory for frames {current_frame + 1} to {current_frame + frames_to_process}")
            else:
                QMessageBox.information(self, "SAM Video", "Video processing was canceled.")

        except Exception as e:
            QMessageBox.critical(self, "SAM Video Error", f"Error during video processing: {str(e)}")
            print(f"Error in run_sam_video_prediction: {e}")
            import traceback
            traceback.print_exc()
        finally:
            progress.close()

        # Clear drawn boxes after processing
        self.image_label.sam_boxes.clear()
        if self.box_button.isChecked():
            self.box_button.setChecked(False)
        self.image_label.update()

        # Switch to FreeHand mask type and reload all contours
        self.switch_mask_type("FreeHand")
        self.mask_type = "FreeHand"
        self.image_label.current_mask_type = "FreeHand"
        print("Switched to FreeHand mask type to show all contours.")

    def create_fluorescence_overlay_for_frame(self, block, nanowell, frame):
        """
        Create fluorescence overlay for a specific frame.
        Returns RGB image suitable for SAM input.
        """
        try:
            # Load CH0 image
            ch0_path = self.get_image_path(block, nanowell, frame, 'CH0')
            ch0_image = cv2.imread(ch0_path, cv2.IMREAD_GRAYSCALE)
            if ch0_image is None:
                return None

            # Load fluorescence channels if enabled
            ch1_path = ch0_path.replace('CH0', 'CH1')
            ch2_path = ch0_path.replace('CH0', 'CH2')
            ch3_path = ch0_path.replace('CH0', 'CH3')

            ch1_image = cv2.imread(ch1_path, cv2.IMREAD_GRAYSCALE) if self.overlay_channels.get("CH1", True) else None
            ch2_image = cv2.imread(ch2_path, cv2.IMREAD_GRAYSCALE) if self.overlay_channels.get("CH2", True) else None
            ch3_image = cv2.imread(ch3_path, cv2.IMREAD_GRAYSCALE) if self.overlay_channels.get("CH3", True) else None

            # Handle missing channels
            ch1_image = ch1_image if ch1_image is not None else np.zeros_like(ch0_image)
            ch2_image = ch2_image if ch2_image is not None else np.zeros_like(ch0_image)
            ch3_image = ch3_image if ch3_image is not None else np.zeros_like(ch0_image)

            # Normalize and blend
            ch0_norm = ch0_image / 255.0
            ch1_norm = ch1_image / 255.0
            ch2_norm = ch2_image / 255.0
            ch3_norm = ch3_image / 255.0

            blended = np.stack([ch0_norm, ch0_norm, ch0_norm], axis=-1)
            blended[..., 1] += 0.5 * ch1_norm  # Green channel
            blended[..., 0] += 0.5 * ch2_norm  # Red channel
            blended[..., 2] += 0.8 * ch3_norm  # Blue channel
            blended = np.clip(blended, 0, 1)
            blended_overlay = (blended * 255).astype(np.uint8)

            return blended_overlay

        except Exception as e:
            print(f"Error creating fluorescence overlay for frame {frame}: {e}")
            return None

    def update_bbox_from_mask(self, mask, original_bbox, padding=(10, 10)):
        """
        Update bounding box position based on predicted mask centroid and extent.
        Enhanced version with better handling of sudden movements.
        """
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
        H, W = mask.shape
        new_x2 = min(new_x2, W)
        new_y2 = min(new_y2, H)
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        
        return np.array([new_x1, new_y1, new_x2, new_y2], dtype=np.float32)

    def update_bbox_from_mask_robust(self, mask, original_bbox, frame_idx, tracking_history=None):
        """
        Enhanced bounding box update with motion prediction and recovery for sudden movements.
        
        Args:
            mask: SAM predicted mask
            original_bbox: Current bounding box [x1, y1, x2, y2]
            frame_idx: Current frame index for velocity calculation
            tracking_history: List of previous bbox positions for motion prediction
        """
        # Initialize tracking history if not provided
        if tracking_history is None:
            tracking_history = []
        
        # Find mask centroid and extent
        mask_binary = (mask > 0.05).astype(np.uint8)
        
        if np.sum(mask_binary) == 0:
            # No detection - try motion prediction
            if len(tracking_history) >= 2:
                predicted_bbox = self.predict_bbox_from_motion(tracking_history, frame_idx)
                if predicted_bbox is not None:
                    print(f"Using motion prediction for frame {frame_idx}")
                    return predicted_bbox
            # Fallback to original bbox
            return original_bbox
        
        # Find contours to get the main object
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return original_bbox
        
        # Get largest contour (main object)
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) < 20:
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
        
        # Calculate movement from previous position
        prev_center_x = (orig_x1 + orig_x2) / 2
        prev_center_y = (orig_y1 + orig_y2) / 2
        movement_x = new_center_x - prev_center_x
        movement_y = new_center_y - prev_center_y
        movement_magnitude = np.sqrt(movement_x**2 + movement_y**2)
        
        # Adaptive size based on movement magnitude
        if movement_magnitude > 50:  # Large movement detected
            # Use more of detected size for large movements
            adaptive_w = int(0.3 * orig_w + 0.7 * w)
            adaptive_h = int(0.3 * orig_h + 0.7 * h)
            # Increase padding for large movements
            padding = 20
        else:
            # Normal adaptive sizing
            adaptive_w = int(0.5 * orig_w + 0.5 * w)
            adaptive_h = int(0.5 * orig_h + 0.5 * h)
            padding = 10
        
        # Add padding and cap size (increased for sudden movements)
        max_size = 150 if movement_magnitude > 50 else 100
        adaptive_w = min(adaptive_w + padding, max_size)
        adaptive_h = min(adaptive_h + padding, max_size)
        
        # Create new bbox centered on detection
        new_x1 = max(0, new_center_x - adaptive_w // 2)
        new_y1 = max(0, new_center_y - adaptive_h // 2)
        new_x2 = new_x1 + adaptive_w
        new_y2 = new_y1 + adaptive_h
        
        # Ensure bbox doesn't go outside image boundaries
        H, W = mask.shape
        new_x2 = min(new_x2, W)
        new_y2 = min(new_y2, H)
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        
        new_bbox = np.array([new_x1, new_y1, new_x2, new_y2], dtype=np.float32)
        
        # Update tracking history
        tracking_history.append({
            'frame': frame_idx,
            'bbox': new_bbox.copy(),
            'center': (new_center_x, new_center_y),
            'movement': (movement_x, movement_y)
        })
        
        # Keep only last 5 frames for velocity calculation
        if len(tracking_history) > 5:
            tracking_history.pop(0)
        
        return new_bbox

    def predict_bbox_from_motion(self, tracking_history, current_frame):
        """
        Predict bbox position based on motion history when detection fails.
        """
        if len(tracking_history) < 2:
            return None
        
        # Calculate average velocity from last few frames
        velocities = []
        for i in range(1, len(tracking_history)):
            prev = tracking_history[i-1]
            curr = tracking_history[i]
            vx = curr['movement'][0]
            vy = curr['movement'][1]
            velocities.append((vx, vy))
        
        if not velocities:
            return None
        
        # Average velocity
        avg_vx = np.mean([v[0] for v in velocities])
        avg_vy = np.mean([v[1] for v in velocities])
        
        # Get last known position
        last_bbox = tracking_history[-1]['bbox']
        last_center = tracking_history[-1]['center']
        
        # Predict new center
        frames_since_last = current_frame - tracking_history[-1]['frame']
        predicted_center_x = last_center[0] + avg_vx * frames_since_last
        predicted_center_y = last_center[1] + avg_vy * frames_since_last
        
        # Use last known bbox size
        bbox_w = last_bbox[2] - last_bbox[0]
        bbox_h = last_bbox[3] - last_bbox[1]
        
        # Create predicted bbox
        pred_x1 = max(0, predicted_center_x - bbox_w // 2)
        pred_y1 = max(0, predicted_center_y - bbox_h // 2)
        pred_x2 = pred_x1 + bbox_w
        pred_y2 = pred_y1 + bbox_h
        
        return np.array([pred_x1, pred_y1, pred_x2, pred_y2], dtype=np.float32)

    def expand_search_area_for_sudden_movement(self, original_bbox, movement_threshold=50):
        """
        Expand the search area when sudden movement is detected.
        """
        x1, y1, x2, y2 = original_bbox
        w, h = x2 - x1, y2 - y1
        
        # Expand by 50% in each direction for sudden movements
        expansion_x = int(w * 0.5)
        expansion_y = int(h * 0.5)
        
        expanded_x1 = max(0, x1 - expansion_x)
        expanded_y1 = max(0, y1 - expansion_y)
        expanded_x2 = x2 + expansion_x
        expanded_y2 = y2 + expansion_y
        
        return np.array([expanded_x1, expanded_y1, expanded_x2, expanded_y2], dtype=np.float32)

    def extract_contours_from_mask(self, mask, min_area=20, max_area=5000):
        """
        Extract contours from a binary mask with filtering.
        Returns list of contour points suitable for FreeHand masks.
        """
        # Threshold the mask
        mask_binary = (mask > 0.05).astype(np.uint8)
        
        # Check if mask has any content
        if np.sum(mask_binary) == 0:
            print("Mask has no content above threshold")
            return []
        
        print(f"Mask binary sum: {np.sum(mask_binary)} pixels")
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Found {len(contours)} raw contours")
        
        cell_contours = []
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            print(f"Contour {i}: area={area:.1f}")
            
            if min_area < area < max_area:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    print(f"Contour {i}: circularity={circularity:.3f}")
                    if circularity > 0.05:  # Filter by circularity
                        cell_contours.append(cnt)
                        print(f"Contour {i} passed filters")
                    else:
                        print(f"Contour {i} failed circularity filter")
                else:
                    print(f"Contour {i} has zero perimeter")
            else:
                print(f"Contour {i} failed area filter (area={area:.1f})")
        
        # If no contours pass the filter, use the largest one
        if len(cell_contours) == 0 and len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            largest_area = cv2.contourArea(largest_contour)
            print(f"No contours passed filters, using largest contour with area {largest_area:.1f}")
            cell_contours = [largest_contour]
        
        print(f"Returning {len(cell_contours)} filtered contours")
        return cell_contours

    def save_contours_as_freehand(self, contours, block, nanowell, frame, mask_idx):
        """
        Save extracted contours as FreeHand masks in the standard format.
        Each bounding box gets its own unique mask files.
        """
        # Create FreeHand directory
        freehand_dir = os.path.join(
            self.base_path, block, "Masks", "FreeHand",
            f"imgNo{nanowell}CH0", f"t{frame}"
        )
        os.makedirs(freehand_dir, exist_ok=True)
        
        # Convert contours to raw coordinate arrays
        raw_contours = []
        for contour in contours:
            # Convert contour to list of (x, y) coordinates
            pts = [(pt[0][0], pt[0][1]) for pt in contour]
            raw_contours.append(pts)
        
        # Save individual mask files for this bounding box
        H, W = self.ch0_image.shape
        for i, pts in enumerate(raw_contours):
            # Create mask from contour
            arr = np.array([(int(round(x)), int(round(y))) for x, y in pts], np.int32)
            mask = np.zeros((H, W), np.uint8)
            cv2.fillPoly(mask, [arr], 255)
            
            # Save mask with unique naming: box{mask_idx}_contour{i+1}
            mask_file = os.path.join(freehand_dir, f"box{mask_idx}_contour{i+1}.npy")
            np.save(mask_file, mask)
        
        # Also save raw contours for this box
        raw_contours_file = os.path.join(freehand_dir, f"raw_contours_box{mask_idx}.npy")
        np.save(raw_contours_file, np.array(raw_contours, dtype=object), allow_pickle=True)
        
        print(f"Saved {len(contours)} contours as FreeHand masks for box {mask_idx} in frame {frame}")

    def apply_gaussian_smoothing_to_contours(self, contours, sigma=2.0):
        """
        Apply Gaussian smoothing to a list of OpenCV contours.
        Uses scipy's gaussian_filter1d to smooth the x and y coordinates separately.
        """
        from scipy.ndimage import gaussian_filter1d
        import numpy as np
        
        smoothed_contours = []
        for contour in contours:
            if len(contour) < 3:
                # Skip contours that are too small to smooth
                smoothed_contours.append(contour)
                continue
                
            # Extract x and y coordinates
            contour_array = np.array(contour).reshape(-1, 2)
            x_coords = contour_array[:, 0].astype(float)
            y_coords = contour_array[:, 1].astype(float)
            
            # Apply Gaussian smoothing to x and y coordinates separately
            x_smoothed = gaussian_filter1d(x_coords, sigma=sigma, mode='wrap')
            y_smoothed = gaussian_filter1d(y_coords, sigma=sigma, mode='wrap')
            
            # Reconstruct the smoothed contour
            smoothed_contour = []
            for i in range(len(x_smoothed)):
                smoothed_contour.append([[int(round(x_smoothed[i])), int(round(y_smoothed[i]))]])
            
            smoothed_contours.append(np.array(smoothed_contour, dtype=np.int32))
        
        return smoothed_contours

    def apply_gaussian_smoothing_to_contour(self, contour):
        """
        Apply Gaussian smoothing to a single contour.
        """
        smoothed_contour = []
        for point in contour:
            smoothed_point = self.apply_gaussian_smoothing_to_point(point)
            smoothed_contour.append(smoothed_point)
        return smoothed_contour

    def apply_gaussian_smoothing_to_point(self, point):
        """
        Apply Gaussian smoothing to a single point.
        """
        smoothed_point = []
        for coordinate in point:
            smoothed_coordinate = self.apply_gaussian_smoothing_to_coordinate(coordinate)
            smoothed_point.append(smoothed_coordinate)
        return smoothed_point

    def apply_gaussian_smoothing_to_coordinate(self, coordinate):
        """
        Apply Gaussian smoothing to a single coordinate.
        """
        smoothed_coordinate = coordinate + np.random.normal(0, self.gaussian_sigma)
        return smoothed_coordinate





    # ------------------------- (Remaining methods: toggle_play, go_to_next_frame, etc.) -------------------------
    # (These remain largely the same as in previous versions.)
    def get_image_path(self, block, nanowell, frame, channel):
        return os.path.join(
            self.base_path,
            f'{block}/images/crops_8bit_s/imgNo{nanowell}{channel}/imgNo{nanowell}{channel}_t{frame}.tif'
        )


 


# =============================================================================
# ContourDrawingLabel: Modified to support multi-selection via hotkeys.
# =============================================================================
class ContourDrawingLabel(QLabel):
    def __init__(self, pixmap_original, scaled_pixmap, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        if not isinstance(pixmap_original, QPixmap):
            self.pixmap_original = QPixmap.fromImage(numpy_to_qimage(pixmap_original))
        else:
            self.pixmap_original = pixmap_original
        self.scaled_pixmap = scaled_pixmap
        self.setPixmap(scaled_pixmap)
        self.setFocusPolicy(Qt.StrongFocus)
        self.angular_sigma     = 5     # default Gaussian s
        self.angular_strength  = 0.5   # 0=no smoothing, 1=full smoothing
        self.angular_samples   = 100   # how many ? points to sample
        self.drawing = False
        self.current_contour = []
        self.contours = []
        self.last_point = None

        self.selected_points = set()
        self.selected_contour_index = -1

        # Multi-selection mode attributes.
        self.multi_selection_mode = False
        self.multi_selected_indices = []

        self.drawing_selection = False
        self.selection_start = QPoint()
        self.selection_end = QPoint()
        self.edit_mode = False
        self.undo_stack = []

        self.rubber_band_mode = False
        self.rubber_band_active = False
        self.rubber_band_start = None
        self.rubber_band_selected_index = None
        self.rubber_band_original_contour = None
        self.rubber_band_radius = 80

        self.raw_contours = []
        self.density_factor = 1.0

        self.region_growth_mode = False
        self.right_drag_start = None

        self.tracking_labels = []
        self.show_tracking_labels = False


        self.sam_boxes = []             # list of QRect
        self.box_mode  = False          # toggle on/off
        self._drawing_box = False       # are we in the middle of dragging?
        self._box_start    = QPoint()   # drag start
        self._box_current  = QPoint()   # drag current
        # in ContourDrawingLabel.__init__ (or right after loading raw_contours)
        # so we always have an un-touched copy
        self.angular_baseline = [list(c) for c in self.raw_contours]

    def setMultiSelectionMode(self, enabled: bool):
        self.multi_selection_mode = enabled
        if not enabled:
            self.multi_selected_indices = []
        self.update()

    def clearMultiSelection(self):
        self.multi_selected_indices = []
        self.multi_selection_mode = False
        self.update()
    def interpolate_contour(self, contour, desired_gap=3, angle_threshold_deg=170):
        """
        Interpolate points on contour while filtering collinear points
        to avoid straight-line artifacts in spline smoothing.
        """
        if not contour or len(contour) < 3:
            return contour.copy()

        def angle_between(p1, p2, p3):
            v1 = np.array([p1.x() - p2.x(), p1.y() - p2.y()])
            v2 = np.array([p3.x() - p2.x(), p3.y() - p2.y()])
            v1_norm = v1 / np.linalg.norm(v1)
            v2_norm = v2 / np.linalg.norm(v2)
            cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
            return np.degrees(np.arccos(cos_angle))

        # First, filter out very straight points
        filtered = [contour[0]]
        for i in range(1, len(contour) - 1):
            angle = angle_between(contour[i - 1], contour[i], contour[i + 1])
            if angle < angle_threshold_deg:
                filtered.append(contour[i])
        filtered.append(contour[-1])

        # Now interpolate by distance
        new_contour = [filtered[0]]
        for i in range(1, len(filtered)):
            start = filtered[i - 1]
            end = filtered[i]
            dist = self.distance(start, end)
            if dist > desired_gap:
                steps = max(2, int(dist / desired_gap))
                for j in range(1, steps):
                    ratio = j / float(steps)
                    interp_x = round(start.x() + ratio * (end.x() - start.x()))
                    interp_y = round(start.y() + ratio * (end.y() - start.y()))
                    new_contour.append(QPoint(interp_x, interp_y))
            new_contour.append(end)
        return new_contour
    def apply_coordinate_gaussian_smoothing(self, sigma: float = None):
        """
        Replace angular smoothing with independent Gaussian smoothing
        on the X and Y sequences of each raw contour.
        """
        from scipy.ndimage import gaussian_filter1d

        s = sigma if sigma is not None else self.angular_sigma  # reuse your angular_sigma as default

        new_contours = []
        for raw in self.raw_contours:
            n = len(raw)
            if n < 3:
                new_contours.append(raw.copy())
                continue

            # extract x/y as floats
            xs = np.array([p.x() for p in raw], dtype=float)
            ys = np.array([p.y() for p in raw], dtype=float)

            # apply 1D Gaussian filter independently
            xs_s = gaussian_filter1d(xs, s, mode='wrap')
            ys_s = gaussian_filter1d(ys, s, mode='wrap')

            # if the contour is closed, ensure we close the loop
            if raw[0] == raw[-1]:
                # also wrap the filter (mode='wrap') keeps it closed
                pass
            else:
                # for open contours, don't wrap ends � you could switch to mode='reflect' if you like
                xs_s = gaussian_filter1d(xs, s, mode='reflect')
                ys_s = gaussian_filter1d(ys, s, mode='reflect')

            # rebuild into QPoints
            pts = [QPoint(int(round(x)), int(round(y))) for x, y in zip(xs_s, ys_s)]
            # re-close if needed
            if raw[0] == raw[-1] and pts[0] != pts[-1]:
                pts.append(QPoint(pts[0]))

            new_contours.append(pts)

        # commit
        self.contours = new_contours
        self.update()

    def apply_angular_smoothing(self,
                                sigma: float = None,
                                num_samples: int = None,
                                strength: float = None):
        """
        Smooth each contour.  Closed-ish loops get radial smoothing;
        U-shaped (large angular gap) contours get arc-length smoothing
        with anchored endpoints so they stay U-shaped.
        """
        sigma       = sigma       if sigma       is not None else self.angular_sigma
        num_samples = num_samples if num_samples is not None else self.angular_samples
        strength    = strength    if strength    is not None else self.angular_strength

        new_contours = []
        for raw in self.raw_contours:
            n = len(raw)
            if n < 3:
                new_contours.append(raw.copy())
                continue

            # Extract XY and centroid
            xs = np.array([p.x() for p in raw], dtype=float)
            ys = np.array([p.y() for p in raw], dtype=float)
            cx, cy = xs.mean(), ys.mean()
            dx, dy = xs - cx, ys - cy

            # Compute unwrapped angles
            angs = np.unwrap(np.arctan2(dy, dx))
            ang_sorted = np.sort(angs)
            # include wrap-around gap
            wrap_gap = (ang_sorted[0] + 2*np.pi) - ang_sorted[-1]
            gaps    = np.diff(ang_sorted)
            max_gap = max(wrap_gap, np.max(gaps))

            if max_gap > np.pi:
                # --- U-shaped: arc-length smoothing without re-closing ---
                # drop the artificial last==first if present
                if raw[0] == raw[-1]:
                    xs, ys = xs[:-1], ys[:-1]

                # cumulative distance along the polyline
                ds = np.hypot(np.diff(xs), np.diff(ys))
                s  = np.concatenate(([0], np.cumsum(ds)))
                # resample to uniform parameter
                s_uniform = np.linspace(0, s[-1], num_samples)
                x_u = np.interp(s_uniform, s, xs)
                y_u = np.interp(s_uniform, s, ys)
                # gaussian filter (reflect so ends don't wrap)
                x_f = gaussian_filter1d(x_u, sigma, mode='reflect')
                y_f = gaussian_filter1d(y_u, sigma, mode='reflect')
                # blend
                x_final = (1 - strength)*x_u + strength*x_f
                y_final = (1 - strength)*y_u + strength*y_f
                # anchor endpoints
                x_final[0],  y_final[0]  = xs[0], ys[0]
                x_final[-1], y_final[-1] = xs[-1], ys[-1]

                pts = [QPoint(int(round(x)), int(round(y)))
                    for x, y in zip(x_final, y_final)]
                new_contours.append(pts)

            else:
                # --- Closed loop: radial/angular smoothing (unchanged) ---
                radii = np.hypot(dx, dy)
                idx   = np.argsort(angs)
                ang_s = angs[idx]
                rad_s = radii[idx]
                uni_ang = np.linspace(ang_s[0], ang_s[-1], num_samples)
                rad_i   = np.interp(uni_ang, ang_s, rad_s)
                rad_f   = gaussian_filter1d(rad_i, sigma, mode='wrap')
                rad_final = (1 - strength)*rad_i + strength*rad_f
                x_new = cx + rad_final * np.cos(uni_ang)
                y_new = cy + rad_final * np.sin(uni_ang)

                pts = [QPoint(int(round(x)), int(round(y)))
                    for x, y in zip(x_new, y_new)]
                pts.append(pts[0])
                new_contours.append(pts)

        self.contours = new_contours
        self.update()

    def smooth_selected_points(self):
        """
        Smooths the set of currently selected points along their spline.
        """
        if len(self.selected_points) > 2:
            # record state BEFORE smoothing
            self.undo_stack.append(self.save_current_state())

            points = sorted(self.selected_points, key=lambda p: (p[0], p[1]))
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            try:
                tck, u = splprep([x, y], s=2)
                smooth_x, smooth_y = splev(u, tck)
                self.selected_points = {
                    (int(smooth_x[i]), int(smooth_y[i]))
                    for i in range(len(points))
                }
            except Exception as e:
                print(f"Error smoothing selected points: {e}")
            self.update()

    def mousePressEvent(self, event):
        # 1) --- SAM-box drawing (if box_mode is enabled) --------------------
        if self.box_mode and event.button() == Qt.LeftButton:
            self._box_start   = event.pos()
            self._box_current = event.pos()
            self._drawing_box = True
            return

        # 2) NEW: rectangular selection in edit (nudge) mode
        if self.edit_mode and event.button() == Qt.LeftButton:
            self.drawing_selection = True
            self.selection_start   = event.pos()
            self.selection_end     = event.pos()
            return

        # 3) Multi-selection mode
        if self.multi_selection_mode and event.button() == Qt.LeftButton:
            idx = self.find_closest_contour(event.pos())
            if idx != -1:
                if idx in self.multi_selected_indices:
                    self.multi_selected_indices.remove(idx)
                    print(f"Contour {idx} deselected.")
                else:
                    self.multi_selected_indices.append(idx)
                    print(f"Contour {idx} selected.")
                self.update()
            return

        # 4) Region-growth mode
        if self.region_growth_mode:
            if event.button() == Qt.RightButton:
                self.clear_editing_state()
                self.selected_contour_index = self.find_closest_contour(event.pos())
                self.update()
                return
            elif event.button() == Qt.LeftButton and self.selected_contour_index != -1:
                self.undo_stack.append(self.save_current_state())
                self.extend_contour(event.pos())
                return

        # 5) Rubber-band edit mode
        if self.rubber_band_mode:
            if event.button() == Qt.LeftButton and 0 <= self.selected_contour_index < len(self.contours):
                self.undo_stack.append(self.save_current_state())
                contour = self.contours[self.selected_contour_index]
                # find nearest point
                min_dist, closest_i = float('inf'), None
                for i, pt in enumerate(contour):
                    d = self.distance(pt, event.pos())
                    if d < min_dist:
                        min_dist, closest_i = d, i
                if min_dist < 10:
                    self.rubber_band_selected_index   = closest_i
                    self.rubber_band_start            = event.pos()
                    self.rubber_band_original_contour = [QPoint(p) for p in contour]
                    self.rubber_band_active           = True
                return
            elif event.button() == Qt.RightButton:
                self.clear_editing_state()
                self.selected_contour_index = self.find_closest_contour(event.pos())
                self.update()
                return

        # 6) Free-hand contour drawing
        if event.button() == Qt.LeftButton:
            self.setFocus()
            if not self.edit_mode:
                self.grabMouse()
                self.drawing = True
                self.current_contour = [event.pos()]
                self.last_point = event.pos()
            return

        # 7) Right-click drag to move a contour
        if event.button() == Qt.RightButton:
            self.undo_stack.append(self.save_current_state())
            self.clear_editing_state()
            self.selected_contour_index = self.find_closest_contour(event.pos())
            self.right_drag_start = event.pos()
            self.update()
            return

        # Default
        super().mousePressEvent(event)




    def mouseMoveEvent(self, event):
        # 1) Box-drawing
        if self._drawing_box:
            self._box_current = event.pos()
            self.update()
            return

        # 2) NEW: update selection rectangle in edit mode
        if self.edit_mode and self.drawing_selection:
            self.selection_end = event.pos()
            self.update()
            return

        # 3) Right-drag moving of a contour
        if (event.buttons() & Qt.RightButton) and \
           self.right_drag_start is not None and \
           not self.rubber_band_mode:
            if self.selected_contour_index != -1:
                delta = event.pos() - self.right_drag_start
                moved = [QPoint(pt.x() + delta.x(), pt.y() + delta.y())
                         for pt in self.contours[self.selected_contour_index]]
                self.contours[self.selected_contour_index] = moved
                self.right_drag_start = event.pos()
                if not sip.isdeleted(self):
                    self.update()
            return

        # 4) Rubber-band editing movement
        if self.rubber_band_mode and self.rubber_band_active:
            delta = event.pos() - self.rubber_band_start
            orig = self.rubber_band_original_contour
            new = []
            for i, pt in enumerate(orig):
                d_idx = abs(i - self.rubber_band_selected_index)
                w = max(0, (self.rubber_band_radius - d_idx) / self.rubber_band_radius)
                new.append(QPoint(pt.x() + int(delta.x()*w),
                                  pt.y() + int(delta.y()*w)))
            self.contours[self.selected_contour_index] = new
            if not sip.isdeleted(self):
                self.update()
            return

        # 5) Freehand drawing
        if self.drawing:
            if self.last_point is None or self.distance(self.last_point, event.pos()) > 1:
                self.current_contour.append(event.pos())
                self.last_point = event.pos()
                if not sip.isdeleted(self):
                    self.update()

        # 6) Selection-rectangle drag (also catches non-edit-mode draws)
        elif self.drawing_selection:
            self.selection_end = event.pos()
            if not sip.isdeleted(self):
                self.update()

        else:
            super().mouseMoveEvent(event)


    def mouseReleaseEvent(self, event):
        # 1) Finish box-draw
        if self._drawing_box and event.button() == Qt.LeftButton:
            rect = QRect(self._box_start, self._box_current).normalized()
            self.sam_boxes.append(rect)
            self._drawing_box = False
            self.update()
            return

        # 2) Handle left-button releases
        if event.button() == Qt.LeftButton:
            # a) Finalize freehand drawing
            if self.drawing:
                self.releaseMouse()
                self.drawing = False
                if len(self.current_contour) > 1:
                    self.current_contour.append(self.current_contour[0])
                    interp = self.interpolate_contour(self.current_contour, desired_gap=3)
                    self.raw_contours.append(interp)
                    npts = int(len(interp) * 10 * self.density_factor)
                    smooth = self.smooth_drawn_contour(interp, s=20, num_points=npts)
                    self.contours.append(smooth)
                self.current_contour = []
                if not sip.isdeleted(self):
                    self.update()
                return

            # b) Finalize selection rectangle
            elif self.drawing_selection:
                self.drawing_selection = False
                self.select_points_in_rectangle()
                if not sip.isdeleted(self):
                    self.update()
                return

            # c) Finalize rubber-band
            elif self.rubber_band_mode and self.rubber_band_active:
                self.rubber_band_active = False
                if not sip.isdeleted(self):
                    self.update()
                return

        # 3) Handle right-button release (end drag)
        elif event.button() == Qt.RightButton:
            self.right_drag_start = None
            return

        super().mouseReleaseEvent(event)


    # In ContourDrawingLabel:

    def keyPressEvent(self, event):
        from PyQt5.QtCore import Qt
        # 1) Swallow C/X if no contour is selected, so they never bubble up
        if event.key() in (Qt.Key_C, Qt.Key_X) and self.selected_contour_index == -1:
            event.accept()
            return

        key = event.key()
        handled = False

        # 2) Nudge/edit mode movement of selected points
        if self.edit_mode and self.selected_points:
            dx = dy = 0
            if key == Qt.Key_Left:
                dx = -1
            elif key == Qt.Key_Right:
                dx = 1
            elif key == Qt.Key_Up:
                dy = -1
            elif key == Qt.Key_Down:
                dy = 1

            if dx or dy:
                self.undo_stack.append(self.save_current_state())
                sel = list(self.selected_points)

                # If exactly two endpoints, only move that segment (end-to-end)
                if len(sel) == 2:
                    p1, p2 = sel
                    ci = None
                    for i, c in enumerate(self.contours):
                        coords = [(pt.x(), pt.y()) for pt in c]
                        if p1 in coords and p2 in coords:
                            ci = i
                            break
                    if ci is not None:
                        contour = self.contours[ci]
                        i1 = contour.index(QPoint(*p1))
                        i2 = contour.index(QPoint(*p2))
                        n = len(contour)
                        # build both possible index ranges
                        seg1 = list(range(i1, i2 + 1)) if i1 <= i2 else list(range(i1, n)) + list(range(0, i2 + 1))
                        seg2 = list(range(i2, i1 + 1)) if i2 <= i1 else list(range(i2, n)) + list(range(0, i1 + 1))
                        segment = seg1 if len(seg1) <= len(seg2) else seg2
                        for j in segment:
                            pt = contour[j]
                            contour[j] = QPoint(pt.x() + dx, pt.y() + dy)
                        # only re-select the two moved endpoints
                        self.selected_points = {
                            (p1[0] + dx, p1[1] + dy),
                            (p2[0] + dx, p2[1] + dy)
                        }

                # Otherwise move only the explicitly selected points
                else:
                    new_sel = {(x + dx, y + dy) for x, y in sel}
                    for contour in self.contours:
                        for idx, pt in enumerate(contour):
                            if (pt.x(), pt.y()) in self.selected_points:
                                contour[idx] = QPoint(pt.x() + dx, pt.y() + dy)
                    self.selected_points = new_sel

                self.update()
                handled = True

        # 3) Undo / Smooth shortcuts
        if not handled:
            if key == Qt.Key_Z and (event.modifiers() & Qt.ControlModifier):
                self.undo_last_action()
                handled = True
            elif key == Qt.Key_S:
                self.smooth_selected_points()
                handled = True

        if handled:
            event.accept()
        else:
            super().keyPressEvent(event)





    def update_pixmap(self, pixmap: QPixmap):
        if sip.isdeleted(self):
            return
        if pixmap is None or pixmap.isNull():
            print("Invalid pixmap received!")
            return
        scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.scaled_pixmap = scaled
        self.setPixmap(scaled)
        if not sip.isdeleted(self):
            self.update()

    def save_current_state(self):
        return [[QPoint(p.x(), p.y()) for p in contour] for contour in self.contours]

    def undo_last_action(self):
        """Undo the most recent incremental editing action using the undo stack."""
        if self.undo_stack:
            self.contours = self.undo_stack.pop()
            if not sip.isdeleted(self):
                self.update()

    def undo_last_contour(self):
        """Completely remove the last drawn contour."""
        if self.contours:
            self.contours.pop()
            if self.raw_contours:
                self.raw_contours.pop()
            if not sip.isdeleted(self):
                self.update()
    def clear_contours(self):
        self.contours = []
        self.raw_contours = []
        if not sip.isdeleted(self):
            self.update()



    def find_closest_point(self, position: QPoint):
        closest = None
        min_dist = float('inf')
        for contour in self.contours:
            for point in contour:
                d = self.distance(position, point)
                if d < min_dist:
                    min_dist = d
                    closest = point
        return closest

    def find_closest_contour(self, position: QPoint):
        min_dist = float('inf')
        idx = -1
        for i, contour in enumerate(self.contours):
            for point in contour:
                d = self.distance(position, point)
                if d < min_dist:
                    min_dist = d
                    idx = i
        return idx

    def smooth_drawn_contour(self, contour, s=5, num_points=None):
        if len(contour) < 4:
            return contour
        x = [p.x() for p in contour]
        y = [p.y() for p in contour]
        closed = (contour[0] == contour[-1])
        per = 1 if closed else 0
        if num_points is None:
            num_points = len(contour) * 10
        try:
            tck, u = splprep([x, y], s=s, per=per)
            u_new = np.linspace(0, 1, num_points)
            smooth_x, smooth_y = splev(u_new, tck)
            new_contour = [QPoint(int(xi), int(yi)) for xi, yi in zip(smooth_x, smooth_y)]
            if closed:
                new_contour.append(new_contour[0])
            return new_contour
        except Exception as e:
            print(f"Error smoothing contour: {e}")
            return contour


    def clear_editing_state(self):
        self.rubber_band_active = False
        self.rubber_band_start = None
        self.rubber_band_selected_index = None
        self.rubber_band_original_contour = None
        self.selected_points.clear()
        self.drawing_selection = False




    # inside ContourDrawingLabel:
    def scale_selected_contour(self, slider_value):
        """
        Uniformly scale the currently selected contour around its centroid.
        Resets the slider to 100% and re-captures the baseline any time you switch to a new contour.
        Expects slider_value in [70..125].
        """
        idx = self.selected_contour_index
        if idx == -1:
            print("No contour selected for scaling.")
            return

        # If user has picked a new contour since last time:
        if getattr(self, '_baseline_index', None) != idx:
            # store every contour as the new "original"
            self.original_contours = [
                [QPoint(p.x(), p.y()) for p in contour]
                for contour in self.contours
            ]
            self._baseline_index = idx

            # reset the parent's slider back to 100%
            parent = self.parent()
            if hasattr(parent, 'resize_slider'):
                parent.resize_slider.blockSignals(True)
                parent.resize_slider.setValue(100)
                parent.resize_slider.blockSignals(False)

        # compute actual scale factor
        scale_factor = slider_value / 100.0

        # get the baseline contour
        original = self.original_contours[idx]
        # centroid
        cx = sum(p.x() for p in original) / len(original)
        cy = sum(p.y() for p in original) / len(original)
        center = QPoint(int(cx), int(cy))

        # rebuild scaled contour
        new_pts = []
        for p in original:
            dx = p.x() - center.x()
            dy = p.y() - center.y()
            new_pts.append(QPoint(
                int(center.x() + dx * scale_factor),
                int(center.y() + dy * scale_factor)
            ))

        # replace and repaint
        self.contours[idx] = new_pts
        self.update()
 

    def update_density(self, density_factor):
        self.density_factor = density_factor
        if self.selected_contour_index != -1 and self.selected_contour_index < len(self.raw_contours):
            raw = self.raw_contours[self.selected_contour_index]
            num_points = int(len(raw) * 10 * self.density_factor)
            smooth_contour = self.smooth_drawn_contour(raw, s=5, num_points=num_points)
            self.contours[self.selected_contour_index] = smooth_contour
            if not sip.isdeleted(self):
                self.update()

    def apply_masks_to_pixmap(self, masks, mask_type):
        self.current_mask_type = mask_type
        if hasattr(self, "fluorescence_overlay") and self.fluorescence_overlay is not None:
            # Use the current fluorescence overlay as the base.
            base = self.fluorescence_overlay.copy()
        else:
            base = qimage_to_numpy(self.pixmap_original.toImage())
        if base.ndim == 2:
            base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        
        # Instead of blending two identical images,
        # Draw colored outlines for each mask onto the base image.
        for mask in masks:
            mask = np.squeeze(mask)
            mask = (mask > 0).astype(np.uint8)
            contours_found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours_found:
                area = cv2.contourArea(cnt)
                if area > 5000:
                    continue
                # Draw the contour outline with a specified color and thickness.
                cv2.drawContours(base, [cnt], -1, (0, 0, 255), thickness=2)
        
        self.setPixmap(QPixmap.fromImage(numpy_to_qimage(base)))
        self.update()
    def select_points_in_rectangle(self):
        """
        Only select the first and last contour-points
        that lie inside the user??s drag-rectangle.
        """
        rect = QRect(self.selection_start, self.selection_end).normalized()
        self.selected_points.clear()

        for contour in self.contours:
            # find all indices of this contour inside the rect
            idxs = [i for i,p in enumerate(contour) if rect.contains(p)]
            if not idxs:
                continue
            # only keep the two ??endpoints??
            first, last = idxs[0], idxs[-1]
            p1, p2 = contour[first], contour[last]
            self.selected_points.add((p1.x(), p1.y()))
            # if there??s more than one point, add the other end
            if last != first:
                self.selected_points.add((p2.x(), p2.y()))

    def add_point_to_contour(self, position: QPoint):
        if self.selected_contour_index == -1:
            return
        contour = self.contours[self.selected_contour_index]
        scaled_position = QPoint(position.x(), position.y())
        min_dist = float('inf')
        insert_index = -1
        for i in range(len(contour) - 1):
            dist = self.distance_to_segment(scaled_position, contour[i], contour[i + 1])
            if dist < min_dist:
                min_dist = dist
                insert_index = i + 1
        contour.insert(insert_index, scaled_position)
        self.contours[self.selected_contour_index] = contour
        if not sip.isdeleted(self):
            self.update()

    def distance_to_segment(self, point: QPoint, start: QPoint, end: QPoint) -> float:
        px, py = point.x(), point.y()
        sx, sy = start.x(), start.y()
        ex, ey = end.x(), end.y()
        if sx == ex and sy == ey:
            return math.hypot(px - sx, py - sy)
        t = max(0, min(1, ((px - sx) * (ex - sx) + (py - sy) * (ey - sy)) / ((ex - sx)**2 + (ey - sy)**2)))
        proj_x = sx + t * (ex - sx)
        proj_y = sy + t * (ey - sy)
        return math.hypot(px - proj_x, py - proj_y)
      
    def paintEvent(self, event):
        if sip.isdeleted(self):
            return
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            if not self.pixmap().isNull():
                painter.drawPixmap(self.contentsRect(), self.pixmap())
            mask_color_map = {
                "GT": Qt.green,
                "MRCNN": Qt.blue,
                "FRCNN": Qt.cyan,
                "SAM": Qt.magenta,
                "FreeHand": Qt.yellow,
                "CT": Qt.darkYellow,
                "Track": Qt.red
            }
            for i, contour in enumerate(self.contours):
                if len(contour) > 1:
                    if i == self.selected_contour_index:
                        pen = QPen(Qt.yellow, 3, Qt.DashLine)
                    else:
                        if hasattr(self, "current_mask_type") and self.current_mask_type in mask_color_map:
                            color = mask_color_map[self.current_mask_type]
                        else:
                            color = Qt.red
                        pen = QPen(color, 2, Qt.SolidLine)
                    painter.setPen(pen)
                    polygon = QPolygonF([QPointF(p) for p in contour])
                    painter.drawPolygon(polygon)
            if self.drawing and len(self.current_contour) > 1:
                realtime_pen = QPen(Qt.blue, 2, Qt.DashLine)
                painter.setPen(realtime_pen)
                polyline = QPolygonF([QPointF(p) for p in self.current_contour])
                painter.drawPolyline(polyline)
            if self.drawing_selection:
                selection_rect = QRect(self.selection_start, self.selection_end)
                selection_pen = QPen(Qt.magenta, 2, Qt.DashLine)
                painter.setPen(selection_pen)
                painter.drawRect(selection_rect)
            if self.selected_points:
                # semi-transparent magenta outline
                pen = QPen(QColor(255, 0, 255, 150), 2)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                for x, y in self.selected_points:
                    painter.drawEllipse(QPoint(x, y), 4, 4)

            if self.multi_selected_indices:
                highlight_pen = QPen(Qt.white, 3, Qt.DotLine)
                painter.setPen(highlight_pen)
                for idx in self.multi_selected_indices:
                    if idx < len(self.contours):
                        poly = QPolygonF([QPointF(p) for p in self.contours[idx]])
                        painter.drawPolygon(poly)
            if self.show_tracking_labels and self.tracking_labels:
                font = painter.font()
                font.setPointSize(10)
                painter.setFont(font)
                fm = QFontMetrics(font)
                for i, contour in enumerate(self.contours):
                    if not contour:
                        continue
                    centroid_x = sum(p.x() for p in contour) / len(contour)
                    centroid_y = sum(p.y() for p in contour) / len(contour)
                    label = self.tracking_labels[i] if i < len(self.tracking_labels) else ""
                    text_width = fm.horizontalAdvance(label)
                    text_height = fm.height()
                    padding = 4
                    bg_rect = QRectF(
                        centroid_x - (text_width / 2) - padding,
                        centroid_y - (text_height / 2) - padding,
                        text_width + 2 * padding,
                        text_height + 2 * padding
                    )
                    painter.save()
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QColor(0, 0, 0, 150))
                    painter.drawRoundedRect(bg_rect, 5, 5)
                    painter.restore()
                    painter.setPen(QPen(Qt.white))
                    painter.drawText(bg_rect, Qt.AlignCenter, label)
            if self.box_mode:
                pen = QPen(Qt.green, 2, Qt.DashLine)
                painter.setPen(pen)
                for rect in self.sam_boxes:
                    painter.drawRect(rect)
                # also draw the ??rubber-band?? while dragging
                if self._drawing_box:
                    curr = QRect(self._box_start, self._box_current).normalized()
                    painter.drawRect(curr)
        finally:
            painter.end()

    @staticmethod
    def distance(p1: QPoint, p2: QPoint) -> float:
        return math.hypot(p1.x() - p2.x(), p1.y() - p2.y())

    def extend_contour(self, pos: QPoint):
        """
        Displace contour points toward the click position, 
        with influence falling off over self.region_growth_radius pixels.
        """
        idx = self.selected_contour_index
        if idx == -1:
            print("No contour selected to extend.")
            return

        contour = self.contours[idx]
        # Compute the displacement vector at the click location
        # (you may also want to snap this to the nearest contour point)
        closest_pt = min(contour, key=lambda p: self.distance(p, pos))
        displacement = pos - closest_pt

        R = getattr(self, 'region_growth_radius', 80)  # px
        new_contour = []
        for pt in contour:
            d = self.distance(pt, pos)
            if d <= R and d > 0:
                weight = (R - d) / R
                new_x = pt.x() + int(displacement.x() * weight)
                new_y = pt.y() + int(displacement.y() * weight)
            else:
                # outside radius: no movement
                new_x, new_y = pt.x(), pt.y()
            new_contour.append(QPoint(new_x, new_y))

        self.contours[idx] = new_contour
        if not sip.isdeleted(self):
            self.update()
        print(f"Extended contour {idx} with radius {R}px and click at {pos}.")


# End of file.


if __name__=='__main__':
    import sys
    app = QApplication(sys.argv)
    if len(sys.argv)<3:
        print('Usage: contour_zoom_widget.py <base_path> <block-nanowell-tFrame> [mask_type]')
        sys.exit(1)
    w = ContourZoomWidget(
        sys.argv[1],
        sys.argv[2],
        mask_type=(sys.argv[3] if len(sys.argv)>3 else None)
    )
    w.show()
    sys.exit(app.exec_())








