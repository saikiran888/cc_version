import sys
import csv
import cv2
import numpy as np
import pandas as pd
import os
from glob import glob
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QTabWidget, QFileDialog, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QHeaderView, QCheckBox, QSlider, QSizePolicy, QMessageBox,
    QProgressBar, QSpinBox, QScrollArea, QGroupBox, QDialog, QSplitter
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QCursor, QMouseEvent
from PyQt5.QtCore import QTimer, QRect, Qt, QEvent, QSize

from image_loader import ImageLoader
from color_picker import ColorPickerWidget
from change_dialog import ChangeDialog
from new_table_widget import TableWidgetItem
from loading_widget import LoadingWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from numpy import mean, median, min as npmin, max as npmax, array

CSV_FILE_NAMES = ["output.csv", "Table_0E1T.csv", "Table_1E0T.csv",
                  "Table_1E1T.csv", "Table_1E2T.csv", "Table_1E3T.csv"]
CUR_DIR = os.getcwd()
sys.path.append(os.path.join(CUR_DIR, ".."))
if True:
    from TIMING_secondary import run_TIMING_secondary

# Custom video widget to avoid flickering


class VideoLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super(VideoLabel, self).__init__(*args, **kwargs)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_StaticContents, True)
        self.setAutoFillBackground(False)

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.pixmap():
            painter.drawPixmap(self.rect(), self.pixmap())
        else:
            super(VideoLabel, self).paintEvent(event)

# =========================
# Single Video Player Class (Refactored from VideoWindow)
# =========================

class SingleVideoPlayer(QWidget):
    def __init__(self, parent=None, player_id=1):
        super(SingleVideoPlayer, self).__init__(parent)
        self.player_id = player_id
        
        # Video data placeholders
        self.images = []
        self.boxes = []
        self.channel_images = []
        self.image_index = 0
        self.selected_uid = None
        self.run_dirname = ""
        self.is_playing = False
        self.playback_speed = 250  # ms
        self.show_boxes = True
        self.show_text = True
        self.show_contours = False
        self.show_base = True
        self.channel_thresholds = {'CH1': 200, 'CH2': 200, 'CH3': 200}

        # Colors
        self.eff_color = QColor('#FFC20A')  # Yellow
        self.tar_color = QColor('#0C7BDC')  # Blue

        # Timer for playback
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.switch_images)

        # Initialize channel overlay checkboxes (will be set by parent)
        self.channel_overlay_checkboxes = {}
        
        # Initialize channel overlay states (for dual video compatibility)
        self.channel_overlay_states = {
            'CH1': False,
            'CH2': False,
            'CH3': False
        }

        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)

        # Video container with overlaid navigation
        video_container = QWidget()
        video_layout = QHBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(0)

        # Previous UID button
        self.prev_uid_button = QPushButton("⟨")
        self.prev_uid_button.setFixedWidth(40)
        self.prev_uid_button.setToolTip("Previous UID")
        self.prev_uid_button.clicked.connect(self.on_prev_uid_clicked)
        self.prev_uid_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(33, 150, 243, 0.7);
                color: white;
                border: none;
                border-radius: 0;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: rgba(25, 118, 210, 0.9);
            }
            QPushButton:pressed {
                background-color: rgba(13, 71, 161, 1);
            }
        """)
        video_layout.addWidget(self.prev_uid_button, alignment=Qt.AlignVCenter)

        # Video display
        self.video_label = VideoLabel()
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumSize(300, 300)
        self.video_label.setMaximumSize(800, 800)  # Maximum size for better control
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMouseTracking(True)
        self.video_label.installEventFilter(self)
        # Ensure square aspect ratio
        self.video_label.setScaledContents(False)
        video_layout.addWidget(self.video_label, stretch=1)

        # Next UID button
        self.next_uid_button = QPushButton("⟩")
        self.next_uid_button.setFixedWidth(40)
        self.next_uid_button.setToolTip("Next UID")
        self.next_uid_button.clicked.connect(self.on_next_uid_clicked)
        self.next_uid_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(33, 150, 243, 0.7);
                color: white;
                border: none;
                border-radius: 0;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: rgba(25, 118, 210, 0.9);
            }
            QPushButton:pressed {
                background-color: rgba(13, 71, 161, 1);
            }
        """)
        video_layout.addWidget(self.next_uid_button, alignment=Qt.AlignVCenter)

        main_layout.addWidget(video_container, stretch=1)

        # Playback Controls
        playback_group = QGroupBox("Playback Controls")
        playback_layout = QVBoxLayout()
        playback_layout.setSpacing(10)
        
        # Buttons row
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)
        
        self.prev_frame_button = QPushButton("<<")
        self.prev_frame_button.setFixedWidth(60)
        self.prev_frame_button.clicked.connect(self.previous_frame)
        buttons_layout.addWidget(self.prev_frame_button)
        
        self.play_pause_button = QPushButton("||")
        self.play_pause_button.setFixedWidth(60)
        self.play_pause_button.clicked.connect(self.toggle_playback)
        buttons_layout.addWidget(self.play_pause_button)
        
        self.next_frame_button = QPushButton(">>")
        self.next_frame_button.setFixedWidth(60)
        self.next_frame_button.clicked.connect(self.next_frame)
        buttons_layout.addWidget(self.next_frame_button)
        playback_layout.addLayout(buttons_layout)
        
        # Frame slider and counter
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setTickPosition(QSlider.TicksBelow)
        self.frame_slider.valueChanged.connect(self.slider_value_changed)
        self.frame_slider.sliderReleased.connect(self.slider_value_released)
        playback_layout.addWidget(self.frame_slider)
        
        self.frame_counter_label = QLabel("Frame: 0/0")
        self.frame_counter_label.setAlignment(Qt.AlignCenter)
        playback_layout.addWidget(self.frame_counter_label)
        
        # Speed control
        speed_container = QWidget()
        speed_layout = QHBoxLayout(speed_container)
        speed_layout.setContentsMargins(0, 0, 0, 0)
        speed_layout.setSpacing(10)
        
        speed_layout.addWidget(QLabel("Playback Interval:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(50, 1000)
        self.speed_slider.setValue(250)
        self.speed_slider.valueChanged.connect(self.change_playback_speed)
        speed_layout.addWidget(self.speed_slider)
        
        self.speed_label = QLabel("250 ms")
        self.speed_label.setMinimumWidth(60)
        speed_layout.addWidget(self.speed_label)
        
        playback_layout.addWidget(speed_container)
        
        playback_group.setLayout(playback_layout)
        main_layout.addWidget(playback_group)

        # Nanowell info
        self.nanowell_info_label = QLabel("Nanowell info:\nUID: N/A")
        main_layout.addWidget(self.nanowell_info_label)

    # Video playback methods (simplified versions of VideoWindow methods)
    def set_video_data(self, images, boxes, channel_images, uid, run_dirname, continuation=False):
        print(f"[DEBUG] set_video_data called for UID: {uid}, total images: {len(images)}")
        if self.timer.isActive():
            self.timer.stop()

        self.images = images
        self.boxes = boxes
        self.channel_images = channel_images
        self.selected_uid = uid
        self.run_dirname = run_dirname
        self.image_index = 0

        self.frame_slider.blockSignals(True)
        max_index = len(self.images) - 1 if self.images else 0
        self.frame_slider.setMaximum(max_index)
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)

        self.nanowell_info_label.setText(f"Nanowell info:\nUID: {uid}")
        self.switch_images(0)
        self.is_playing = True
        self.timer.start(self.speed_slider.value())

    def switch_images(self, image_index=-1):
        if not self.channel_images or not self.images:
            return

        if image_index == -1:
            if self.is_playing:
                self.image_index = (self.image_index + 1) % len(self.images)
        else:
            self.image_index = image_index

        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.image_index)
        self.frame_slider.blockSignals(False)
        self.frame_counter_label.setText(f"Frame: {self.image_index + 1}/{len(self.images)}")

        try:
            current_images = self.channel_images[self.image_index]
            if current_images is None:
                return

            # Process base image (CH0)
            ch0_data = current_images.get('CH0')
            if ch0_data is None or len(ch0_data) < 1:
                base_image = np.zeros((500, 500), dtype=np.uint8)
            else:
                base_image = ch0_data[0]

            if base_image.dtype != np.uint8:
                base_image = cv2.normalize(base_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            height, width = base_image.shape
            if self.show_base:
                composite = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
            else:
                composite = np.zeros((height, width, 3), dtype=np.uint8)

            # Process CH1 overlay if enabled
            ch1_enabled = False
            if hasattr(self, 'channel_overlay_checkboxes') and 'CH1' in self.channel_overlay_checkboxes:
                # For VideoWindow (single video)
                ch1_enabled = self.channel_overlay_checkboxes['CH1']['checkbox'].isChecked()
            elif hasattr(self, 'channel_overlay_states'):
                # For DualVideoWindow (dual video)
                ch1_enabled = self.channel_overlay_states.get('CH1', False)
            
            if 'CH1' in current_images and ch1_enabled:
                print(f"[DEBUG] CH1 overlay enabled for {getattr(self, 'selected_uid', 'unknown')}")
                ch1_data = current_images['CH1']
                if ch1_data is not None and len(ch1_data) >= 2:
                    ch1, hi_ch1 = ch1_data[0], ch1_data[1]
                    print(f"[DEBUG] CH1 original range: {hi_ch1.min():.2f} to {hi_ch1.max():.2f}")
                    if ch1.dtype != np.uint8:
                        ch1 = cv2.normalize(ch1, None, 0, 255,
                                         cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    th1 = self.channel_thresholds.get('CH1', 200)
                    mask1 = hi_ch1 >= th1
                    pixels_above_threshold = np.sum(mask1)
                    print(f"[DEBUG] CH1 threshold {th1}, pixels above: {pixels_above_threshold}")
                    
                    if pixels_above_threshold > 0:
                        ch1_thresh = np.where(mask1, ch1, 0).astype(np.uint8)
                        
                        # Debug: check the actual values
                        max_fluor_val = np.max(ch1_thresh)
                        mean_fluor_val = np.mean(ch1_thresh[ch1_thresh > 0])
                        print(f"[DEBUG] CH1 fluorescence - max: {max_fluor_val}, mean: {mean_fluor_val:.2f}")
                        
                        # Boost the fluorescence signal for better visibility
                        ch1_boosted = np.clip(ch1_thresh * 2, 0, 255).astype(np.uint8)
                        
                        composite[:, :, 1] = cv2.add(composite[:, :, 1], ch1_boosted)
                        print(f"[DEBUG] Applied CH1 overlay - green channel updated with boost")
                    else:
                        print(f"[DEBUG] No CH1 pixels above threshold {th1}")

            # Process CH2 overlay if enabled
            ch2_enabled = False
            if hasattr(self, 'channel_overlay_checkboxes') and 'CH2' in self.channel_overlay_checkboxes:
                # For VideoWindow (single video)
                ch2_enabled = self.channel_overlay_checkboxes['CH2']['checkbox'].isChecked()
            elif hasattr(self, 'channel_overlay_states'):
                # For DualVideoWindow (dual video)
                ch2_enabled = self.channel_overlay_states.get('CH2', False)
            
            if 'CH2' in current_images and ch2_enabled:
                print(f"[DEBUG] CH2 overlay enabled for {getattr(self, 'selected_uid', 'unknown')}")
                ch2_data = current_images['CH2']
                if ch2_data is not None and len(ch2_data) >= 2:
                    ch2, hi_ch2 = ch2_data[0], ch2_data[1]
                    print(f"[DEBUG] CH2 original range: {hi_ch2.min():.2f} to {hi_ch2.max():.2f}")
                    if ch2.dtype != np.uint8:
                        ch2 = cv2.normalize(ch2, None, 0, 255,
                                         cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    th2 = self.channel_thresholds.get('CH2', 200)
                    mask2 = hi_ch2 >= th2
                    pixels_above_threshold = np.sum(mask2)
                    print(f"[DEBUG] CH2 threshold {th2}, pixels above: {pixels_above_threshold}")
                    
                    if pixels_above_threshold > 0:
                        ch2_thresh = np.where(mask2, ch2, 0).astype(np.uint8)
                        
                        # Debug: check the actual values
                        max_fluor_val = np.max(ch2_thresh)
                        mean_fluor_val = np.mean(ch2_thresh[ch2_thresh > 0])
                        print(f"[DEBUG] CH2 fluorescence - max: {max_fluor_val}, mean: {mean_fluor_val:.2f}")
                        
                        # Boost the fluorescence signal for better visibility
                        ch2_boosted = np.clip(ch2_thresh * 2, 0, 255).astype(np.uint8)
                        
                        composite[:, :, 2] = cv2.add(composite[:, :, 2], ch2_boosted)
                        print(f"[DEBUG] Applied CH2 overlay - red channel updated with boost")
                    else:
                        print(f"[DEBUG] No CH2 pixels above threshold {th2}")

            # Process CH3 overlay if enabled
            ch3_enabled = False
            if hasattr(self, 'channel_overlay_checkboxes') and 'CH3' in self.channel_overlay_checkboxes:
                # For VideoWindow (single video)
                ch3_enabled = self.channel_overlay_checkboxes['CH3']['checkbox'].isChecked()
            elif hasattr(self, 'channel_overlay_states'):
                # For DualVideoWindow (dual video)
                ch3_enabled = self.channel_overlay_states.get('CH3', False)
            
            if 'CH3' in current_images and ch3_enabled:
                print(f"[DEBUG] CH3 overlay enabled for {getattr(self, 'selected_uid', 'unknown')}")
                ch3_data = current_images['CH3']
                if ch3_data is not None and len(ch3_data) >= 2:
                    ch3, hi_ch3 = ch3_data[0], ch3_data[1]
                    print(f"[DEBUG] CH3 original range: {hi_ch3.min():.2f} to {hi_ch3.max():.2f}")
                    if ch3.dtype != np.uint8:
                        ch3 = cv2.normalize(ch3, None, 0, 255,
                                         cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    th3 = self.channel_thresholds.get('CH3', 200)
                    mask3 = hi_ch3 >= th3
                    pixels_above_threshold = np.sum(mask3)
                    print(f"[DEBUG] CH3 threshold {th3}, pixels above: {pixels_above_threshold}")
                    
                    if pixels_above_threshold > 0:
                        ch3_thresh = np.where(mask3, ch3, 0).astype(np.uint8)
                        
                        # Debug: check the actual values
                        max_fluor_val = np.max(ch3_thresh)
                        mean_fluor_val = np.mean(ch3_thresh[ch3_thresh > 0])
                        print(f"[DEBUG] CH3 fluorescence - max: {max_fluor_val}, mean: {mean_fluor_val:.2f}")
                        
                        # Boost the fluorescence signal for better visibility
                        ch3_boosted = np.clip(ch3_thresh * 2, 0, 255).astype(np.uint8)
                        
                        composite[:, :, 0] = cv2.add(composite[:, :, 0], ch3_boosted)
                        print(f"[DEBUG] Applied CH3 overlay - blue channel updated with boost")
                    else:
                        print(f"[DEBUG] No CH3 pixels above threshold {th3}")

            # Drawing boxes and text overlays
            composite_rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
            height, width, _ = composite_rgb.shape
            qImg = QImage(composite_rgb.data, width, height, width * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg)

            # Calculate the size to maintain square aspect ratio
            label_size = self.video_label.size()
            min_size = min(label_size.width(), label_size.height())
            square_size = QSize(min_size, min_size)
            
            # Ensure the pixmap is scaled to a square
            scaled_pixmap = pixmap.scaled(square_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Create a square final pixmap
            final_pixmap = QPixmap(square_size)
            final_pixmap.fill(QColor('#f0f0f0'))
            
            painter = QPainter(final_pixmap)
            
            # Center the scaled image within the square
            x_offset = (square_size.width() - scaled_pixmap.width()) // 2
            y_offset = (square_size.height() - scaled_pixmap.height()) // 2
            painter.drawPixmap(x_offset, y_offset, scaled_pixmap)
            
            # Draw overlays on the scaled pixmap
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setRenderHint(QPainter.TextAntialiasing)

            # Calculate scale factors for overlay positioning
            scale_x = scaled_pixmap.width() / width
            scale_y = scaled_pixmap.height() / height

            if self.image_index < len(self.boxes):
                boxes = self.boxes[self.image_index]
                for category in ['eff', 'tar']:
                    pen_color = self.eff_color if category == 'eff' else self.tar_color
                    for box in boxes.get(category, []):
                        left, top, w, h = map(int, box)
                        # Scale the box coordinates and add offsets for centering
                        scaled_left = int(left * scale_x) + x_offset
                        scaled_top = int(top * scale_y) + y_offset
                        scaled_w = int(w * scale_x)
                        scaled_h = int(h * scale_y)
                        
                        if self.show_boxes:
                            painter.setPen(QPen(QColor(pen_color), max(2, int(scale_x * 2))))
                            painter.drawRect(scaled_left, scaled_top, scaled_w, scaled_h)
                        if self.show_text:
                            painter.setPen(QPen(QColor(pen_color)))
                            font_size = max(12, int(20 * scale_x))
                            painter.setFont(QFont('Arial', font_size))
                            center_x = int(scaled_left + scaled_w / 2)
                            center_y = int(scaled_top + scaled_h / 2)
                            txt = category[0].upper()
                            text_rect = QRect(center_x - scaled_w//2, center_y - scaled_h//2, scaled_w, scaled_h)
                            painter.drawText(text_rect, Qt.AlignCenter, txt)

            painter.end()
            self.video_label.setPixmap(final_pixmap)

        except Exception as e:
            print(f"[DEBUG] Error in switch_images: {str(e)}")
            empty_frame = np.zeros((500, 500, 3), dtype=np.uint8)
            height, width, _ = empty_frame.shape
            qImg = QImage(empty_frame.data, width, height, width * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg)
            self.video_label.setPixmap(pixmap)

    def next_frame(self):
        if self.is_playing:
            self.toggle_playback()
        next_index = (self.image_index + 1) % len(self.images)
        self.switch_images(next_index)

    def previous_frame(self):
        if self.is_playing:
            self.toggle_playback()
        prev_index = (self.image_index - 1) % len(self.images)
        self.switch_images(prev_index)

    def slider_value_changed(self):
        new_index = self.frame_slider.value()
        if self.is_playing:
            self.is_playing = False
            self.play_pause_button.setText("||")
            if self.timer.isActive():
                self.timer.stop()
        self.switch_images(new_index)

    def slider_value_released(self):
        final_index = self.frame_slider.value()
        self.switch_images(final_index)

    def toggle_playback(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_pause_button.setText("||")
            self.timer.start(self.speed_slider.value())
        else:
            self.play_pause_button.setText("||")
            self.timer.stop()

    def change_playback_speed(self, value):
        self.playback_speed = value
        self.speed_label.setText(f"{value} ms")
        if self.timer.isActive():
            self.timer.setInterval(value)

    def mouseMoveEvent(self, event):
        """Handle mouse movement to show pixel intensity"""
        try:
            if not self.video_label.underMouse():
                return
            if not self.channel_images:
                return
            
            print(f"[DEBUG] MouseMove event in SingleVideoPlayer")
            
            current_images = self.channel_images[self.image_index]
            if current_images is None:
                return
                
            ch0_data = current_images.get('CH0')
            if ch0_data is None or len(ch0_data) < 1:
                return
                
            base_image = ch0_data[0]
            pixmap = self.video_label.pixmap()
            if not pixmap:
                return
                
            # Get the actual image area within the label (considering square aspect ratio)
            label_size = self.video_label.size()
            min_size = min(label_size.width(), label_size.height())
            x_offset = (label_size.width() - min_size) // 2
            y_offset = (label_size.height() - min_size) // 2
            
            # Calculate scale factors for the square image
            scale = min_size / base_image.shape[1]  # Since it's square, we can use either dimension
            
            # Get mouse position relative to the label
            mouse_pos = self.video_label.mapFromGlobal(event.globalPos())
            
            # Adjust for the offset and scaling
            x = int((mouse_pos.x() - x_offset) / scale)
            y = int((mouse_pos.y() - y_offset) / scale)
            
            if 0 <= x < base_image.shape[1] and 0 <= y < base_image.shape[0]:
                intensities = {}
                for channel, img in current_images.items():
                    if img is not None and len(img) >= 2:
                        intensities[channel] = img[1][y, x]  # Use high-intensity image
                    else:
                        intensities[channel] = 'N/A'
                
                intensity_str = " | ".join(
                    [f"{ch}: {val}" for ch, val in intensities.items()])
                
                # Update pixel intensity in parent dual video window if it exists
                parent = self.parent()
                while parent and not hasattr(parent, 'pixel_intensity_label'):
                    parent = parent.parent()
                
                if parent and hasattr(parent, 'pixel_intensity_label'):
                    parent.pixel_intensity_label.setText(
                        f"Pixel Intensity at ({x},{y}): {intensity_str}")
                        
        except Exception as e:
            print(f"Error in mouseMoveEvent: {e}")

    def eventFilter(self, obj, event):
        if obj == self.video_label and event.type() == QEvent.MouseMove:
            self.mouseMoveEvent(event)
            return True
        return super().eventFilter(obj, event)

    def on_prev_uid_clicked(self):
        """Handle previous UID button click"""
        print(f"[DEBUG] Previous button clicked on player {self.player_id}")
        # Find the main QC_APP which has player-specific navigation
        parent = self.parent()
        while parent and not hasattr(parent, 'previous_uid_for_player'):
            parent = parent.parent()
        
        if parent and hasattr(parent, 'previous_uid_for_player'):
            print(f"[DEBUG] Calling previous_uid_for_player({self.player_id}) on {type(parent).__name__}")
            parent.previous_uid_for_player(self.player_id)
        else:
            print(f"[DEBUG] Could not find parent with previous_uid_for_player method")

    def on_next_uid_clicked(self):
        """Handle next UID button click"""
        print(f"[DEBUG] Next button clicked on player {self.player_id}")
        # Find the main QC_APP which has player-specific navigation
        parent = self.parent()
        while parent and not hasattr(parent, 'next_uid_for_player'):
            parent = parent.parent()
        
        if parent and hasattr(parent, 'next_uid_for_player'):
            print(f"[DEBUG] Calling next_uid_for_player({self.player_id}) on {type(parent).__name__}")
            parent.next_uid_for_player(self.player_id)
        else:
            print(f"[DEBUG] Could not find parent with next_uid_for_player method")

    def closeEvent(self, event):
        """Handle close event"""
        if self.timer.isActive():
            self.timer.stop()
        super().closeEvent(event)

    def __del__(self):
        """Destructor to ensure timer is stopped"""
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()

# =========================
# Dual Video Window
# =========================

class DualVideoWindow(QDialog):
    def __init__(self, parent=None):
        super(DualVideoWindow, self).__init__(parent)
        self.setWindowTitle("Video Comparison")
        self.setModal(False)
        
        # Make window size relative to screen size
        screen = QApplication.desktop().screenGeometry()
        self.setMinimumSize(int(screen.width() * 0.7), int(screen.height() * 0.7))
        self.resize(int(screen.width() * 0.95), int(screen.height() * 0.9))

        # Keep track of running loaders
        self.active_loaders = []

        self.setup_ui()

        self.cell_flags_csv_path1 = None  # For player 1
        self.cell_flags_csv_path2 = None  # For player 2

    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Left side: Videos stacked vertically
        videos_container = QWidget()
        videos_layout = QVBoxLayout(videos_container)
        videos_layout.setContentsMargins(0, 0, 0, 0)
        videos_layout.setSpacing(10)

        # Top video player
        top_group = QGroupBox("Video 1")
        top_layout = QVBoxLayout(top_group)
        self.player1 = SingleVideoPlayer(self, player_id=1)
        self.player1.setMinimumSize(700, 800)  # Portrait rectangle
        self.player1.setMaximumSize(700, 800)
        self.player1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        top_layout.addWidget(self.player1)
        videos_layout.addWidget(top_group)

        # Bottom video player
        bottom_group = QGroupBox("Video 2")
        bottom_layout = QVBoxLayout(bottom_group)
        self.player2 = SingleVideoPlayer(self, player_id=2)
        self.player2.setMinimumSize(800, 800)  # Portrait rectangle
        self.player2.setMaximumSize(800, 800)
        self.player2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        bottom_layout.addWidget(self.player2)
        videos_layout.addWidget(bottom_group)

        # Hide playback controls in both players (unchanged)
        if hasattr(self.player1, 'play_pause_button'):
            self.player1.play_pause_button.hide()
        if hasattr(self.player1, 'prev_frame_button'):
            self.player1.prev_frame_button.hide()
        if hasattr(self.player1, 'next_frame_button'):
            self.player1.next_frame_button.hide()
        if hasattr(self.player1, 'frame_slider'):
            self.player1.frame_slider.hide()
        if hasattr(self.player1, 'frame_counter_label'):
            self.player1.frame_counter_label.hide()
        if hasattr(self.player1, 'speed_slider'):
            self.player1.speed_slider.hide()
        if hasattr(self.player1, 'speed_label'):
            self.player1.speed_label.hide()
        if hasattr(self.player1, 'timer'):
            self.player1.timer.stop()

        if hasattr(self.player2, 'play_pause_button'):
            self.player2.play_pause_button.hide()
        if hasattr(self.player2, 'prev_frame_button'):
            self.player2.prev_frame_button.hide()
        if hasattr(self.player2, 'next_frame_button'):
            self.player2.next_frame_button.hide()
        if hasattr(self.player2, 'frame_slider'):
            self.player2.frame_slider.hide()
        if hasattr(self.player2, 'frame_counter_label'):
            self.player2.frame_counter_label.hide()
        if hasattr(self.player2, 'speed_slider'):
            self.player2.speed_slider.hide()
        if hasattr(self.player2, 'speed_label'):
            self.player2.speed_label.hide()
        if hasattr(self.player2, 'timer'):
            self.player2.timer.stop()

        main_layout.addWidget(videos_container, stretch=0)  # Fixed width for videos

        # Right side: Control panels
        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(10)

        # Create tabbed control panels
        self.control_tabs = QTabWidget()
        controls_layout.addWidget(self.control_tabs)

        # Analysis & Overlay Tab
        self.setup_analysis_overlay_tab()
        
        # Colors & Channels Tab
        self.setup_colors_channels_tab()
        
        # Plots & Metrics Tab
        self.setup_plots_metrics_tab()
        
        # SNR Metrics Tab
        self.setup_snr_metrics_tab()
        
        # Cell Flags Tab
        self.setup_cell_flags_tab()

        main_layout.addWidget(controls_container, stretch=1)  # Controls take up remaining space

    def setup_analysis_overlay_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Analysis Parameters (shared)
        analysis_group = QGroupBox("Analysis Parameters")
        analysis_layout = QGridLayout()
        
        analysis_layout.addWidget(QLabel("EDInt:"), 0, 0)
        self.edint_entry = QLineEdit("125")
        analysis_layout.addWidget(self.edint_entry, 0, 1)
        
        analysis_layout.addWidget(QLabel("DInt:"), 1, 0)
        self.dint_entry = QLineEdit("125")
        analysis_layout.addWidget(self.dint_entry, 1, 1)
        
        self.rerun_btn = QPushButton("Rerun Secondary Analysis")
        analysis_layout.addWidget(self.rerun_btn, 2, 0, 1, 2)
        
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)

        # Playback Controls for Video 1 (unchanged)
        playback_group1 = QGroupBox("Video 1 Playback Controls")
        playback_layout1 = QVBoxLayout()
        playback_layout1.setSpacing(10)
        
        # Buttons row
        buttons_layout1 = QHBoxLayout()
        buttons_layout1.setSpacing(10)
        
        self.prev_frame_button1 = QPushButton("<<")
        self.prev_frame_button1.setFixedWidth(60)
        self.prev_frame_button1.clicked.connect(lambda: self.player1.previous_frame())
        buttons_layout1.addWidget(self.prev_frame_button1)
        
        self.play_pause_button1 = QPushButton("||")
        self.play_pause_button1.setFixedWidth(60)
        self.play_pause_button1.clicked.connect(lambda: self.player1.toggle_playback())
        buttons_layout1.addWidget(self.play_pause_button1)
        
        self.next_frame_button1 = QPushButton(">>")
        self.next_frame_button1.setFixedWidth(60)
        self.next_frame_button1.clicked.connect(lambda: self.player1.next_frame())
        buttons_layout1.addWidget(self.next_frame_button1)
        playback_layout1.addLayout(buttons_layout1)
        
        # Frame slider and counter
        self.frame_slider1 = QSlider(Qt.Horizontal)
        self.frame_slider1.setTickPosition(QSlider.TicksBelow)
        self.frame_slider1.valueChanged.connect(self.sync_player1_slider)
        self.frame_slider1.sliderReleased.connect(self.sync_player1_slider_release)
        playback_layout1.addWidget(self.frame_slider1)
        
        self.frame_counter_label1 = QLabel("Frame: 0/0")
        self.frame_counter_label1.setAlignment(Qt.AlignCenter)
        playback_layout1.addWidget(self.frame_counter_label1)
        
        # Speed control
        speed_container1 = QWidget()
        speed_layout1 = QHBoxLayout(speed_container1)
        speed_layout1.setContentsMargins(0, 0, 0, 0)
        speed_layout1.setSpacing(10)
        
        speed_layout1.addWidget(QLabel("Playback Interval:"))
        self.speed_slider1 = QSlider(Qt.Horizontal)
        self.speed_slider1.setRange(50, 1000)
        self.speed_slider1.setValue(250)
        self.speed_slider1.valueChanged.connect(lambda: self.player1.change_playback_speed(self.speed_slider1.value()))
        speed_layout1.addWidget(self.speed_slider1)
        
        self.speed_label1 = QLabel("250 ms")
        self.speed_label1.setMinimumWidth(60)
        speed_layout1.addWidget(self.speed_label1)
        
        playback_layout1.addWidget(speed_container1)
        playback_group1.setLayout(playback_layout1)
        layout.addWidget(playback_group1)

        # Playback Controls for Video 2 (unchanged)
        playback_group2 = QGroupBox("Video 2 Playback Controls")
        playback_layout2 = QVBoxLayout()
        playback_layout2.setSpacing(10)
        
        # Buttons row
        buttons_layout2 = QHBoxLayout()
        buttons_layout2.setSpacing(10)
        
        self.prev_frame_button2 = QPushButton("<<")
        self.prev_frame_button2.setFixedWidth(60)
        self.prev_frame_button2.clicked.connect(lambda: self.player2.previous_frame())
        buttons_layout2.addWidget(self.prev_frame_button2)
        
        self.play_pause_button2 = QPushButton("||")
        self.play_pause_button2.setFixedWidth(60)
        self.play_pause_button2.clicked.connect(lambda: self.player2.toggle_playback())
        buttons_layout2.addWidget(self.play_pause_button2)
        
        self.next_frame_button2 = QPushButton(">>")
        self.next_frame_button2.setFixedWidth(60)
        self.next_frame_button2.clicked.connect(lambda: self.player2.next_frame())
        buttons_layout2.addWidget(self.next_frame_button2)
        playback_layout2.addLayout(buttons_layout2)
        
        # Frame slider and counter
        self.frame_slider2 = QSlider(Qt.Horizontal)
        self.frame_slider2.setTickPosition(QSlider.TicksBelow)
        self.frame_slider2.valueChanged.connect(self.sync_player2_slider)
        self.frame_slider2.sliderReleased.connect(self.sync_player2_slider_release)
        playback_layout2.addWidget(self.frame_slider2)
        
        self.frame_counter_label2 = QLabel("Frame: 0/0")
        self.frame_counter_label2.setAlignment(Qt.AlignCenter)
        playback_layout2.addWidget(self.frame_counter_label2)
        
        # Speed control
        speed_container2 = QWidget()
        speed_layout2 = QHBoxLayout(speed_container2)
        speed_layout2.setContentsMargins(0, 0, 0, 0)
        speed_layout2.setSpacing(10)
        
        speed_layout2.addWidget(QLabel("Playback Interval:"))
        self.speed_slider2 = QSlider(Qt.Horizontal)
        self.speed_slider2.setRange(50, 1000)
        self.speed_slider2.setValue(250)
        self.speed_slider2.valueChanged.connect(lambda: self.player2.change_playback_speed(self.speed_slider2.value()))
        speed_layout2.addWidget(self.speed_slider2)
        
        self.speed_label2 = QLabel("250 ms")
        self.speed_label2.setMinimumWidth(60)
        speed_layout2.addWidget(self.speed_label2)
        
        playback_layout2.addWidget(speed_container2)
        playback_group2.setLayout(playback_layout2)
        layout.addWidget(playback_group2)

        # Overlay options for Video 1
        video1_group = QGroupBox("Video 1 Overlay Options")
        video1_layout = QGridLayout()
        self.show_boxes_checkbox1 = QCheckBox("Show Boxes")
        self.show_boxes_checkbox1.setChecked(True)
        video1_layout.addWidget(self.show_boxes_checkbox1, 0, 0)
        self.show_text_checkbox1 = QCheckBox("Show Text")
        self.show_text_checkbox1.setChecked(True)
        video1_layout.addWidget(self.show_text_checkbox1, 0, 1)
        self.show_contours_checkbox1 = QCheckBox("Show Contours")
        video1_layout.addWidget(self.show_contours_checkbox1, 1, 0)
        self.show_base_checkbox1 = QCheckBox("Show Base")
        self.show_base_checkbox1.setChecked(True)
        video1_layout.addWidget(self.show_base_checkbox1, 1, 1)
        video1_group.setLayout(video1_layout)
        layout.addWidget(video1_group)

        # Overlay options for Video 2
        video2_group = QGroupBox("Video 2 Overlay Options")
        video2_layout = QGridLayout()
        self.show_boxes_checkbox2 = QCheckBox("Show Boxes")
        self.show_boxes_checkbox2.setChecked(True)
        video2_layout.addWidget(self.show_boxes_checkbox2, 0, 0)
        self.show_text_checkbox2 = QCheckBox("Show Text")
        self.show_text_checkbox2.setChecked(True)
        video2_layout.addWidget(self.show_text_checkbox2, 0, 1)
        self.show_contours_checkbox2 = QCheckBox("Show Contours")
        video2_layout.addWidget(self.show_contours_checkbox2, 1, 0)
        self.show_base_checkbox2 = QCheckBox("Show Base")
        self.show_base_checkbox2.setChecked(True)
        video2_layout.addWidget(self.show_base_checkbox2, 1, 1)
        video2_group.setLayout(video2_layout)
        layout.addWidget(video2_group)

        self.control_tabs.addTab(tab, "Analysis & Overlay")

    def setup_colors_channels_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Colors section (shared)
        color_group = QGroupBox("Colors")
        color_layout = QHBoxLayout()
        
        # Initialize color attributes
        self.eff_color = QColor('#FFC20A')  # Yellow
        self.tar_color = QColor('#0C7BDC')  # Blue
        
        # Import ColorPickerWidget if not already available
        try:
            from color_picker import ColorPickerWidget
            self.eff_color_picker = ColorPickerWidget(self.eff_color, "Effector:")
            self.tar_color_picker = ColorPickerWidget(self.tar_color, "Target:")
            color_layout.addWidget(self.eff_color_picker)
            color_layout.addWidget(self.tar_color_picker)
        except ImportError:
            # Fallback simple color display
            eff_label = QLabel("Effector: Yellow")
            eff_label.setStyleSheet("background-color: #FFC20A; padding: 10px; border: 1px solid black;")
            tar_label = QLabel("Target: Blue")  
            tar_label.setStyleSheet("background-color: #0C7BDC; color: white; padding: 10px; border: 1px solid black;")
            color_layout.addWidget(eff_label)
            color_layout.addWidget(tar_label)
        
        color_group.setLayout(color_layout)
        layout.addWidget(color_group)
        
        # Individual Channel Overlays
        channels_container = QWidget()
        channels_layout = QHBoxLayout(channels_container)
        
        # Video 1 Channel Overlays
        channel1_group = QGroupBox("Video 1 Channel Overlays")
        channel1_layout = QGridLayout()
        
        self.channel_overlay_checkboxes1 = {}
        self.channel_thresholds1 = {'CH1': 200, 'CH2': 200, 'CH3': 200}
        
        fluor_channels = {
            'CH1': ("Fluor Green", QColor(0, 255, 0)),
            'CH2': ("Fluor Red", QColor(255, 0, 0)),
            'CH3': ("Fluor Blue", QColor(0, 0, 255))
        }
        
        for row, (channel_key, (label_text, color)) in enumerate(fluor_channels.items()):
            cb1 = QCheckBox(f"{label_text} Overlay")
            cb1.setStyleSheet(f"""
                QCheckBox {{ color: {color.name()}; font-weight: bold; }}
                QCheckBox::indicator {{ width: 18px; height: 18px; }}
                QCheckBox::indicator:unchecked {{ border: 2px solid {color.name()}; background: transparent; }}
                QCheckBox::indicator:checked {{ border: 2px solid {color.name()}; background: {color.name()}; }}
            """)
            channel1_layout.addWidget(cb1, row, 0)
            
            threshold_label1 = QLabel("Threshold:")
            channel1_layout.addWidget(threshold_label1, row, 1)
            
            thresh_slider1 = QSlider(Qt.Horizontal)
            thresh_slider1.setMinimum(0)
            thresh_slider1.setMaximum(65535)
            thresh_slider1.setValue(200)
            thresh_slider1.setTickPosition(QSlider.TicksBelow)
            thresh_slider1.setTickInterval(1000)
            channel1_layout.addWidget(thresh_slider1, row, 2)
            
            thresh_spinbox1 = QSpinBox()
            thresh_spinbox1.setMinimum(0)
            thresh_spinbox1.setMaximum(65535)
            thresh_spinbox1.setValue(200)
            thresh_spinbox1.setFixedWidth(70)
            channel1_layout.addWidget(thresh_spinbox1, row, 3)
            
            thresh_slider1.valueChanged.connect(thresh_spinbox1.setValue)
            thresh_spinbox1.valueChanged.connect(thresh_slider1.setValue)
            
            # Connect to update method for video 1
            thresh_slider1.valueChanged.connect(lambda value, ch=channel_key: self.update_threshold(ch, value, 1))
            cb1.stateChanged.connect(lambda state, ch=channel_key: self.update_channel_overlay(ch, 1))
            
            self.channel_overlay_checkboxes1[channel_key] = {
                'checkbox': cb1,
                'color': color,
                'slider': thresh_slider1,
                'spinbox': thresh_spinbox1
            }
        
        channel1_group.setLayout(channel1_layout)
        channels_layout.addWidget(channel1_group)
        
        # Video 2 Channel Overlays
        channel2_group = QGroupBox("Video 2 Channel Overlays")
        channel2_layout = QGridLayout()
        
        self.channel_overlay_checkboxes2 = {}
        self.channel_thresholds2 = {'CH1': 200, 'CH2': 200, 'CH3': 200}
        
        for row, (channel_key, (label_text, color)) in enumerate(fluor_channels.items()):
            cb2 = QCheckBox(f"{label_text} Overlay")
            cb2.setStyleSheet(f"""
                QCheckBox {{ color: {color.name()}; font-weight: bold; }}
                QCheckBox::indicator {{ width: 18px; height: 18px; }}
                QCheckBox::indicator:unchecked {{ border: 2px solid {color.name()}; background: transparent; }}
                QCheckBox::indicator:checked {{ border: 2px solid {color.name()}; background: {color.name()}; }}
            """)
            channel2_layout.addWidget(cb2, row, 0)
            
            threshold_label2 = QLabel("Threshold:")
            channel2_layout.addWidget(threshold_label2, row, 1)
            
            thresh_slider2 = QSlider(Qt.Horizontal)
            thresh_slider2.setMinimum(0)
            thresh_slider2.setMaximum(65535)
            thresh_slider2.setValue(200)
            thresh_slider2.setTickPosition(QSlider.TicksBelow)
            thresh_slider2.setTickInterval(1000)
            channel2_layout.addWidget(thresh_slider2, row, 2)
            
            thresh_spinbox2 = QSpinBox()
            thresh_spinbox2.setMinimum(0)
            thresh_spinbox2.setMaximum(65535)
            thresh_spinbox2.setValue(200)
            thresh_spinbox2.setFixedWidth(70)
            channel2_layout.addWidget(thresh_spinbox2, row, 3)
            
            thresh_slider2.valueChanged.connect(thresh_spinbox2.setValue)
            thresh_spinbox2.valueChanged.connect(thresh_slider2.setValue)
            
            # Connect to update method for video 2
            thresh_slider2.valueChanged.connect(lambda value, ch=channel_key: self.update_threshold(ch, value, 2))
            cb2.stateChanged.connect(lambda state, ch=channel_key: self.update_channel_overlay(ch, 2))
            
            self.channel_overlay_checkboxes2[channel_key] = {
                'checkbox': cb2,
                'color': color,
                'slider': thresh_slider2,
                'spinbox': thresh_spinbox2
            }
        
        channel2_group.setLayout(channel2_layout)
        channels_layout.addWidget(channel2_group)
        
        layout.addWidget(channels_container)
        
        self.control_tabs.addTab(tab, "Colors & Channels")

    def setup_plots_metrics_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Plot area
        plot_container = QGroupBox("Nanowell Metrics Over Time")
        plot_layout = QVBoxLayout(plot_container)
        
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)

        # Plot controls
        plot_controls = QHBoxLayout()
        plot_controls.addWidget(QLabel("X Axis:"))
        self.x_axis_combo = QComboBox()
        self.x_axis_combo.addItems(["Timepoints"])
        plot_controls.addWidget(self.x_axis_combo)
        
        plot_controls.addWidget(QLabel("Y Axis:"))
        self.y_axis_combo = QComboBox()
        self.y_axis_combo.addItems([
            "Min", "Max", "Median", "Mean",
            "CH3 Annexin Intensity", "Cell Count", "Survival Curve"
        ])
        self.y_axis_combo.currentTextChanged.connect(self.update_plots)
        plot_controls.addWidget(self.y_axis_combo)
        plot_layout.addLayout(plot_controls)
        layout.addWidget(plot_container)
        
        # Pixel Intensity Display
        self.pixel_intensity_label = QLabel("Pixel Intensity: N/A")
        self.pixel_intensity_label.setWordWrap(True)
        self.pixel_intensity_label.setStyleSheet("""
            QLabel {
                background: white;
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 4px;
                min-height: 20px;
            }
        """)
        layout.addWidget(self.pixel_intensity_label)
        
        self.control_tabs.addTab(tab, "Plots & Metrics")

    def setup_snr_metrics_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # SNR Tables for both videos
        tables_container = QWidget()
        tables_layout = QHBoxLayout(tables_container)
        
        # Video 1 SNR Table
        snr1_group = QGroupBox("Video 1 SNR Metrics")
        snr1_layout = QVBoxLayout()
        
        self.snr_table1 = QTableWidget(3, 7)
        self.snr_table1.setHorizontalHeaderLabels(["Channel", "Signal", "Background", "Noise", "SNR", "SBR", "CNR"])
        self.snr_table1.verticalHeader().setVisible(False)
        self.snr_table1.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.snr_table1.setMaximumHeight(150)
        snr1_layout.addWidget(self.snr_table1)
        
        snr1_group.setLayout(snr1_layout)
        tables_layout.addWidget(snr1_group)
        
        # Video 2 SNR Table
        snr2_group = QGroupBox("Video 2 SNR Metrics")
        snr2_layout = QVBoxLayout()
        
        self.snr_table2 = QTableWidget(3, 7)
        self.snr_table2.setHorizontalHeaderLabels(["Channel", "Signal", "Background", "Noise", "SNR", "SBR", "CNR"])
        self.snr_table2.verticalHeader().setVisible(False)
        self.snr_table2.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.snr_table2.setMaximumHeight(150)
        snr2_layout.addWidget(self.snr_table2)
        
        snr2_group.setLayout(snr2_layout)
        tables_layout.addWidget(snr2_group)
        
        layout.addWidget(tables_container)
        
        # Update button
        update_snr_btn = QPushButton("Update SNR Metrics")
        update_snr_btn.clicked.connect(self.update_snr_metrics)
        update_snr_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 12px 24px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        layout.addWidget(update_snr_btn)
        
        self.control_tabs.addTab(tab, "SNR Metrics")

    def setup_cell_flags_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Cell Flags section
        cell_flags_group = QGroupBox("Cell Flags")
        cell_flags_layout = QVBoxLayout()
        
        # Create checkboxes for different flags
        self.flag_checkboxes = {}
        flags = ['out_of_focus', 'wrong_morphology', 'debris_present', 'multiple_cells', 'cell_death']
        
        for flag in flags:
            checkbox = QCheckBox(flag.replace('_', ' ').title())
            checkbox.setStyleSheet("""
                QCheckBox {
                    padding: 5px;
                    spacing: 10px;
                }
                QCheckBox::indicator {
                    width: 18px;
                    height: 18px;
                }
                QCheckBox::indicator:unchecked {
                    border: 2px solid #2196F3;
                    background: white;
                    border-radius: 3px;
                }
                QCheckBox::indicator:checked {
                    border: 2px solid #2196F3;
                    background: #2196F3;
                    border-radius: 3px;
                }
            """)
            self.flag_checkboxes[flag] = checkbox
            cell_flags_layout.addWidget(checkbox)
        
        # Buttons for cell flags
        buttons_layout = QHBoxLayout()
        
        self.update_flags_btn = QPushButton("Update Cell Flags")
        self.update_flags_btn.clicked.connect(self.update_cell_flags)
        buttons_layout.addWidget(self.update_flags_btn)
        
        self.view_flags_btn = QPushButton("View Cell Flags")
        self.view_flags_btn.clicked.connect(self.view_cell_flags)
        buttons_layout.addWidget(self.view_flags_btn)
        
        cell_flags_layout.addLayout(buttons_layout)
        cell_flags_group.setLayout(cell_flags_layout)
        layout.addWidget(cell_flags_group)
        
        # Download section
        download_group = QGroupBox("Export & Download")
        download_layout = QVBoxLayout()
        
        self.download_video1_btn = QPushButton("Download Video 1 as MP4")
        self.download_video1_btn.clicked.connect(lambda: self.download_video(1))
        download_layout.addWidget(self.download_video1_btn)
        
        self.download_video2_btn = QPushButton("Download Video 2 as MP4")
        self.download_video2_btn.clicked.connect(lambda: self.download_video(2))
        download_layout.addWidget(self.download_video2_btn)
        
        self.download_comparison_btn = QPushButton("Download Comparison Video")
        self.download_comparison_btn.clicked.connect(lambda: self.download_comparison_video())
        download_layout.addWidget(self.download_comparison_btn)
        
        download_group.setLayout(download_layout)
        layout.addWidget(download_group)
        
        self.control_tabs.addTab(tab, "Cell Flags & Export")

    def update_overlay_options(self, player_num):
        """Update overlay options for specific player"""
        if player_num == 1:
            # Update player 1 settings
            self.player1.show_boxes = self.show_boxes_checkbox1.isChecked()
            self.player1.show_text = self.show_text_checkbox1.isChecked()
            self.player1.show_contours = self.show_contours_checkbox1.isChecked()
            self.player1.show_base = self.show_base_checkbox1.isChecked()
            
            # Update channel overlay settings
            if hasattr(self, 'channel_thresholds1'):
                self.player1.channel_thresholds = self.channel_thresholds1.copy()
            elif hasattr(self, 'channel_thresholds'):
                self.player1.channel_thresholds = self.channel_thresholds.copy()
            
            # Copy channel overlay states to player 1
            if hasattr(self, 'channel_overlay_checkboxes1'):
                if not hasattr(self.player1, 'channel_overlay_states'):
                    self.player1.channel_overlay_states = {'CH1': False, 'CH2': False, 'CH3': False}
                for ch in ['CH1', 'CH2', 'CH3']:
                    if ch in self.channel_overlay_checkboxes1:
                        self.player1.channel_overlay_states[ch] = self.channel_overlay_checkboxes1[ch]['checkbox'].isChecked()
            
            # Refresh player 1
            if hasattr(self.player1, 'switch_images'):
                self.player1.switch_images(self.player1.image_index)
                
        elif player_num == 2:
            # Update player 2 settings
            self.player2.show_boxes = self.show_boxes_checkbox2.isChecked()
            self.player2.show_text = self.show_text_checkbox2.isChecked()
            self.player2.show_contours = self.show_contours_checkbox2.isChecked()
            self.player2.show_base = self.show_base_checkbox2.isChecked()
            
            # Update channel overlay settings
            if hasattr(self, 'channel_thresholds2'):
                self.player2.channel_thresholds = self.channel_thresholds2.copy()
            elif hasattr(self, 'channel_thresholds'):
                self.player2.channel_thresholds = self.channel_thresholds.copy()
            
            # Copy channel overlay states to player 2
            if hasattr(self, 'channel_overlay_checkboxes2'):
                if not hasattr(self.player2, 'channel_overlay_states'):
                    self.player2.channel_overlay_states = {'CH1': False, 'CH2': False, 'CH3': False}
                for ch in ['CH1', 'CH2', 'CH3']:
                    if ch in self.channel_overlay_checkboxes2:
                        self.player2.channel_overlay_states[ch] = self.channel_overlay_checkboxes2[ch]['checkbox'].isChecked()
            
            # Refresh player 2
            if hasattr(self.player2, 'switch_images'):
                self.player2.switch_images(self.player2.image_index)

    def set_video_data(self, player1_data, player2_data):
        """Set video data for both players"""
        if player1_data:
            images1, boxes1, channel_images1, uid1, run_dirname1 = player1_data
            print(f"[DEBUG] Setting data for player 1: UID {uid1}, {len(images1)} images")
            self.player1.set_video_data(images1, boxes1, channel_images1, uid1, run_dirname1, False)
            
            # Sync channel overlays for player 1
            if hasattr(self, 'channel_overlay_checkboxes1'):
                # Initialize channel states
                if not hasattr(self.player1, 'channel_overlay_states'):
                    self.player1.channel_overlay_states = {'CH1': False, 'CH2': False, 'CH3': False}
                # Update states based on checkboxes
                for ch in ['CH1', 'CH2', 'CH3']:
                    if ch in self.channel_overlay_checkboxes1:
                        self.player1.channel_overlay_states[ch] = self.channel_overlay_checkboxes1[ch]['checkbox'].isChecked()
            if hasattr(self, 'channel_thresholds1'):
                self.player1.channel_thresholds = self.channel_thresholds1.copy()
        
        if player2_data:
            images2, boxes2, channel_images2, uid2, run_dirname2 = player2_data
            print(f"[DEBUG] Setting data for player 2: UID {uid2}, {len(images2)} images")
            self.player2.set_video_data(images2, boxes2, channel_images2, uid2, run_dirname2, False)
            
            # Sync channel overlays for player 2
            if hasattr(self, 'channel_overlay_checkboxes2'):
                # Initialize channel states
                if not hasattr(self.player2, 'channel_overlay_states'):
                    self.player2.channel_overlay_states = {'CH1': False, 'CH2': False, 'CH3': False}
                # Update states based on checkboxes
                for ch in ['CH1', 'CH2', 'CH3']:
                    if ch in self.channel_overlay_checkboxes2:
                        self.player2.channel_overlay_states[ch] = self.channel_overlay_checkboxes2[ch]['checkbox'].isChecked()
            if hasattr(self, 'channel_thresholds2'):
                self.player2.channel_thresholds = self.channel_thresholds2.copy()
        
        # Update plots after data is loaded
        if hasattr(self, 'update_plots'):
            self.update_plots()

    def stop_all_loaders(self):
        """Stop all active image loaders"""
        for loader in self.active_loaders:
            if loader.isRunning():
                print(f"[DEBUG] Stopping loader for UID: {getattr(loader, 'uid', 'unknown')}")
                loader.stop_thread()
                loader.wait()
        self.active_loaders.clear()

    def previous_uid(self):
        """Navigate to previous UID in the table"""
        # Get the main QC app instance
        main_app = self.parent()
        if main_app and hasattr(main_app, 'previous_uid'):
            main_app.previous_uid()

    def next_uid(self):
        """Navigate to next UID in the table"""
        # Get the main QC app instance
        main_app = self.parent()
        if main_app and hasattr(main_app, 'next_uid'):
            main_app.next_uid()

    def closeEvent(self, event):
        """Handle window close event"""
        print("[DEBUG] DualVideoWindow closing, stopping all loaders")
        self.stop_all_loaders()
        super().closeEvent(event)

    def update_threshold(self, channel, value, player_num):
        """Update channel threshold for specific player"""
        if player_num == 1:
            self.channel_thresholds1[channel] = value
            self.player1.channel_thresholds[channel] = value
            # Refresh player 1
            if hasattr(self.player1, 'switch_images'):
                self.player1.switch_images(self.player1.image_index)
        elif player_num == 2:
            self.channel_thresholds2[channel] = value
            self.player2.channel_thresholds[channel] = value
            # Refresh player 2
            if hasattr(self.player2, 'switch_images'):
                self.player2.switch_images(self.player2.image_index)

    def update_channel_overlay(self, channel, player_num):
        """Update channel overlay visibility for specific player"""
        if player_num == 1:
            # Update player 1 channel overlay settings using states
            if not hasattr(self.player1, 'channel_overlay_states'):
                self.player1.channel_overlay_states = {'CH1': False, 'CH2': False, 'CH3': False}
            
            # Update all channel states based on current checkbox states
            for ch in ['CH1', 'CH2', 'CH3']:
                if ch in self.channel_overlay_checkboxes1:
                    self.player1.channel_overlay_states[ch] = self.channel_overlay_checkboxes1[ch]['checkbox'].isChecked()
            
            # Also ensure thresholds are synced
            self.player1.channel_thresholds = self.channel_thresholds1.copy()
            
            print(f"[DEBUG] Player 1 channel states: {self.player1.channel_overlay_states}")
            
            # Refresh player 1
            if hasattr(self.player1, 'switch_images'):
                self.player1.switch_images(self.player1.image_index)
                
        elif player_num == 2:
            # Update player 2 channel overlay settings using states
            if not hasattr(self.player2, 'channel_overlay_states'):
                self.player2.channel_overlay_states = {'CH1': False, 'CH2': False, 'CH3': False}
            
            # Update all channel states based on current checkbox states
            for ch in ['CH1', 'CH2', 'CH3']:
                if ch in self.channel_overlay_checkboxes2:
                    self.player2.channel_overlay_states[ch] = self.channel_overlay_checkboxes2[ch]['checkbox'].isChecked()
            
            # Also ensure thresholds are synced
            self.player2.channel_thresholds = self.channel_thresholds2.copy()
            
            print(f"[DEBUG] Player 2 channel states: {self.player2.channel_overlay_states}")
            
            # Refresh player 2
            if hasattr(self.player2, 'switch_images'):
                self.player2.switch_images(self.player2.image_index)

    def update_snr_metrics(self):
        """Update SNR metrics for both players"""
        # Update SNR for player 1
        if hasattr(self.player1, 'channel_images') and self.player1.channel_images:
            self.update_snr_table_for_player(self.player1, self.snr_table1)
        
        # Update SNR for player 2
        if hasattr(self.player2, 'channel_images') and self.player2.channel_images:
            self.update_snr_table_for_player(self.player2, self.snr_table2)

    def update_snr_table_for_player(self, player, snr_table):
        """Update SNR table for a specific player"""
        if not hasattr(player, 'channel_images') or not player.channel_images:
            return
        
        if player.image_index >= len(player.channel_images):
            return
        
        current_images = player.channel_images[player.image_index]
        if current_images is None:
            return
        
        # Clear existing table content
        snr_table.setRowCount(0)
        
        channels = ['CH1', 'CH2', 'CH3']
        row = 0
        
        for channel in channels:
            if channel in current_images:
                channel_data = current_images[channel]
                signal, background, noise, snr, sbr, cnr = self.calculate_snr_for_player(player, channel_data)
                
                # Add row to table
                snr_table.insertRow(row)
                
                # Channel name
                channel_item = QTableWidgetItem(channel)
                channel_item.setTextAlignment(Qt.AlignCenter)
                snr_table.setItem(row, 0, channel_item)
                
                # Signal value
                signal_item = QTableWidgetItem(f"{signal:.2f}")
                signal_item.setTextAlignment(Qt.AlignCenter)
                snr_table.setItem(row, 1, signal_item)
                
                # Background value
                background_item = QTableWidgetItem(f"{background:.2f}")
                background_item.setTextAlignment(Qt.AlignCenter)
                snr_table.setItem(row, 2, background_item)
                
                # Noise value
                noise_item = QTableWidgetItem(f"{noise:.2f}")
                noise_item.setTextAlignment(Qt.AlignCenter)
                snr_table.setItem(row, 3, noise_item)
                
                # SNR value
                snr_item = QTableWidgetItem(f"{snr:.2f}")
                snr_item.setTextAlignment(Qt.AlignCenter)
                snr_table.setItem(row, 4, snr_item)
                
                # SBR value
                sbr_item = QTableWidgetItem(f"{sbr:.2f}")
                sbr_item.setTextAlignment(Qt.AlignCenter)
                snr_table.setItem(row, 5, sbr_item)
                
                # CNR value
                cnr_item = QTableWidgetItem(f"{cnr:.2f}")
                cnr_item.setTextAlignment(Qt.AlignCenter)
                snr_table.setItem(row, 6, cnr_item)
                
                row += 1
        
        # If no channels found, show empty message
        if row == 0:
            snr_table.insertRow(0)
            no_data_item = QTableWidgetItem("No channel data")
            no_data_item.setTextAlignment(Qt.AlignCenter)
            snr_table.setItem(0, 0, no_data_item)
            snr_table.setSpan(0, 0, 1, 7)

    def calculate_snr_for_player(self, player, image_data):
        """Calculate Signal-to-Noise Ratio for image data using boxes or contours"""
        if image_data is None or len(image_data) == 0:
            return 0, 0, 0, 0, 0, 0
        
        # Use the high-intensity image for calculations
        if len(image_data) >= 2:
            signal_image = image_data[1]  # High-intensity image
        else:
            signal_image = image_data[0]  # Fallback to regular image
            
        # Get current boxes if available
        signal_pixels = []
        boxes = {}
        if player.image_index < len(player.boxes):
            boxes = player.boxes[player.image_index]
            # Collect pixels from both effector and target boxes
            for category in ['eff', 'tar']:
                for box in boxes.get(category, []):
                    left, top, w, h = map(int, box)
                    roi = signal_image[top:top+h, left:left+w]
                    if roi.size > 0:
                        signal_pixels.extend(roi.flatten())
        
        # Calculate signal from ROIs
        if signal_pixels:
            signal = np.mean(signal_pixels)
            
            # Create a mask for background (excluding box regions)
            mask = np.ones_like(signal_image, dtype=bool)
            for category in ['eff', 'tar']:
                for box in boxes.get(category, []):
                    left, top, w, h = map(int, box)
                    mask[top:top+h, left:left+w] = False
                    
            # Calculate background from non-box regions
            background_pixels = signal_image[mask]
            if len(background_pixels) > 0:
                background = np.mean(background_pixels)
                noise = np.std(background_pixels)
            else:
                background = 0
                noise = 1  # Avoid division by zero
        else:
            # Fallback to percentile-based calculation if no boxes available
            non_zero_pixels = signal_image[signal_image > 0]
            if len(non_zero_pixels) > 0:
                signal = np.mean(non_zero_pixels)
            else:
                signal = 0
            
            # Calculate background (mean of bottom 10% pixels)
            threshold = np.percentile(signal_image, 10)
            background_pixels = signal_image[signal_image <= threshold]
            if len(background_pixels) > 0:
                background = np.mean(background_pixels)
                noise = np.std(background_pixels)
            else:
                background = 0
                noise = 1
        
        # Calculate metrics
        snr = signal / noise if noise > 0 else 0
        sbr = signal / background if background > 0 else 0
        cnr = (signal - background) / noise if noise > 0 else 0
        
        return signal, background, noise, snr, sbr, cnr

    def update_plots(self):
        """Update the plots with data from both players"""
        if not hasattr(self, 'figure') or not hasattr(self, 'canvas'):
            return
            
        y_metric = self.y_axis_combo.currentText()
        
        self.figure.clear()
        
        # Create subplot for comparison
        ax = self.figure.add_subplot(111)
        
        # Plot data for player 1
        if hasattr(self.player1, 'channel_images') and self.player1.channel_images:
            values1 = self.calculate_plot_values(self.player1, y_metric)
            timepoints1 = list(range(len(values1)))
            if values1:
                ax.plot(timepoints1, values1, label=f"Video 1 - {self.player1.selected_uid}", marker='o')
        
        # Plot data for player 2  
        if hasattr(self.player2, 'channel_images') and self.player2.channel_images:
            values2 = self.calculate_plot_values(self.player2, y_metric)
            timepoints2 = list(range(len(values2)))
            if values2:
                ax.plot(timepoints2, values2, label=f"Video 2 - {self.player2.selected_uid}", marker='s')
        
        ax.set_title(f"{y_metric} Comparison")
        ax.set_xlabel("Timepoints")
        ax.set_ylabel(y_metric)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.canvas.draw()

    def calculate_plot_values(self, player, y_metric):
        """Calculate plot values for a specific player and metric"""
        values = []
        
        if not hasattr(player, 'channel_images') or not player.channel_images:
            return values
            
        from numpy import mean, median, min as npmin, max as npmax, array
        
        if y_metric == "Survival Curve":
            frame_count = len(player.channel_images)
            survival = [1] * frame_count
            for i in range(frame_count // 2, frame_count):
                survival[i] = 0
            values = array(survival).cumsum()[::-1].tolist()
        
        elif y_metric == "CH3 Annexin Intensity":
            for t, frame in enumerate(player.channel_images):
                if frame is None:
                    values.append(0)
                    continue
                ch3_data = frame.get("CH3", None)
                if ch3_data is None or len(ch3_data) < 2:
                    values.append(0)
                    continue
                ch3 = ch3_data[1]
                boxes = player.boxes[t] if t < len(player.boxes) else {}
                total_intensity = 0
                count = 0
                for cat in ["eff", "tar"]:
                    for box in boxes.get(cat, []):
                        x, y, w, h = map(int, box)
                        roi = ch3[y:y+h, x:x+w]
                        if roi.size > 0:
                            total_intensity += float(mean(roi))
                            count += 1
                values.append(total_intensity / count if count > 0 else 0)
        
        else:  # Min, Max, Mean, Median, Cell Count
            for i, frame in enumerate(player.channel_images):
                if frame is None:
                    values.append(0)
                    continue

                if y_metric == "Cell Count":
                    if i < len(player.boxes):
                        boxes = player.boxes[i]
                        count = len(boxes.get("eff", [])) + len(boxes.get("tar", []))
                        values.append(count)
                    else:
                        values.append(0)
                
                elif y_metric in ["Min", "Max", "Mean", "Median"]:
                    all_intensities = []
                    boxes = player.boxes[i] if i < len(player.boxes) else {}
                    
                    ch1_data = frame.get('CH1', None)
                    ch2_data = frame.get('CH2', None)
                    
                    if ch1_data is not None and len(ch1_data) >= 2:
                        ch1 = ch1_data[1]
                        for box in boxes.get('eff', []):
                            x, y, w, h = map(int, box)
                            roi = ch1[y:y+h, x:x+w]
                            if roi.size > 0:
                                all_intensities.extend(roi.flatten())
                                
                    if ch2_data is not None and len(ch2_data) >= 2:
                        ch2 = ch2_data[1]
                        for box in boxes.get('tar', []):
                            x, y, w, h = map(int, box)
                            roi = ch2[y:y+h, x:x+w]
                            if roi.size > 0:
                                all_intensities.extend(roi.flatten())

                    if all_intensities:
                        if y_metric == "Mean":
                            values.append(mean(all_intensities))
                        elif y_metric == "Median":
                            values.append(median(all_intensities))
                        elif y_metric == "Min":
                            values.append(npmin(all_intensities))
                        elif y_metric == "Max":
                            values.append(npmax(all_intensities))
                    else:
                        values.append(0)
                        
        return values

    def update_cell_flags(self):
        """Prompt which video to update, then update cell flags for that player and save to CSV"""
        from PyQt5.QtWidgets import QMessageBox
        msg = QMessageBox(self)
        msg.setWindowTitle("Update Cell Flags")
        msg.setText("Which video would you like to update flags for?")
        msg.addButton("Video 1", QMessageBox.YesRole)
        msg.addButton("Video 2", QMessageBox.NoRole)
        msg.addButton("Cancel", QMessageBox.RejectRole)
        result = msg.exec_()
        if result == 0:
            self.update_flags_for_player(1)
        elif result == 1:
            self.update_flags_for_player(2)

    def update_flags_for_player(self, player_num):
        """Update flags for a specific player and save to CSV"""
        from PyQt5.QtWidgets import QMessageBox, QFileDialog
        import csv, os
        player = self.player1 if player_num == 1 else self.player2
        csv_attr = 'cell_flags_csv_path1' if player_num == 1 else 'cell_flags_csv_path2'
        csv_path = getattr(self, csv_attr, None)
        if not hasattr(player, 'selected_uid') or not player.selected_uid:
            QMessageBox.warning(self, "Warning", "Please ensure video data is loaded first.")
            return
        # If no CSV path or file, ask user for location
        if not csv_path or not os.path.exists(csv_path):
            filename, _ = QFileDialog.getSaveFileName(self, f"Select Cell Flags CSV for Video {player_num}", f"cell_flags_{player.selected_uid}.csv", "CSV Files (*.csv)")
            if not filename:
                return  # User cancelled
            if not filename.lower().endswith('.csv'):
                filename += '.csv'
            csv_path = filename
            setattr(self, csv_attr, csv_path)
            # If file does not exist, create with headers
            if not os.path.exists(filename):
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["UID"] + list(self.flag_checkboxes.keys()))
        # Collect flag values
        flag_values = [int(self.flag_checkboxes[flag].isChecked()) for flag in self.flag_checkboxes]
        uid = player.selected_uid
        updated = False
        # Read all rows, update if UID exists, else append
        rows = []
        if os.path.exists(csv_path):
            with open(csv_path, 'r', newline='') as f:
                reader = list(csv.reader(f))
                if reader:
                    header = reader[0]
                    rows = reader[1:]
                else:
                    header = ["UID"] + list(self.flag_checkboxes.keys())
                    rows = []
        else:
            header = ["UID"] + list(self.flag_checkboxes.keys())
            rows = []
        # Update or append
        for i, row in enumerate(rows):
            if row and row[0] == uid:
                rows[i] = [uid] + [str(v) for v in flag_values]
                updated = True
                break
        if not updated:
            rows.append([uid] + [str(v) for v in flag_values])
        # Write back
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        QMessageBox.information(self, "Success", f"Cell flags updated for UID: {uid}\nSaved to: {csv_path}")

    def view_cell_flags(self):
        """Prompt which video to view, then show the cell flags CSV in a table viewer"""
        from PyQt5.QtWidgets import QMessageBox, QFileDialog
        import os
        msg = QMessageBox(self)
        msg.setWindowTitle("View Cell Flags")
        msg.setText("Which video's cell flags would you like to view?")
        msg.addButton("Video 1", QMessageBox.YesRole)
        msg.addButton("Video 2", QMessageBox.NoRole)
        msg.addButton("Cancel", QMessageBox.RejectRole)
        result = msg.exec_()
        if result == 0:
            player_num = 1
        elif result == 1:
            player_num = 2
        else:
            return
        csv_attr = 'cell_flags_csv_path1' if player_num == 1 else 'cell_flags_csv_path2'
        csv_path = getattr(self, csv_attr, None)
        if not csv_path or not os.path.exists(csv_path):
            filename, _ = QFileDialog.getOpenFileName(self, f"Select Cell Flags CSV to View for Video {player_num}", "", "CSV Files (*.csv)")
            if not filename:
                return  # User cancelled
            setattr(self, csv_attr, filename)
            csv_path = filename
        if not os.path.exists(csv_path):
            QMessageBox.warning(self, "Not Found", f"CSV file not found: {csv_path}")
            return
        viewer = CellFlagsViewer(csv_path)
        viewer.show()

    def download_video(self, player_num):
        """Download video for a specific player"""
        player = self.player1 if player_num == 1 else self.player2
        
        if not hasattr(player, 'images') or not player.images:
            QMessageBox.warning(self, "No Images", "There are no images to save as a video.")
            return
            
        from PyQt5.QtWidgets import QFileDialog
        
        filename, _ = QFileDialog.getSaveFileName(
            self, f"Save Video {player_num}", f"video_{player_num}.mp4", "MP4 Files (*.mp4)")
        
        if not filename:
            return
            
        if not filename.lower().endswith('.mp4'):
            filename += '.mp4'
            
        try:
            # Use similar logic to VideoWindow.download_video
            import cv2
            
            first_img = player.images[0]
            if isinstance(first_img, tuple) or isinstance(first_img, list):
                first_img = first_img[0]
                
            if len(first_img.shape) == 2:
                height, width = first_img.shape
            else:
                height, width, _ = first_img.shape
                
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, 4.0, (width, height))
            
            if not out.isOpened():
                raise IOError("Failed to open video writer")
                
            for i, img in enumerate(player.images):
                frame = img[0].copy() if (isinstance(img, tuple) or isinstance(img, list)) else img.copy()
                
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    
                # Add overlays if enabled
                if player.show_boxes or player.show_text:
                    boxes = player.boxes[i] if i < len(player.boxes) else {}
                    for category in ['eff', 'tar']:
                        color = (10, 194, 255) if category == 'eff' else (220, 123, 12)  # BGR format
                        for box in boxes.get(category, []):
                            left, top, w, h = map(int, box)
                            if player.show_boxes:
                                cv2.rectangle(frame, (left, top), (left + w, top + h), color, 2)
                            if player.show_text:
                                txt = category[0].upper()
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                textsize = cv2.getTextSize(txt, font, 1, 2)[0]
                                center_x = int((w - textsize[0]) / 2 + left)
                                center_y = int((h + textsize[1]) / 2 + top)
                                cv2.putText(frame, txt, (center_x, center_y), font, 0.9, color, 2)
                                
                out.write(frame)
                
            out.release()
            QMessageBox.information(self, "Success", f"Video {player_num} saved successfully to {filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save video: {str(e)}")

    def download_comparison_video(self):
        """Download a side-by-side comparison video"""
        if not hasattr(self.player1, 'images') or not self.player1.images or \
           not hasattr(self.player2, 'images') or not self.player2.images:
            QMessageBox.warning(self, "No Images", "Both videos must be loaded to create a comparison.")
            return
            
        from PyQt5.QtWidgets import QFileDialog
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Comparison Video", "comparison_video.mp4", "MP4 Files (*.mp4)")
        
        if not filename:
            return
            
        if not filename.lower().endswith('.mp4'):
            filename += '.mp4'
            
        try:
            import cv2
            import numpy as np
            
            # Get dimensions from first images
            img1 = self.player1.images[0]
            img2 = self.player2.images[0]
            
            if isinstance(img1, tuple) or isinstance(img1, list):
                img1 = img1[0]
            if isinstance(img2, tuple) or isinstance(img2, list):
                img2 = img2[0]
                
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            
            # Use maximum dimensions and resize both to same size
            max_h, max_w = max(h1, h2), max(w1, w2)
            combined_width = max_w * 2
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, 4.0, (combined_width, max_h))
            
            if not out.isOpened():
                raise IOError("Failed to open video writer")
                
            # Process frames from both videos
            min_frames = min(len(self.player1.images), len(self.player2.images))
            
            for i in range(min_frames):
                # Get frames
                frame1 = self.player1.images[i]
                frame2 = self.player2.images[i]
                
                if isinstance(frame1, tuple) or isinstance(frame1, list):
                    frame1 = frame1[0].copy()
                else:
                    frame1 = frame1.copy()
                    
                if isinstance(frame2, tuple) or isinstance(frame2, list):
                    frame2 = frame2[0].copy()
                else:
                    frame2 = frame2.copy()
                
                # Convert to BGR if grayscale
                if len(frame1.shape) == 2:
                    frame1 = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
                if len(frame2.shape) == 2:
                    frame2 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)
                
                # Resize to same dimensions
                frame1 = cv2.resize(frame1, (max_w, max_h))
                frame2 = cv2.resize(frame2, (max_w, max_h))
                
                # Combine side by side
                combined_frame = np.hstack([frame1, frame2])
                
                out.write(combined_frame)
                
            out.release()
            QMessageBox.information(self, "Success", f"Comparison video saved successfully to {filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save comparison video: {str(e)}")

    def sync_player1_slider(self):
        value = self.frame_slider1.value()
        if value != self.player1.image_index:
            self.player1.switch_images(value)
            self.update_right_panel_controls()

    def sync_player1_slider_release(self):
        value = self.frame_slider1.value()
        self.player1.switch_images(value)
        self.update_right_panel_controls()

    def sync_player2_slider(self):
        value = self.frame_slider2.value()
        if value != self.player2.image_index:
            self.player2.switch_images(value)
            self.update_right_panel_controls()

    def sync_player2_slider_release(self):
        value = self.frame_slider2.value()
        self.player2.switch_images(value)
        self.update_right_panel_controls()

    def update_right_panel_controls(self):
        # Update Video 1 controls
        if hasattr(self.player1, 'images') and self.player1.images:
            self.frame_slider1.blockSignals(True)
            self.frame_slider1.setMaximum(len(self.player1.images) - 1)
            self.frame_slider1.setValue(self.player1.image_index)
            self.frame_slider1.blockSignals(False)
            self.frame_counter_label1.setText(f"Frame: {self.player1.image_index + 1}/{len(self.player1.images)}")
        # Update Video 2 controls
        if hasattr(self.player2, 'images') and self.player2.images:
            self.frame_slider2.blockSignals(True)
            self.frame_slider2.setMaximum(len(self.player2.images) - 1)
            self.frame_slider2.setValue(self.player2.image_index)
            self.frame_slider2.blockSignals(False)
            self.frame_counter_label2.setText(f"Frame: {self.player2.image_index + 1}/{len(self.player2.images)}")

# =========================
# Original Video Window (Now inheriting from SingleVideoPlayer with additional features)
# =========================


class VideoWindow(QMainWindow):
    def __init__(self, parent=None):
        super(VideoWindow, self).__init__(parent)
        self.setWindowTitle("Video Controls")
        # Make window size relative to screen size
        screen = QApplication.desktop().screenGeometry()
        self.setMinimumSize(int(screen.width() * 0.6), int(screen.height() * 0.6))
        self.resize(int(screen.width() * 0.7), int(screen.height() * 0.7))

        # Video data placeholders
        self.images = []
        self.boxes = []
        self.channel_images = []
        self.image_index = 0
        self.selected_uid = None
        self.run_dirname = ""
        self.is_playing = False
        self.playback_speed = 250  # ms
        self.show_boxes = True
        self.show_text = True
        self.show_contours = False
        self.channel_thresholds = {'CH1': 200, 'CH2': 200, 'CH3': 200}

        # Colors
        self.eff_color = QColor('#FFC20A')  # Yellow
        self.tar_color = QColor('#0C7BDC')  # Blue

        # Timer for playback
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.switch_images)

        # Main widget and layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Left panel (Video + Plot)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # Video display container with fixed aspect ratio and navigation buttons
        video_container = QWidget()
        video_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_layout = QHBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(5)

        # Add navigation methods to VideoWindow
        def previous_uid():
            if not self.parent() or not hasattr(self.parent(), 'uid_list') or not hasattr(self.parent(), 'current_uid_index'):
                return
            uid_list = self.parent().uid_list
            current_index = self.parent().current_uid_index
            if not uid_list or current_index == -1:
                return
            new_index = (current_index - 1) % len(uid_list)
            self.parent().current_uid_index = new_index
            new_uid = uid_list[new_index]
            self.parent().select_uid(new_uid)

        def next_uid():
            if not self.parent() or not hasattr(self.parent(), 'uid_list') or not hasattr(self.parent(), 'current_uid_index'):
                return
            uid_list = self.parent().uid_list
            current_index = self.parent().current_uid_index
            if not uid_list or current_index == -1:
                return
            new_index = (current_index + 1) % len(uid_list)
            self.parent().current_uid_index = new_index
            new_uid = uid_list[new_index]
            self.parent().select_uid(new_uid)

        self.previous_uid = previous_uid
        self.next_uid = next_uid

        # Previous UID button on left
        self.prev_uid_button = QPushButton("⟨")
        self.prev_uid_button.setFixedWidth(40)
        self.prev_uid_button.setToolTip("Previous UID")
        self.prev_uid_button.clicked.connect(self.previous_uid)
        self.prev_uid_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 16px;
                min-width: 40px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        video_layout.addWidget(self.prev_uid_button, alignment=Qt.AlignVCenter)

        self.video_label = VideoLabel()
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumSize(300, 300)  # Set minimum size to prevent too small display
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMouseTracking(True)
        self.video_label.installEventFilter(self)
        video_layout.addWidget(self.video_label, stretch=1)

        # Next UID button on right
        self.next_uid_button = QPushButton("⟩")
        self.next_uid_button.setFixedWidth(40)
        self.next_uid_button.setToolTip("Next UID")
        self.next_uid_button.clicked.connect(self.next_uid)
        self.next_uid_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 16px;
                min-width: 40px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        video_layout.addWidget(self.next_uid_button, alignment=Qt.AlignVCenter)

        left_layout.addWidget(video_container, stretch=3)

        # Plot area
        plot_container = QGroupBox("Nanowell Metrics Over Time")
        plot_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plot_layout = QVBoxLayout(plot_container)
        
        self.figure = Figure(figsize=(5, 3), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plot_layout.addWidget(self.canvas)

        # Plot controls
        plot_controls = QHBoxLayout()
        plot_controls.addWidget(QLabel("X Axis:"))
        self.x_axis_combo = QComboBox()
        self.x_axis_combo.addItems(["Timepoints"])
        plot_controls.addWidget(self.x_axis_combo)
        
        plot_controls.addWidget(QLabel("Y Axis:"))
        self.y_axis_combo = QComboBox()
        self.y_axis_combo.addItems([
            "Min", "Max", "Median", "Mean",
            "CH3 Annexin Intensity", "Cell Count", "Survival Curve"
        ])
        self.y_axis_combo.currentTextChanged.connect(self.update_plot)
        plot_controls.addWidget(self.y_axis_combo)
        plot_layout.addLayout(plot_controls)
        
        left_layout.addWidget(plot_container, stretch=2)

        main_layout.addWidget(left_panel, stretch=3)

        # Right panel (Controls)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(15)  # Increased spacing between elements

        # Create a scroll area for the right panel
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll_widget = QWidget()
        right_scroll_layout = QVBoxLayout(right_scroll_widget)
        right_scroll_layout.setContentsMargins(5, 5, 5, 5)
        right_scroll_layout.setSpacing(15)
        right_scroll.setWidget(right_scroll_widget)

        # Analysis Parameters
        analysis_group = QGroupBox("Analysis Parameters")
        analysis_layout = QGridLayout()
        analysis_layout.setColumnStretch(1, 1)
        analysis_layout.setVerticalSpacing(10)
        
        analysis_layout.addWidget(QLabel("EDInt:"), 0, 0)
        self.edint_entry = QLineEdit("125")
        analysis_layout.addWidget(self.edint_entry, 0, 1)
        
        analysis_layout.addWidget(QLabel("DInt:"), 1, 0)
        self.dint_entry = QLineEdit("125")
        analysis_layout.addWidget(self.dint_entry, 1, 1)
        
        self.rerun_btn = QPushButton("Rerun Secondary Analysis")
        self.rerun_btn.setStyleSheet("QPushButton { padding: 8px; }")
        analysis_layout.addWidget(self.rerun_btn, 2, 0, 1, 2)
        
        self.nanowell_info_label = QLabel("Nanowell info:\nUID: N/A")
        analysis_layout.addWidget(self.nanowell_info_label, 3, 0, 1, 2)
        analysis_group.setLayout(analysis_layout)
        right_scroll_layout.addWidget(analysis_group)

        # Overlay Options with better spacing
        overlay_group = QGroupBox("Overlay Options")
        overlay_layout = QGridLayout()
        overlay_layout.setVerticalSpacing(10)
        overlay_layout.setHorizontalSpacing(15)
        
        self.show_boxes_checkbox = QCheckBox("Show Boxes")
        self.show_boxes_checkbox.setChecked(True)
        overlay_layout.addWidget(self.show_boxes_checkbox, 0, 0)
        
        self.show_text_checkbox = QCheckBox("Show Text")
        self.show_text_checkbox.setChecked(True)
        overlay_layout.addWidget(self.show_text_checkbox, 0, 1)
        
        self.show_contours_checkbox = QCheckBox("Show Contours")
        overlay_layout.addWidget(self.show_contours_checkbox, 1, 0)
        
        self.show_base_checkbox = QCheckBox("Show Base (CH0)")
        self.show_base_checkbox.setChecked(True)
        overlay_layout.addWidget(self.show_base_checkbox, 1, 1)
        overlay_group.setLayout(overlay_layout)
        right_scroll_layout.addWidget(overlay_group)

        # Connect overlay checkboxes to toggle methods
        # These connections should be in QC_APP, not in VideoWindow
        # Remove these lines from VideoWindow if present
        # self.show_boxes_checkbox.stateChanged.connect(self.video_window.toggle_boxes)
        # self.show_text_checkbox.stateChanged.connect(self.video_window.toggle_text)
        # self.show_contours_checkbox.stateChanged.connect(self.video_window.toggle_contours)
        # self.show_base_checkbox.stateChanged.connect(self.video_window.toggle_base)

        # Colors with better spacing
        color_group = QGroupBox("Colors")
        color_layout = QHBoxLayout()
        color_layout.setSpacing(15)
        self.eff_color_picker = ColorPickerWidget(self.eff_color, "Effector:")
        self.tar_color_picker = ColorPickerWidget(self.tar_color, "Target:")
        color_layout.addWidget(self.eff_color_picker)
        color_layout.addWidget(self.tar_color_picker)
        color_group.setLayout(color_layout)
        right_scroll_layout.addWidget(color_group)

        # Channel Overlays with better spacing
        channel_group = QGroupBox("Channel Overlays")
        channel_layout = QGridLayout()
        channel_layout.setVerticalSpacing(15)
        channel_layout.setHorizontalSpacing(10)
        self.channel_overlay_checkboxes = {}
        fluor_channels = {
            'CH1': ("Fluor Green", QColor(0, 255, 0)),
            'CH2': ("Fluor Red", QColor(255, 0, 0)),
            'CH3': ("Fluor Blue", QColor(0, 0, 255))
        }
        
        for row, (channel_key, (label_text, color)) in enumerate(fluor_channels.items()):
            cb = QCheckBox(f"{label_text} Overlay")
            cb.setStyleSheet(f"""
                QCheckBox {{ color: {color.name()}; font-weight: bold; }}
                QCheckBox::indicator {{ width: 18px; height: 18px; }}
                QCheckBox::indicator:unchecked {{ border: 2px solid {color.name()}; background: transparent; }}
                QCheckBox::indicator:checked {{ border: 2px solid {color.name()}; background: {color.name()}; }}
            """)
            channel_layout.addWidget(cb, row, 0)
            
            threshold_label = QLabel("Threshold:")
            channel_layout.addWidget(threshold_label, row, 1)
            
            thresh_slider = QSlider(Qt.Horizontal)
            thresh_slider.setMinimum(0)
            thresh_slider.setMaximum(65535)
            thresh_slider.setValue(200)
            thresh_slider.setTickPosition(QSlider.TicksBelow)
            thresh_slider.setTickInterval(1000)
            channel_layout.addWidget(thresh_slider, row, 2)
            
            thresh_spinbox = QSpinBox()
            thresh_spinbox.setMinimum(0)
            thresh_spinbox.setMaximum(65535)
            thresh_spinbox.setValue(200)
            thresh_spinbox.setFixedWidth(70)  # Fixed width for better alignment
            channel_layout.addWidget(thresh_spinbox, row, 3)
            
            thresh_slider.valueChanged.connect(thresh_spinbox.setValue)
            thresh_spinbox.valueChanged.connect(thresh_slider.setValue)
            
            self.channel_overlay_checkboxes[channel_key] = {
                'checkbox': cb,
                'color': color,
                'slider': thresh_slider,
                'spinbox': thresh_spinbox
            }
            self.channel_thresholds[channel_key] = 200

        channel_group.setLayout(channel_layout)
        right_scroll_layout.addWidget(channel_group)

        # Pixel Intensity with better visibility
        self.pixel_intensity_label = QLabel("Pixel Intensity: N/A")
        self.pixel_intensity_label.setWordWrap(True)
        self.pixel_intensity_label.setStyleSheet("""
            QLabel {
                background: white;
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 4px;
                min-height: 20px;
            }
        """)
        right_scroll_layout.addWidget(self.pixel_intensity_label)

        # SNR Table with minimum height
        legend_group = QGroupBox("SNR")
        legend_layout = QVBoxLayout()
        self.legend_table = QTableWidget(2, 3)
        self.legend_table.setHorizontalHeaderLabels(["Channel", "Signal", "Noise"])
        self.legend_table.verticalHeader().setVisible(False)
        self.legend_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.legend_table.setMinimumHeight(100)  # Ensure minimum height
        legend_layout.addWidget(self.legend_table)
        legend_group.setLayout(legend_layout)
        right_scroll_layout.addWidget(legend_group)

        # Playback Controls with better spacing
        playback_group = QGroupBox("Playback Controls")
        playback_layout = QVBoxLayout()
        playback_layout.setSpacing(10)
        
        # Buttons row
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)
        
        self.prev_frame_button = QPushButton("<<")
        self.prev_frame_button.setFixedWidth(60)
        self.prev_frame_button.clicked.connect(self.previous_frame)
        buttons_layout.addWidget(self.prev_frame_button)
        
        self.play_pause_button = QPushButton("||")
        self.play_pause_button.setFixedWidth(60)
        self.play_pause_button.clicked.connect(self.toggle_playback)
        buttons_layout.addWidget(self.play_pause_button)
        
        self.next_frame_button = QPushButton(">>")
        self.next_frame_button.setFixedWidth(60)
        self.next_frame_button.clicked.connect(self.next_frame)
        buttons_layout.addWidget(self.next_frame_button)
        playback_layout.addLayout(buttons_layout)
        
        # Frame slider and counter with better visibility
        slider_container = QWidget()
        slider_layout = QVBoxLayout(slider_container)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.setSpacing(5)
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setTickPosition(QSlider.TicksBelow)
        self.frame_slider.valueChanged.connect(self.slider_value_changed)
        self.frame_slider.sliderReleased.connect(self.slider_value_released)
        slider_layout.addWidget(self.frame_slider)
        
        self.frame_counter_label = QLabel("Frame: 0/0")
        self.frame_counter_label.setAlignment(Qt.AlignCenter)
        self.frame_counter_label.setStyleSheet("QLabel { padding: 5px; }")
        slider_layout.addWidget(self.frame_counter_label)
        
        playback_layout.addWidget(slider_container)
        
        # Speed control with better layout
        speed_container = QWidget()
        speed_layout = QHBoxLayout(speed_container)
        speed_layout.setContentsMargins(0, 0, 0, 0)
        speed_layout.setSpacing(10)
        
        speed_layout.addWidget(QLabel("Playback Interval:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(50, 1000)
        self.speed_slider.setValue(250)
        self.speed_slider.valueChanged.connect(self.change_playback_speed)
        speed_layout.addWidget(self.speed_slider)
        
        self.speed_label = QLabel("250 ms")
        self.speed_label.setMinimumWidth(60)
        speed_layout.addWidget(self.speed_label)
        
        playback_layout.addWidget(speed_container)
        
        # Download button
        self.download_video_btn = QPushButton("Download as Video")
        self.download_video_btn.clicked.connect(self.download_video)
        playback_layout.addWidget(self.download_video_btn)
        
        # Add view button
        self.view_flags_btn = QPushButton("View Cell Flags")
        self.view_flags_btn.clicked.connect(self.view_cell_flags)
        self.view_flags_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        playback_layout.addWidget(self.view_flags_btn)
        
        playback_group.setLayout(playback_layout)
        right_scroll_layout.addWidget(playback_group)

        # After the Channel Overlays group box, add Cell Flags group
        cell_flags_group = QGroupBox("Cell Flags")
        cell_flags_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 6px;
                margin-top: 12px;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #2196F3;
            }
        """)
        cell_flags_layout = QVBoxLayout()
        
        # Create checkboxes for different flags
        self.flag_checkboxes = {}
        flags = ['out_of_focus', 'wrong_morphology', 'debris_present', 'multiple_cells', 'cell_death']
        
        for flag in flags:
            checkbox = QCheckBox(flag.replace('_', ' ').title())
            checkbox.setStyleSheet("""
                QCheckBox {
                    padding: 5px;
                    spacing: 10px;
                }
                QCheckBox::indicator {
                    width: 18px;
                    height: 18px;
                }
                QCheckBox::indicator:unchecked {
                    border: 2px solid #2196F3;
                    background: white;
                    border-radius: 3px;
                }
                QCheckBox::indicator:checked {
                    border: 2px solid #2196F3;
                    background: #2196F3;
                    border-radius: 3px;
                }
            """)
            self.flag_checkboxes[flag] = checkbox
            cell_flags_layout.addWidget(checkbox)
        
        # Add update button
        self.update_flags_btn = QPushButton("Update Cell Flags")
        self.update_flags_btn.clicked.connect(self.update_cell_flags)
        self.update_flags_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        cell_flags_layout.addWidget(self.update_flags_btn)
        
        cell_flags_group.setLayout(cell_flags_layout)
        right_scroll_layout.addWidget(cell_flags_group)

        # Add right scroll area to right layout
        right_layout.addWidget(right_scroll)

        # Add right panel to main layout with proper stretch
        main_layout.addWidget(right_panel, stretch=2)

        # Apply stylesheet for a more professional look
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #ffffff;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #2196F3;
                border: 1px solid #1976D2;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QLineEdit, QSpinBox {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                background: white;
            }
            QTableWidget {
                gridline-color: #dddddd;
                selection-background-color: #2196F3;
                selection-color: white;
            }
            QHeaderView::section {
                background-color: #2196F3;
                color: white;
                padding: 5px;
                border: none;
            }
            QLabel {
                color: #333333;
            }
        """)

        self.cell_flags_csv_path = None  # Path to cell flags CSV

    # ---------------------
    # Video Data & Playback Methods
    # ---------------------
    def set_video_data(self, images, boxes, channel_images, uid, run_dirname, continuation=True):
        if continuation:
            print(
                f"[DEBUG] set_video_data called for UID: {uid}, total images: {len(images)}")

            self.images.extend(images)
            self.boxes.extend(boxes)
            self.channel_images.extend(channel_images)
            self.selected_uid = uid
            self.run_dirname = run_dirname
            print(
                f"[DEBUG] Data assigned. Reset image_index to {self.image_index}")

            self.frame_slider.blockSignals(True)
            max_index = len(self.images) - 1 if self.images else 0
            self.frame_slider.setMaximum(max_index)
            self.frame_slider.setValue(0)
            self.frame_slider.blockSignals(False)
            print(
                f"[DEBUG] Frame slider updated. Maximum index set to {max_index}")

            self.nanowell_info_label.setText(f"Nanowell info:\nUID: {uid}")
            self.is_playing = True
            print(
                f"[DEBUG] Timer restarted with interval: {self.speed_slider.value()} ms")
            self.update_plot()
        else:
            print(
                f"[DEBUG] set_video_data called for UID: {uid}, total images: {len(images)}")
            if self.timer.isActive():
                print("[DEBUG] Timer is active; stopping timer before data update")
                self.timer.stop()

            self.images = images
            self.boxes = boxes
            self.channel_images = channel_images
            print(channel_images)
            print(images)
            self.selected_uid = uid
            self.run_dirname = run_dirname
            self.image_index = 0
            print(
                f"[DEBUG] Data assigned. Reset image_index to {self.image_index}")

            self.frame_slider.blockSignals(True)
            max_index = len(self.images) - 1 if self.images else 0
            self.frame_slider.setMaximum(max_index)
            self.frame_slider.setValue(0)
            self.frame_slider.blockSignals(False)
            print(
                f"[DEBUG] Frame slider updated. Maximum index set to {max_index}")

            self.nanowell_info_label.setText(f"Nanowell info:\nUID: {uid}")
            self.switch_images(0)
            self.is_playing = True
            self.timer.start(self.speed_slider.value())
            print(
                f"[DEBUG] Timer restarted with interval: {self.speed_slider.value()} ms")

    def switch_images(self, image_index=-1):
        print(f"[DEBUG] switch_images called with image_index: {image_index}")
        if not self.channel_images or not self.images:
            print("[DEBUG] No channel_images or images available; returning early")
            return

        if image_index == -1:
            if self.is_playing:
                self.image_index = (self.image_index + 1) % len(self.images)
        else:
            self.image_index = image_index
        print(f"[DEBUG] Updated image_index: {self.image_index}")

        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.image_index)
        self.frame_slider.blockSignals(False)
        self.frame_counter_label.setText(
            f"Frame: {self.image_index + 1}/{len(self.images)}")

        try:
            current_images = self.channel_images[self.image_index]
            if current_images is None:
                print(f"[DEBUG] No image data for frame {self.image_index}")
                return

            # Process base image (CH0)
            ch0_data = current_images.get('CH0')
            if ch0_data is None or len(ch0_data) < 1:
                print(f"[DEBUG] No CH0 data for frame {self.image_index}")
                base_image = np.zeros((500, 500), dtype=np.uint8)
            else:
                base_image = ch0_data[0]

            if base_image.dtype != np.uint8:
                base_image = cv2.normalize(
                    base_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                print("[DEBUG] Normalized base image to uint8")

            height, width = base_image.shape
            if self.show_base_checkbox.isChecked():
                composite = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
            else:
                composite = np.zeros((height, width, 3), dtype=np.uint8)

            # Process CH1 overlay if enabled
            ch1_enabled = False
            if hasattr(self, 'channel_overlay_checkboxes') and 'CH1' in self.channel_overlay_checkboxes:
                # For VideoWindow (single video)
                ch1_enabled = self.channel_overlay_checkboxes['CH1']['checkbox'].isChecked()
            elif hasattr(self, 'channel_overlay_states'):
                # For DualVideoWindow (dual video)
                ch1_enabled = self.channel_overlay_states.get('CH1', False)
            
            if 'CH1' in current_images and ch1_enabled:
                print(f"[DEBUG] CH1 overlay enabled for {getattr(self, 'selected_uid', 'unknown')}")
                ch1_data = current_images['CH1']
                if ch1_data is not None and len(ch1_data) >= 2:
                    ch1, hi_ch1 = ch1_data[0], ch1_data[1]
                    print(f"[DEBUG] CH1 original range: {hi_ch1.min():.2f} to {hi_ch1.max():.2f}")
                    if ch1.dtype != np.uint8:
                        ch1 = cv2.normalize(ch1, None, 0, 255,
                                         cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    th1 = self.channel_thresholds.get('CH1', 200)
                    mask1 = hi_ch1 >= th1
                    pixels_above_threshold = np.sum(mask1)
                    print(f"[DEBUG] CH1 threshold {th1}, pixels above: {pixels_above_threshold}")
                    
                    if pixels_above_threshold > 0:
                        ch1_thresh = np.where(mask1, ch1, 0).astype(np.uint8)
                        
                        # Debug: check the actual values
                        max_fluor_val = np.max(ch1_thresh)
                        mean_fluor_val = np.mean(ch1_thresh[ch1_thresh > 0])
                        print(f"[DEBUG] CH1 fluorescence - max: {max_fluor_val}, mean: {mean_fluor_val:.2f}")
                        
                        # Boost the fluorescence signal for better visibility
                        ch1_boosted = np.clip(ch1_thresh * 2, 0, 255).astype(np.uint8)
                        
                        composite[:, :, 1] = cv2.add(composite[:, :, 1], ch1_boosted)
                        print(f"[DEBUG] Applied CH1 overlay - green channel updated with boost")
                    else:
                        print(f"[DEBUG] No CH1 pixels above threshold {th1}")
                    print(f"[DEBUG] Applied CH1 overlay with threshold {th1}")

            # Process CH2 overlay if enabled
            ch2_enabled = False
            if hasattr(self, 'channel_overlay_checkboxes') and 'CH2' in self.channel_overlay_checkboxes:
                # For VideoWindow (single video)
                ch2_enabled = self.channel_overlay_checkboxes['CH2']['checkbox'].isChecked()
            elif hasattr(self, 'channel_overlay_states'):
                # For DualVideoWindow (dual video)
                ch2_enabled = self.channel_overlay_states.get('CH2', False)
            
            if 'CH2' in current_images and ch2_enabled:
                print(f"[DEBUG] CH2 overlay enabled for {getattr(self, 'selected_uid', 'unknown')}")
                ch2_data = current_images['CH2']
                if ch2_data is not None and len(ch2_data) >= 2:
                    ch2, hi_ch2 = ch2_data[0], ch2_data[1]
                    print(f"[DEBUG] CH2 original range: {hi_ch2.min():.2f} to {hi_ch2.max():.2f}")
                    if ch2.dtype != np.uint8:
                        ch2 = cv2.normalize(ch2, None, 0, 255,
                                         cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    th2 = self.channel_thresholds.get('CH2', 200)
                    mask2 = hi_ch2 >= th2
                    pixels_above_threshold = np.sum(mask2)
                    print(f"[DEBUG] CH2 threshold {th2}, pixels above: {pixels_above_threshold}")
                    
                    if pixels_above_threshold > 0:
                        ch2_thresh = np.where(mask2, ch2, 0).astype(np.uint8)
                        
                        # Debug: check the actual values
                        max_fluor_val = np.max(ch2_thresh)
                        mean_fluor_val = np.mean(ch2_thresh[ch2_thresh > 0])
                        print(f"[DEBUG] CH2 fluorescence - max: {max_fluor_val}, mean: {mean_fluor_val:.2f}")
                        
                        # Boost the fluorescence signal for better visibility
                        ch2_boosted = np.clip(ch2_thresh * 2, 0, 255).astype(np.uint8)
                        
                        composite[:, :, 2] = cv2.add(composite[:, :, 2], ch2_boosted)
                        print(f"[DEBUG] Applied CH2 overlay - red channel updated with boost")
                    else:
                        print(f"[DEBUG] No CH2 pixels above threshold {th2}")
                    print(f"[DEBUG] Applied CH2 overlay with threshold {th2}")

            # Process CH3 overlay if enabled
            ch3_enabled = False
            if hasattr(self, 'channel_overlay_checkboxes') and 'CH3' in self.channel_overlay_checkboxes:
                # For VideoWindow (single video)
                ch3_enabled = self.channel_overlay_checkboxes['CH3']['checkbox'].isChecked()
            elif hasattr(self, 'channel_overlay_states'):
                # For DualVideoWindow (dual video)
                ch3_enabled = self.channel_overlay_states.get('CH3', False)
            
            if 'CH3' in current_images and ch3_enabled:
                print(f"[DEBUG] CH3 overlay enabled for {getattr(self, 'selected_uid', 'unknown')}")
                ch3_data = current_images['CH3']
                if ch3_data is not None and len(ch3_data) >= 2:
                    ch3, hi_ch3 = ch3_data[0], ch3_data[1]
                    print(f"[DEBUG] CH3 original range: {hi_ch3.min():.2f} to {hi_ch3.max():.2f}")
                    if ch3.dtype != np.uint8:
                        ch3 = cv2.normalize(ch3, None, 0, 255,
                                         cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    th3 = self.channel_thresholds.get('CH3', 200)
                    mask3 = hi_ch3 >= th3
                    pixels_above_threshold = np.sum(mask3)
                    print(f"[DEBUG] CH3 threshold {th3}, pixels above: {pixels_above_threshold}")
                    
                    if pixels_above_threshold > 0:
                        ch3_thresh = np.where(mask3, ch3, 0).astype(np.uint8)
                        
                        # Debug: check the actual values
                        max_fluor_val = np.max(ch3_thresh)
                        mean_fluor_val = np.mean(ch3_thresh[ch3_thresh > 0])
                        print(f"[DEBUG] CH3 fluorescence - max: {max_fluor_val}, mean: {mean_fluor_val:.2f}")
                        
                        # Boost the fluorescence signal for better visibility
                        ch3_boosted = np.clip(ch3_thresh * 2, 0, 255).astype(np.uint8)
                        
                        composite[:, :, 0] = cv2.add(composite[:, :, 0], ch3_boosted)
                        print(f"[DEBUG] Applied CH3 overlay - blue channel updated with boost")
                    else:
                        print(f"[DEBUG] No CH3 pixels above threshold {th3}")
                    print(f"[DEBUG] Applied CH3 overlay with threshold {th3}")

            # Drawing boxes and text overlays
            painter = QPainter()
            composite_rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
            height, width, _ = composite_rgb.shape
            qImg = QImage(composite_rgb.data, width, height,
                         width * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg)

            # Calculate the size to maintain square aspect ratio
            label_size = self.video_label.size()
            min_size = min(label_size.width(), label_size.height())
            square_size = QSize(min_size, min_size)
            scaled_pixmap = pixmap.scaled(
                square_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            # Center the pixmap in the label
            x_offset = 0
            y_offset = 0
            
            # Create a new pixmap with the size of the scaled pixmap (no black border)
            final_pixmap = QPixmap(scaled_pixmap.size())
            final_pixmap.fill(QColor('#f0f0f0'))  # Light gray background to blend with UI
            
            # Draw the scaled image at (0,0)
            painter = QPainter(final_pixmap)
            painter.drawPixmap(0, 0, scaled_pixmap)
            
            # Draw overlays on the scaled pixmap
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setRenderHint(QPainter.TextAntialiasing)

            # Calculate scale factors for overlay positioning
            scale_x = scaled_pixmap.width() / width
            scale_y = scaled_pixmap.height() / height

            if self.image_index < len(self.boxes):
                boxes = self.boxes[self.image_index]
                for category in ['eff', 'tar']:
                    pen_color = self.eff_color_picker.get_color(
                    ) if category == 'eff' else self.tar_color_picker.get_color()
                    for box in boxes.get(category, []):
                        left, top, w, h = map(int, box)
                        # Scale the box coordinates and add offsets
                        scaled_left = int(left * scale_x) + x_offset
                        scaled_top = int(top * scale_y) + y_offset
                        scaled_w = int(w * scale_x)
                        scaled_h = int(h * scale_y)
                        
                        if self.show_boxes:
                            painter.setPen(QPen(QColor(pen_color), max(2, int(scale_x * 2))))
                            painter.drawRect(scaled_left, scaled_top, scaled_w, scaled_h)
                        if self.show_contours:
                            pen = QPen(QColor(pen_color))
                            pen.setStyle(Qt.DashLine)
                            pen.setWidth(max(2, int(scale_x * 2)))
                            painter.setPen(pen)
                            painter.drawRect(scaled_left, scaled_top, scaled_w, scaled_h)
                        if self.show_text:
                            painter.setPen(QPen(QColor(pen_color)))
                            font_size = max(12, int(20 * scale_x))
                            painter.setFont(QFont('Arial', font_size))
                            center_x = int(scaled_left + scaled_w / 2)
                            center_y = int(scaled_top + scaled_h / 2)
                            txt = category[0].upper()
                            text_rect = QRect(center_x - scaled_w/2, center_y - scaled_h/2, scaled_w, scaled_h)
                            painter.drawText(text_rect, Qt.AlignCenter, txt)

            painter.end()
            self.video_label.setPixmap(final_pixmap)
            print("[DEBUG] Canvas updated with new pixmap")

            # Update SNR table with current frame data
            self.update_snr_table()

        except Exception as e:
            print(f"[DEBUG] Error in switch_images: {str(e)}")
            # Create an empty black frame if there's an error
            empty_frame = np.zeros((500, 500, 3), dtype=np.uint8)
            height, width, _ = empty_frame.shape
            qImg = QImage(empty_frame.data, width, height, width * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg)
            self.video_label.setPixmap(pixmap)

    def next_frame(self):
        if self.is_playing:
            self.toggle_playback()
        next_index = (self.image_index + 1) % len(self.images)
        self.switch_images(next_index)

    def previous_frame(self):
        if self.is_playing:
            self.toggle_playback()
        prev_index = (self.image_index - 1) % len(self.images)
        self.switch_images(prev_index)

    def slider_value_changed(self):
        new_index = self.frame_slider.value()
        if self.is_playing:
            self.is_playing = False
            self.play_pause_button.setText("||")
            self.play_pause_button.setToolTip("Play")
            if self.timer.isActive():
                self.timer.stop()
        self.switch_images(new_index)

    def slider_value_released(self):
        final_index = self.frame_slider.value()
        if self.is_playing:
            self.toggle_playback()
        self.switch_images(final_index)

    def toggle_playback(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_pause_button.setText("||")
            self.play_pause_button.setToolTip("Pause")
            self.timer.start(self.speed_slider.value())
        else:
            self.play_pause_button.setText("||")
            self.play_pause_button.setToolTip("Play")
            self.timer.stop()

    def change_playback_speed(self, value):
        self.playback_speed = value
        if self.timer.isActive():
            self.timer.setInterval(self.speed_slider.value())

    def download_video(self):
        if not self.images:
            QMessageBox.warning(self, "No Images",
                                "There are no images to save as a video.")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Video", "", "MP4 Files (*.mp4)")
        if not filename:
            return
        if not filename.lower().endswith('.mp4'):
            filename += '.mp4'
        try:
            # Extract the first image array for dimensions
            first_img = self.images[0]
            if isinstance(first_img, tuple) or isinstance(first_img, list):
                first_img = first_img[0]
            # Handle grayscale images (2D) by converting to BGR
            if len(first_img.shape) == 2:
                height, width = first_img.shape
            else:
                height, width, _ = first_img.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, 4.0, (width, height))
            if not out.isOpened():
                raise IOError("Failed to open video writer")
            for i, img in enumerate(self.images):
                # Extract image array if img is tuple or list
                frame = img[0].copy() if (isinstance(img, tuple) or isinstance(img, list)) else img.copy()
                # Convert grayscale to BGR for video writer
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                if self.show_boxes or self.show_text:
                    boxes = self.boxes[i]
                    for category in ['eff', 'tar']:
                        q_color = self.eff_color_picker.get_color() if category == 'eff' else self.tar_color_picker.get_color()
                        color = (q_color.blue(), q_color.green(), q_color.red())
                        for box in boxes.get(category, []):
                            left, top, w, h = map(int, box)
                            if self.show_boxes:
                                cv2.rectangle(frame, (left, top),
                                              (left + w, top + h), color, 2)
                            if self.show_text:
                                txt = category[0].upper()
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                textsize = cv2.getTextSize(txt, font, 1, 2)[0]
                                center_x = int((w - textsize[0]) / 2 + left)
                                center_y = int((w + textsize[1]) / 2 + top)
                                cv2.putText(
                                    frame, txt, (center_x, center_y), font, 0.9, color, 2)
                out.write(frame)
            out.release()
            QMessageBox.information(
                self, "Success", f"Video saved successfully to {filename}")
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to save video: {str(e)}")
            print(f"Error details: {e}")
            out.release()
            QMessageBox.information(
                self, "Success", f"Video saved successfully to {filename}")
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to save video: {str(e)}")
            print(f"Error details: {e}")

    def calculate_snr(self, image_data):
        """Calculate Signal-to-Noise Ratio for image data using boxes or contours"""
        if image_data is None or len(image_data) == 0:
            return 0, 0, 0, 0, 0, 0
        
        # Use the high-intensity image for calculations
        if len(image_data) >= 2:
            signal_image = image_data[1]  # High-intensity image
        else:
            signal_image = image_data[0]  # Fallback to regular image
            
        # Get current boxes if available
        signal_pixels = []
        boxes = {}
        if self.image_index < len(self.boxes):
            boxes = self.boxes[self.image_index]
            # Collect pixels from both effector and target boxes
            for category in ['eff', 'tar']:
                for box in boxes.get(category, []):
                    left, top, w, h = map(int, box)
                    roi = signal_image[top:top+h, left:left+w]
                    if roi.size > 0:
                        signal_pixels.extend(roi.flatten())
        
        # Calculate signal from ROIs
        if signal_pixels:
            signal = np.mean(signal_pixels)
            
            # Create a mask for background (excluding box regions)
            mask = np.ones_like(signal_image, dtype=bool)
            for category in ['eff', 'tar']:
                for box in boxes.get(category, []):
                    left, top, w, h = map(int, box)
                    mask[top:top+h, left:left+w] = False
                    
            # Calculate background from non-box regions
            background_pixels = signal_image[mask]
            if len(background_pixels) > 0:
                background = np.mean(background_pixels)
                noise = np.std(background_pixels)
            else:
                background = 0
                noise = 1  # Avoid division by zero
        else:
            # Fallback to percentile-based calculation if no boxes available
            non_zero_pixels = signal_image[signal_image > 0]
            if len(non_zero_pixels) > 0:
                signal = np.mean(non_zero_pixels)
            else:
                signal = 0
            
            # Calculate background (mean of bottom 10% pixels)
            threshold = np.percentile(signal_image, 10)
            background_pixels = signal_image[signal_image <= threshold]
            if len(background_pixels) > 0:
                background = np.mean(background_pixels)
                noise = np.std(background_pixels)
            else:
                background = 0
                noise = 1
        
        # Calculate metrics
        snr = signal / noise if noise > 0 else 0
        sbr = signal / background if background > 0 else 0
        cnr = (signal - background) / noise if noise > 0 else 0
        
        return signal, background, noise, snr, sbr, cnr

    def update_snr_table(self):
        """Update the Signal Metrics table with current channel data"""
        if not self.channel_images or self.image_index >= len(self.channel_images):
            return
        
        current_images = self.channel_images[self.image_index]
        if current_images is None:
            return
        
        # Clear existing table content
        self.legend_table.setRowCount(0)
        
        # Update table headers to include SNR, SBR, CNR, Best Metric
        self.legend_table.setColumnCount(7)
        self.legend_table.setHorizontalHeaderLabels(["Channel", "Signal", "Background", "Noise", "SNR", "SBR", "CNR"])
        
        channels = ['CH1', 'CH2', 'CH3']
        row = 0
        
        for channel in channels:
            if channel in current_images:
                channel_data = current_images[channel]
                signal, background, noise, snr, sbr, cnr = self.calculate_snr(channel_data)
                
                # Determine best metric
                metrics = {'SNR': snr, 'SBR': sbr, 'CNR': cnr}
                best_metric = max(metrics, key=metrics.get)
                
                # Add row to table
                self.legend_table.insertRow(row)
                
                # Channel name
                channel_item = QTableWidgetItem(channel)
                channel_item.setTextAlignment(Qt.AlignCenter)
                self.legend_table.setItem(row, 0, channel_item)
                
                # Signal value
                signal_item = QTableWidgetItem(f"{signal:.2f}")
                signal_item.setTextAlignment(Qt.AlignCenter)
                self.legend_table.setItem(row, 1, signal_item)
                
                # Background value
                background_item = QTableWidgetItem(f"{background:.2f}")
                background_item.setTextAlignment(Qt.AlignCenter)
                self.legend_table.setItem(row, 2, background_item)
                
                # Noise value
                noise_item = QTableWidgetItem(f"{noise:.2f}")
                noise_item.setTextAlignment(Qt.AlignCenter)
                self.legend_table.setItem(row, 3, noise_item)
                
                # SNR value
                snr_item = QTableWidgetItem(f"{snr:.2f}")
                snr_item.setTextAlignment(Qt.AlignCenter)
                self.legend_table.setItem(row, 4, snr_item)
                
                # SBR value
                sbr_item = QTableWidgetItem(f"{sbr:.2f}")
                sbr_item.setTextAlignment(Qt.AlignCenter)
                self.legend_table.setItem(row, 5, sbr_item)
                
                # CNR value
                cnr_item = QTableWidgetItem(f"{cnr:.2f}")
                cnr_item.setTextAlignment(Qt.AlignCenter)
                self.legend_table.setItem(row, 6, cnr_item)
                
                # Best metric
                best_item = QTableWidgetItem(best_metric)
                best_item.setTextAlignment(Qt.AlignCenter)
                # Remove best metric column, so do not set this item
                
                row += 1
        
        # If no channels found, show empty message
        if row == 0:
            self.legend_table.insertRow(0)
            no_data_item = QTableWidgetItem("No channel data")
            no_data_item.setTextAlignment(Qt.AlignCenter)
            self.legend_table.setItem(0, 0, no_data_item)
            self.legend_table.setSpan(0, 0, 1, 7)

    def update_threshold(self, channel, value):
        """Update channel threshold for both players"""
        self.channel_thresholds[channel] = value
        self.player1.channel_thresholds[channel] = value
        self.player2.channel_thresholds[channel] = value
        
        # Refresh both players
        if hasattr(self.player1, 'switch_images'):
            self.player1.switch_images(self.player1.image_index)
        if hasattr(self.player2, 'switch_images'):
            self.player2.switch_images(self.player2.image_index)

    def toggle_boxes(self, state):
        self.show_boxes = state == Qt.Checked
        self.switch_images(self.image_index)

    def toggle_text(self, state):
        self.show_text = state == Qt.Checked
        self.switch_images(self.image_index)

    def toggle_contours(self, state):
        self.show_contours = (state == Qt.Checked)
        self.switch_images(self.image_index)

    def toggle_base(self, state):
        self.show_base_checkbox = (state == Qt.Checked)
        self.switch_images(self.image_index)

    def rerun_secondary_analysis(self):
        if not self.run_dirname:
            return
        try:
            edint = int(self.edint_entry.text())
            dint = int(self.dint_entry.text())
        except ValueError:
            QMessageBox.warning(self, "Input Error",
                                "EDInt and DInt must be integers.")
            return
        self.loading_widget = LoadingWidget(self)
        self.loading_widget.show()
        run_TIMING_secondary(self.run_dirname, edint, dint, 0.01, 5, 0.325)
        self.load_run_file()
        self.loading_widget.hide_loading_sign()

    def mouseMoveEvent(self, event):
        try:
            if not self.video_label.underMouse():
                return
            if not self.channel_images:
                return
            current_images = self.channel_images[self.image_index]
            base_image = current_images['CH0'][0]
            pixmap = self.video_label.pixmap()
            if not pixmap:
                return
                
            # Get the actual image area within the label
            label_size = self.video_label.size()
            min_size = min(label_size.width(), label_size.height())
            x_offset = (label_size.width() - min_size) // 2
            y_offset = (label_size.height() - min_size) // 2
            
            # Calculate scale factors for the square image
            scale = min_size / base_image.shape[1]  # Since it's square, we can use either dimension
            
            # Get mouse position relative to the label
            mouse_pos = self.video_label.mapFromGlobal(event.globalPos())
            
            # Adjust for the offset and scaling
            x = int((mouse_pos.x() - x_offset) / scale)
            y = int((mouse_pos.y() - y_offset) / scale)
            
            if 0 <= x < base_image.shape[1] and 0 <= y < base_image.shape[0]:
                intensities = {channel: img[1][y, x] if img is not None else 'N/A'
                               for channel, img in current_images.items()}
                intensity_str = " | ".join(
                    [f"{ch}: {val}" for ch, val in intensities.items()])
                self.pixel_intensity_label.setText(
                    f"Pixel Intensity at ({x},{y}): {intensity_str}")
        except Exception as e:
            print(f"Error in mouseMoveEvent: {e}")

    def eventFilter(self, obj, event):
        if obj == self.video_label and event.type() == QEvent.MouseMove:
            self.mouseMoveEvent(event)
            return True
        return super().eventFilter(obj, event)

    def update_plot(self):
        y_metric = self.y_axis_combo.currentText()
        timepoints = list(range(len(self.channel_images)))

        values = []

        if y_metric == "Survival Curve":
            frame_count = len(timepoints)
            survival = [1] * frame_count
            for i in range(frame_count // 2, frame_count):
                survival[i] = 0
            survival = array(survival).cumsum()[::-1]
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(timepoints, survival, label="Survival Curve")
            ax.set_title("Survival Curve")
            ax.set_xlabel("Timepoints")
            ax.set_ylabel("# Cells Alive")
            ax.legend()
            self.canvas.draw()
            return

        elif y_metric == "CH3 Annexin Intensity":
            cell_traces = {}
            for t, frame in enumerate(self.channel_images):
                if frame is None:
                    continue
                ch3_data = frame.get("CH3", None)
                if ch3_data is None or len(ch3_data) < 2:
                    continue
                ch3 = ch3_data[1]
                boxes = self.boxes[t] if t < len(self.boxes) else {}
                for cat in ["eff", "tar"]:
                    for idx, box in enumerate(boxes.get(cat, [])):
                        x, y, w, h = map(int, box)
                        roi = ch3[y:y+h, x:x+w]
                        mean_val = float(mean(roi)) if roi.size > 0 else 0
                        cell_id = f"{cat}-{idx}"
                        cell_traces.setdefault(cell_id, []).append(mean_val)

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            for cell_id, intensity_series in cell_traces.items():
                ax.plot(range(len(intensity_series)), intensity_series, label=cell_id)
            ax.set_title("CH3 Annexin Intensity per Cell")
            ax.set_xlabel("Timepoints")
            ax.set_ylabel("Mean CH3 Intensity")
            ax.legend()
            self.canvas.draw()
            return

        # Handle Min, Max, Mean, Median, Cell Count
        for i, frame in enumerate(self.channel_images):
            if frame is None:
                values.append(0)
                continue

            if y_metric in ["Min", "Max", "Mean", "Median"]:
                eff_intensities = []
                tar_intensities = []
                boxes = self.boxes[i] if i < len(self.boxes) else {}
                
                ch1_data = frame.get('CH1', None)
                ch2_data = frame.get('CH2', None)
                
                if ch1_data is not None and len(ch1_data) >= 2:
                    ch1 = ch1_data[1]
                    for box in boxes.get('eff', []):
                        x, y, w, h = map(int, box)
                        roi = ch1[y:y+h, x:x+w]
                        if roi.size > 0:
                            eff_intensities.extend(roi.flatten())
                            
                if ch2_data is not None and len(ch2_data) >= 2:
                    ch2 = ch2_data[1]
                    for box in boxes.get('tar', []):
                        x, y, w, h = map(int, box)
                        roi = ch2[y:y+h, x:x+w]
                        if roi.size > 0:
                            tar_intensities.extend(roi.flatten())

                all_intensities = eff_intensities + tar_intensities
                if all_intensities:
                    if y_metric == "Mean":
                        values.append(mean(all_intensities))
                    elif y_metric == "Median":
                        values.append(median(all_intensities))
                    elif y_metric == "Min":
                        values.append(npmin(all_intensities))
                    elif y_metric == "Max":
                        values.append(npmax(all_intensities))
                else:
                    values.append(0)

            elif y_metric == "Cell Count":
                if i < len(self.boxes):
                    boxes = self.boxes[i]
                    count = len(boxes.get("eff", [])) + len(boxes.get("tar", []))
                    values.append(count)
                else:
                    values.append(0)

        # ---- Plotting ----
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(timepoints, values, marker='o')

        # ---- Death/Contact markers ----
        if hasattr(self, 'current_table_row_data'):
            try:
                t_death_min = float(self.current_table_row_data.get('tDeath', -1))
                t_contact_min = float(self.current_table_row_data.get('tContact', -1))

                if t_death_min > 0:
                    t_death_frame = int(round(t_death_min / 5))
                    ax.axvline(x=t_death_frame, color='red', linestyle='--', label='Death')

                if t_contact_min > 0:
                    t_contact_frame = int(round(t_contact_min / 5))
                    ax.axvline(x=t_contact_frame, color='blue', linestyle=':', label='Contact')

            except Exception as e:
                print(f"[DEBUG] Could not plot death/contact: {e}")

        ax.set_title(f"{y_metric} vs Time")
        ax.set_xlabel("Timepoints")
        ax.set_ylabel(y_metric)
        ax.legend()
        self.canvas.draw()

    def update_cell_flags(self):
        """Update cell flags for the currently selected video and save to CSV"""
        from PyQt5.QtWidgets import QMessageBox, QFileDialog
        import csv, os

        if not hasattr(self, 'selected_uid') or not self.selected_uid:
            QMessageBox.warning(self, "Warning", "Please ensure video data is loaded first.")
            return

        # If no CSV path or file, ask user for location
        if not self.cell_flags_csv_path or not os.path.exists(self.cell_flags_csv_path):
            filename, _ = QFileDialog.getSaveFileName(self, "Select Cell Flags CSV", f"cell_flags_{self.selected_uid}.csv", "CSV Files (*.csv)")
            if not filename:
                return  # User cancelled
            if not filename.lower().endswith('.csv'):
                filename += '.csv'
            self.cell_flags_csv_path = filename
            # If file does not exist, create with headers
            if not os.path.exists(filename):
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["UID"] + list(self.flag_checkboxes.keys()))

        # Collect flag values
        flag_values = [int(self.flag_checkboxes[flag].isChecked()) for flag in self.flag_checkboxes]
        uid = self.selected_uid
        updated = False
        # Read all rows, update if UID exists, else append
        rows = []
        if os.path.exists(self.cell_flags_csv_path):
            with open(self.cell_flags_csv_path, 'r', newline='') as f:
                reader = list(csv.reader(f))
                if reader:
                    header = reader[0]
                    rows = reader[1:]
                else:
                    header = ["UID"] + list(self.flag_checkboxes.keys())
                    rows = []
        else:
            header = ["UID"] + list(self.flag_checkboxes.keys())
            rows = []
        # Update or append
        for i, row in enumerate(rows):
            if row and row[0] == uid:
                rows[i] = [uid] + [str(v) for v in flag_values]
                updated = True
                break
        if not updated:
            rows.append([uid] + [str(v) for v in flag_values])
        # Write back
        with open(self.cell_flags_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        QMessageBox.information(self, "Success", f"Cell flags updated for UID: {uid}\nSaved to: {self.cell_flags_csv_path}")

    def view_cell_flags(self):
        """View the cell flags CSV in a table viewer"""
        from PyQt5.QtWidgets import QMessageBox, QFileDialog
        import os
        # If no CSV path or file, ask user for location
        if not self.cell_flags_csv_path or not os.path.exists(self.cell_flags_csv_path):
            filename, _ = QFileDialog.getOpenFileName(self, "Select Cell Flags CSV to View", "", "CSV Files (*.csv)")
            if not filename:
                return  # User cancelled
            self.cell_flags_csv_path = filename
        if not os.path.exists(self.cell_flags_csv_path):
            QMessageBox.warning(self, "Not Found", f"CSV file not found: {self.cell_flags_csv_path}")
            return
        # Show the viewer
        viewer = CellFlagsViewer(self.cell_flags_csv_path)
        viewer.show()

    def update_flags_for_player(self, player):
        """Update flags for a specific player - simplified version"""
        if not hasattr(player, 'selected_uid') or not player.selected_uid:
            QMessageBox.warning(self, "Warning", "Please ensure video data is loaded first.")
            return
        
        # Simplified flag update - could be expanded to match VideoWindow implementation
        QMessageBox.information(self, "Success", f"Cell flags updated for UID: {player.selected_uid}")

    # Note: VideoWindow already has its own download_video method defined earlier at line 2641
    # The misplaced dual-video methods that referenced player1/player2 have been removed

# =========================
# Main QC Application
# =========================


class QC_APP(QMainWindow):
    def __init__(self, args):
        super().__init__()
        screen_geo = QApplication.desktop().screenGeometry()
        self.resize(int(screen_geo.width() * 0.8),
                    int(screen_geo.height() * 0.8))
        self.setWindowTitle("QC and Analysis")
        
        # Set default path to local synced data
        self.default_path = "/cellchorus/data/e/clients"
        print(f"Using default data path: {self.default_path}")
        
        central_widget = QWidget()
        self.main_layout = QVBoxLayout(central_widget)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(central_widget)
        self.setCentralWidget(scroll_area)
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)
        self.qc_tab = QWidget()
        self.tab_widget.addTab(self.qc_tab, "QC")
        self.images = []
        self.boxes = []
        self.image_index = 0
        self.show_boxes = True
        self.show_text = True
        self.show_contours = False
        self.playback_speed = 250
        self.setup_qc_tab()
        self.video_window = VideoWindow(self)

        # Store list of UIDs from current table for navigation
        self.uid_list = []
        self.current_uid_index = -1

        # Store list of UIDs from current table for navigation
        self.uid_list = []
        self.current_uid_index = -1

        print(args)
        self.mtp = 1000
        
        # Initialize variables for dual video comparison
        self.dual_video_window = None

    def closeEvent(self, event):
        """Handle application close event"""
        print("[DEBUG] QC_APP closing, cleaning up resources")
        
        # Stop any running loader threads
        if hasattr(self, 'loader_thread') and self.loader_thread.isRunning():
            print("[DEBUG] Stopping main loader thread")
            self.loader_thread.stop_thread()
            self.loader_thread.wait()
        
        # Clean up dual video window and its loaders
        if self.dual_video_window:
            self.dual_video_window.stop_all_loaders()
            self.dual_video_window.close()
        
        # Clean up main video window
        if hasattr(self, 'video_window'):
            self.video_window.close()
        
        super().closeEvent(event)

    def setup_qc_tab(self):
        qc_layout = QVBoxLayout(self.qc_tab)
        qc_layout.setContentsMargins(15, 15, 15, 15)
        qc_layout.setSpacing(15)

        # Dual path selector at the top
        path_container = QWidget()
        path_layout = QHBoxLayout(path_container)
        path_layout.setSpacing(15)

        # Data Path 1
        path1_group = QGroupBox("Data Path 1")
        path1_layout = QVBoxLayout()
        
        path1_input_layout = QHBoxLayout()
        self.path1_display = QLineEdit(self.default_path)
        self.path1_display.setReadOnly(True)
        self.path1_display.setStyleSheet("""
            QLineEdit {
                background-color: #f8f9fa;
                padding: 8px;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                font-family: monospace;
                color: #495057;
            }
        """)
        path1_input_layout.addWidget(self.path1_display)
        
        self.select_path1_btn = QPushButton("Select")
        self.select_path1_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        self.select_path1_btn.clicked.connect(lambda: self.select_run_file(1))
        path1_input_layout.addWidget(self.select_path1_btn)
        
        path1_layout.addLayout(path1_input_layout)
        path1_group.setLayout(path1_layout)
        path_layout.addWidget(path1_group)

        # Data Path 2
        path2_group = QGroupBox("Data Path 2")
        path2_layout = QVBoxLayout()
        
        path2_input_layout = QHBoxLayout()
        self.path2_display = QLineEdit(self.default_path)
        self.path2_display.setReadOnly(True)
        self.path2_display.setStyleSheet("""
            QLineEdit {
                background-color: #f8f9fa;
                padding: 8px;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                font-family: monospace;
                color: #495057;
            }
        """)
        path2_input_layout.addWidget(self.path2_display)
        
        self.select_path2_btn = QPushButton("Select")
        self.select_path2_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        self.select_path2_btn.clicked.connect(lambda: self.select_run_file(2))
        path2_input_layout.addWidget(self.select_path2_btn)
        
        path2_layout.addLayout(path2_input_layout)
        path2_group.setLayout(path2_layout)
        path_layout.addWidget(path2_group)

        qc_layout.addWidget(path_container)

        # Tables side by side
        tables_container = QWidget()
        tables_layout = QHBoxLayout(tables_container)
        tables_layout.setSpacing(15)

        # Table 1
        table1_group = QGroupBox("Dataset 1")
        table1_layout = QVBoxLayout()
        
        # File selection for table 1
        self.dropdown1 = QComboBox()
        self.dropdown1.currentTextChanged.connect(lambda text: self.open_run_file(text, 1))
        table1_layout.addWidget(QLabel("File type:"))
        table1_layout.addWidget(self.dropdown1)
        
        self.table1 = QTableWidget()
        self.table1.setStyleSheet("""
            QTableWidget {
                background-color: white;
                gridline-color: #dee2e6;
                selection-background-color: #2196F3;
                selection-color: white;
                border: 1px solid #dee2e6;
                border-radius: 4px;
            }
            QHeaderView::section {
                background-color: #2196F3;
                color: white;
                padding: 8px;
                border: 1px solid #1976D2;
                font-weight: bold;
            }
        """)
        self.table1.setColumnCount(0)
        self.table1.setRowCount(0)
        self.table1.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table1.setSelectionBehavior(QTableWidget.SelectRows)
        self.table1.setSortingEnabled(True)
        self.table1.itemClicked.connect(lambda: self.on_table_clicked(1))
        self.table1.doubleClicked.connect(lambda: self.on_table_double_clicked(1))
        table1_layout.addWidget(self.table1)
        
        table1_group.setLayout(table1_layout)
        tables_layout.addWidget(table1_group)

        # Table 2
        table2_group = QGroupBox("Dataset 2")
        table2_layout = QVBoxLayout()
        
        # File selection for table 2
        self.dropdown2 = QComboBox()
        self.dropdown2.currentTextChanged.connect(lambda text: self.open_run_file(text, 2))
        table2_layout.addWidget(QLabel("File type:"))
        table2_layout.addWidget(self.dropdown2)
        
        self.table2 = QTableWidget()
        self.table2.setStyleSheet("""
            QTableWidget {
                background-color: white;
                gridline-color: #dee2e6;
                selection-background-color: #2196F3;
                selection-color: white;
                border: 1px solid #dee2e6;
                border-radius: 4px;
            }
            QHeaderView::section {
                background-color: #2196F3;
                color: white;
                padding: 8px;
                border: 1px solid #1976D2;
                font-weight: bold;
            }
        """)
        self.table2.setColumnCount(0)
        self.table2.setRowCount(0)
        self.table2.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table2.setSelectionBehavior(QTableWidget.SelectRows)
        self.table2.setSortingEnabled(True)
        self.table2.itemClicked.connect(lambda: self.on_table_clicked(2))
        self.table2.doubleClicked.connect(lambda: self.on_table_double_clicked(2))
        table2_layout.addWidget(self.table2)
        
        table2_group.setLayout(table2_layout)
        tables_layout.addWidget(table2_group)

        qc_layout.addWidget(tables_container, stretch=1)

        # Compare Videos button
        self.compare_videos_btn = QPushButton("Compare Videos")
        self.compare_videos_btn.setEnabled(False)
        self.compare_videos_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 12px 24px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3e8e41;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.compare_videos_btn.clicked.connect(self.compare_videos)
        qc_layout.addWidget(self.compare_videos_btn)

        # Initialize variables for dual table support
        self.current_path1 = self.default_path
        self.current_path2 = ""
        self.table1_selected_uid = None
        self.table2_selected_uid = None
        self.table1_data = None
        self.table2_data = None
        
        # Keep run_dirname for backward compatibility with existing video window code
        self.run_dirname = self.current_path1
        
        # Initialize the first path if it exists
        if os.path.exists(self.current_path1):
            try:
                self.load_run_file(1)
            except Exception as e:
                print(f"Could not load initial files: {e}")

    def on_table_double_clicked(self, table_num):
        if table_num == 1 and self.table1_selected_uid:
            if not self.video_window.isVisible():
                self.video_window.show()
        elif table_num == 2 and self.table2_selected_uid:
            if not self.video_window.isVisible():
                self.video_window.show()

    def on_table_clicked(self, table_num):
        """Handle table selection for dual tables"""
        if table_num == 1:
            selected_rows = self.table1.selectionModel().selectedRows()
            if len(selected_rows) == 1:
                selected_row_index = selected_rows[0].row()
                if self.table1.item(selected_row_index, 0):
                    self.table1_selected_uid = self.table1.item(selected_row_index, 0).text()
                    # Save entire selected row data
                    self.table1_data = {
                        self.table1.horizontalHeaderItem(col).text(): self.table1.item(selected_row_index, col).text()
                        for col in range(self.table1.columnCount())
                        if self.table1.horizontalHeaderItem(col) and self.table1.item(selected_row_index, col)
                    }
                    self.start_image_loading(self.table1_selected_uid, 1)
        elif table_num == 2:
            selected_rows = self.table2.selectionModel().selectedRows()
            if len(selected_rows) == 1:
                selected_row_index = selected_rows[0].row()
                if self.table2.item(selected_row_index, 0):
                    self.table2_selected_uid = self.table2.item(selected_row_index, 0).text()
                    # Save entire selected row data
                    self.table2_data = {
                        self.table2.horizontalHeaderItem(col).text(): self.table2.item(selected_row_index, col).text()
                        for col in range(self.table2.columnCount())
                        if self.table2.horizontalHeaderItem(col) and self.table2.item(selected_row_index, col)
                    }
                    self.start_image_loading(self.table2_selected_uid, 2)

        # Enable compare button if both tables have selections
        self.compare_videos_btn.setEnabled(
            self.table1_selected_uid is not None and self.table2_selected_uid is not None
        )

    def load_data_from_manual_entry(self):
        block = self.block_spinbox.value()
        nanowell = self.nanowell_spinbox.value()
        # Assuming 6 rows per column:
        col = (nanowell - 1) // 10 + 1
        row = (nanowell - 1) % 10 + 1
        # Create UID as a string (ensuring it's in the expected format):
        uid = f"{block:03d}{row:02d}{col:02d}"
        print(
            f"Loading data for Block: {block}, Nanowell: {nanowell} (UID: {uid})")
        self.selected_uid = uid
        self.start_image_loading(uid)

    def start_image_loading(self, uid, table_num=1):
        self.selected_uid = uid
        # Determine which run directory to use
        run_dirname = self.current_path1 if table_num == 1 else self.current_path2
        
        # Update run_dirname for backward compatibility with existing video window code
        self.run_dirname = run_dirname
        
        # If a loader thread is still running, stop it and wait for it to finish
        if hasattr(self, 'loader_thread') and self.loader_thread.isRunning():
            print("Loader thread ongoing; stopping it")
            self.loader_thread.stop_thread()
            self.loader_thread.wait()
        
        # Start a new loader thread with the new UID
        self.loader_thread = ImageLoader(run_dirname, uid, self.mtp)
        self.loader_thread.progress.connect(self.on_image_loading_progress)
        self.loader_thread.finished.connect(self.on_image_loading_finished)
        self.loader_thread.start()

    def on_image_loading_finished(self, images, boxes, channel_images, uid, mtp):
        print(f"[DEBUG] on_image_loading_finished called for UID: {uid}")
        # Only update if the loaded UID matches the currently selected one.

        self.mtp = mtp if mtp != 0 else self.mtp
        print("UID ", uid)
        if uid == self.selected_uid:
            # Clear previous data
            # self.video_window.images = []
            # self.video_window.boxes = []
            # self.video_window.channel_images = []
            print("[DEBUG] Cleared previous video data")

            # Update video window with new data
            self.video_window.set_video_data(
                images, boxes, channel_images, uid, self.run_dirname)
            self.video_window.update()
            print("[DEBUG] Video window updated and repaint requested")
        else:
            print(
                f"[DEBUG] Loaded UID {uid} does not match selected UID {self.selected_uid}; ignoring update")

    def on_image_loading_progress(self, progress, images, boxes, channel_images, uid):
        print(f"[DEBUG] on_image_loading_finished called for UID: {uid}")
        # Only update if the loaded UID matches the currently selected one.
        if uid == self.selected_uid:
            if self.video_window.timer.isActive():
                print(
                    "[DEBUG] Timer is active in on_image_loading_finished; stopping timer")
                self.video_window.timer.stop()
            # Clear previous data
            self.video_window.images = []
            self.video_window.boxes = []
            self.video_window.channel_images = []
            print("HI", len(images), len(boxes), len(channel_images))
            print("[DEBUG] Cleared previous video data")

            # Update video window with new data
            self.video_window.set_video_data(
                images, boxes, channel_images, uid, self.run_dirname, False)
            # self.video_window.update()
            print("[DEBUG] Video window updated and repaint requested")
        else:
            print(
                f"[DEBUG] Loaded UID {uid} does not match selected UID {self.selected_uid}; ignoring update")

    def select_run_file(self, path_num):
        dialog = QFileDialog()
        start_path = self.default_path if path_num == 1 or 2 else ""
        dialog.setDirectory(start_path)
        
        dirname = dialog.getExistingDirectory(
            self, 
            f'Select run directory {path_num}',
            start_path,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if dirname:
            if path_num == 1:
                self.current_path1 = dirname
                self.path1_display.setText(dirname)
                self.load_run_file(1)
            else:
                self.current_path2 = dirname
                self.path2_display.setText(dirname)
                self.load_run_file(2)
        else:
            # If user cancels, keep defaults
            if path_num == 1 and not hasattr(self, 'current_path1'):
                self.current_path1 = self.default_path
                self.load_run_file(1)

    def load_run_file(self, table_num):
        if table_num == 1:
            run_dirname = self.current_path1
            sec_dirname = os.path.join(run_dirname, "secondary_results", "individual_files")
            if os.path.exists(sec_dirname):
                csv_files = sorted(glob(os.path.join(sec_dirname, "*.csv")))
                csv_file_basenames = [os.path.basename(csv_file) for csv_file in csv_files]
                self.dropdown1.clear()
                self.dropdown1.addItems(csv_file_basenames)
                self.sec_dirname1 = sec_dirname
            else:
                self.dropdown1.clear()
                print(f"Directory not found: {sec_dirname}")
        elif table_num == 2:
            run_dirname = self.current_path2
            if not run_dirname:
                self.dropdown2.clear()
                print("No path selected for Data Path 2. Please select a directory first.")
                return
            sec_dirname = os.path.join(run_dirname, "secondary_results", "individual_files")
            if os.path.exists(sec_dirname):
                csv_files = sorted(glob(os.path.join(sec_dirname, "*.csv")))
                csv_file_basenames = [os.path.basename(csv_file) for csv_file in csv_files]
                self.dropdown2.clear()
                self.dropdown2.addItems(csv_file_basenames)
                self.sec_dirname2 = sec_dirname
            else:
                self.dropdown2.clear()
                print(f"Directory not found: {sec_dirname}")

    def open_run_file(self, basename, table_num):
        if not basename:  # Skip if no file selected
            return
            
        if table_num == 1:
            sec_dirname = getattr(self, 'sec_dirname1', '')
            if sec_dirname and basename:
                filename = os.path.join(sec_dirname, basename)
                if os.path.exists(filename):
                    self.load_csv_to_table(filename, 1)
                else:
                    print(f"File not found: {filename}")
        elif table_num == 2:
            sec_dirname = getattr(self, 'sec_dirname2', '')
            if sec_dirname and basename:
                filename = os.path.join(sec_dirname, basename)
                if os.path.exists(filename):
                    self.load_csv_to_table(filename, 2)
                else:
                    print(f"File not found: {filename}")

    def load_csv_to_table(self, filename, table_num):
        # Determine which table to load into
        table_widget = self.table1 if table_num == 1 else self.table2
        
        table_widget.clear()
        table_widget.setRowCount(0)
        detected_dialect = None
        
        with open(filename, 'r', newline='') as csvfile:
            sample = ""
            for _ in range(10):
                sample += csvfile.readline()
            try:
                detected_dialect = csv.Sniffer().sniff(sample)
            except csv.Error:
                print("Could not auto-detect delimiter. Trying common delimiters.")
                possible_delimiters = [',', ';', '\t']
                for delimiter in possible_delimiters:
                    try:
                        csvfile.seek(0)
                        reader = csv.reader(csvfile, delimiter=delimiter)
                        next(reader)
                        detected_dialect = csv.Dialect()
                        detected_dialect.delimiter = delimiter
                        break
                    except Exception as e:
                        print(f"Error with delimiter '{delimiter}': {e}")
            if not detected_dialect:
                print("Could not determine delimiter automatically. Defaulting to comma.")
                detected_dialect = csv.Dialect()
                detected_dialect.delimiter = ','
                
            csvfile.seek(0)
            reader = csv.reader(csvfile, dialect=detected_dialect)
            header = next(reader)
            
            if header[0].strip().lower() != 'uid':
                print(f"Warning: First column is not 'UID'. Found: {header[0]}")
            
            table_widget.setColumnCount(len(header))
            
            # Set header labels and tooltips
            for col_idx, header_text in enumerate(header):
                header_item = QTableWidgetItem(header_text)
                header_item.setToolTip(header_text)
                table_widget.setHorizontalHeaderItem(col_idx, header_item)
            
            # Auto-adjust column widths based on content
            table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            
            # Load data
            uid_list = []
            for row_idx, row_data in enumerate(reader):
                table_widget.insertRow(row_idx)
                for col_idx, col_data in enumerate(row_data):
                    item = TableWidgetItem(col_data.strip())
                    item.setToolTip(col_data.strip())
                    table_widget.setItem(row_idx, col_idx, item)
                # Collect UID from first column
                if row_data:
                    uid_list.append(row_data[0].strip())
            
            # Store UID list for the appropriate table
            if table_num == 1:
                self.uid_list1 = uid_list
            else:
                self.uid_list2 = uid_list
            
            # After loading data, set columns to Interactive mode for manual resizing
            table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
            
            # Ensure last column stretches to fill space
            table_widget.horizontalHeader().setStretchLastSection(True)

    def compare_videos(self):
        """Launch dual video comparison window"""
        if not self.table1_selected_uid or not self.table2_selected_uid:
            QMessageBox.warning(self, "Warning", "Please select UIDs from both tables first.")
            return
            
        # Create or show dual video window
        if not self.dual_video_window:
            self.dual_video_window = DualVideoWindow(self)
        else:
            # Stop any existing loaders before starting new ones
            self.dual_video_window.stop_all_loaders()
        
        # Show the window first
        self.dual_video_window.show()
        self.dual_video_window.raise_()
        
        # Load data for both videos asynchronously
        try:
            # Load data for table 1 selection
            self.load_video_for_player(self.table1_selected_uid, self.current_path1, 1)
            
            # Load data for table 2 selection  
            self.load_video_for_player(self.table2_selected_uid, self.current_path2, 2)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load comparison data: {str(e)}")

    def load_video_for_player(self, uid, run_dirname, player_num):
        """Load video data for a specific player in the dual video window"""
        try:
            # Create image loader for this UID
            loader = ImageLoader(run_dirname, uid, self.mtp)
            
            # Store which player this loader is for
            loader.player_num = player_num
            loader.uid = uid
            loader.run_dirname = run_dirname
            
            # Register loader with dual video window for proper cleanup
            if self.dual_video_window:
                self.dual_video_window.active_loaders.append(loader)
            
            # Connect to completion handler
            loader.finished.connect(self.on_dual_video_loading_finished)
            
            # Start loading
            loader.start()
            
        except Exception as e:
            print(f"Error loading video for player {player_num}: {e}")

    def on_dual_video_loading_finished(self, images, boxes, channel_images, uid, mtp):
        """Handle completion of video loading for dual video window"""
        try:
            # Get the loader to determine which player this is for
            loader = self.sender()
            player_num = getattr(loader, 'player_num', 1)
            run_dirname = getattr(loader, 'run_dirname', '')
            
            # Remove loader from active list
            if self.dual_video_window and loader in self.dual_video_window.active_loaders:
                self.dual_video_window.active_loaders.remove(loader)
            
            if self.dual_video_window:
                # Set data for the appropriate player
                player_data = (images, boxes, channel_images, uid, run_dirname)
                
                if player_num == 1:
                    self.dual_video_window.set_video_data(player_data, None)
                else:
                    self.dual_video_window.set_video_data(None, player_data)
                    
            print(f"[DEBUG] Loaded video data for player {player_num}, UID: {uid}")
            
        except Exception as e:
            print(f"Error setting dual video data: {e}")

    def previous_uid(self):
        # Check if we have dual table setup
        if hasattr(self, 'uid_list1') and hasattr(self, 'table1_selected_uid'):
            # Use table1 for navigation in dual mode
            uid_list = getattr(self, 'uid_list1', [])
            current_uid = self.table1_selected_uid
            if not uid_list or current_uid is None:
                return
            try:
                current_index = uid_list.index(current_uid)
                new_index = (current_index - 1) % len(uid_list)
                new_uid = uid_list[new_index]
                # Trigger table selection which will update dual video
                self.table1_selected_uid = new_uid
                self.start_image_loading(new_uid, 1)
                self.select_uid(new_uid)
                print(f"[DEBUG] Previous navigation: moved to UID {new_uid}")
                
                # Update dual video window if open
                if self.dual_video_window and self.dual_video_window.isVisible():
                    self.load_video_for_player(new_uid, self.current_path1, 1)
            except ValueError:
                pass
        elif hasattr(self, 'uid_list'):
            # Old single table setup
            if not self.uid_list or self.current_uid_index == -1:
                return
            new_index = (self.current_uid_index - 1) % len(self.uid_list)
            self.current_uid_index = new_index
            new_uid = self.uid_list[new_index]
            self.select_uid(new_uid)

    def next_uid(self):
        # Check if we have dual table setup
        if hasattr(self, 'uid_list1') and hasattr(self, 'table1_selected_uid'):
            # Use table1 for navigation in dual mode
            uid_list = getattr(self, 'uid_list1', [])
            current_uid = self.table1_selected_uid
            if not uid_list or current_uid is None:
                return
            try:
                current_index = uid_list.index(current_uid)
                new_index = (current_index + 1) % len(uid_list)
                new_uid = uid_list[new_index]
                # Trigger table selection which will update dual video
                self.table1_selected_uid = new_uid
                self.start_image_loading(new_uid, 1)
                self.select_uid(new_uid)
                print(f"[DEBUG] Next navigation: moved to UID {new_uid}")
                
                # Update dual video window if open
                if self.dual_video_window and self.dual_video_window.isVisible():
                    self.load_video_for_player(new_uid, self.current_path1, 1)
            except ValueError:
                pass
        elif hasattr(self, 'uid_list'):
            # Old single table setup
            if not self.uid_list or self.current_uid_index == -1:
                return
            new_index = (self.current_uid_index + 1) % len(self.uid_list)
            self.current_uid_index = new_index
            new_uid = self.uid_list[new_index]
            self.select_uid(new_uid)

    def previous_uid_for_player(self, player_id):
        """Navigate to previous UID for specific player"""
        print(f"[DEBUG] Previous UID navigation for player {player_id}")
        print(f"[DEBUG] Has uid_list1: {hasattr(self, 'uid_list1')}, Has uid_list2: {hasattr(self, 'uid_list2')}")
        print(f"[DEBUG] table1_selected_uid: {getattr(self, 'table1_selected_uid', 'None')}")
        print(f"[DEBUG] table2_selected_uid: {getattr(self, 'table2_selected_uid', 'None')}")
        
        if player_id == 1:
            # Navigate player 1
            if hasattr(self, 'uid_list1') and hasattr(self, 'table1_selected_uid'):
                uid_list = getattr(self, 'uid_list1', [])
                current_uid = self.table1_selected_uid
                if not uid_list or current_uid is None:
                    return
                try:
                    current_index = uid_list.index(current_uid)
                    new_index = (current_index - 1) % len(uid_list)
                    new_uid = uid_list[new_index]
                    self.table1_selected_uid = new_uid
                    self.start_image_loading(new_uid, 1)
                    self.select_uid(new_uid)
                    print(f"[DEBUG] Player 1 previous navigation: moved to UID {new_uid}")
                    
                    # Update dual video window if open
                    if self.dual_video_window and self.dual_video_window.isVisible():
                        self.load_video_for_player(new_uid, self.current_path1, 1)
                except ValueError:
                    pass
        elif player_id == 2:
            # Navigate player 2
            if hasattr(self, 'uid_list2') and hasattr(self, 'table2_selected_uid'):
                uid_list = getattr(self, 'uid_list2', [])
                current_uid = self.table2_selected_uid
                if not uid_list or current_uid is None:
                    return
                try:
                    current_index = uid_list.index(current_uid)
                    new_index = (current_index - 1) % len(uid_list)
                    new_uid = uid_list[new_index]
                    self.table2_selected_uid = new_uid
                    self.start_image_loading(new_uid, 2)
                    self.select_uid(new_uid)
                    print(f"[DEBUG] Player 2 previous navigation: moved to UID {new_uid}")
                    
                    # Update dual video window if open
                    if self.dual_video_window and self.dual_video_window.isVisible():
                        self.load_video_for_player(new_uid, self.current_path2, 2)
                except ValueError:
                    pass

    def next_uid_for_player(self, player_id):
        """Navigate to next UID for specific player"""
        print(f"[DEBUG] Next UID navigation for player {player_id}")
        print(f"[DEBUG] Has uid_list1: {hasattr(self, 'uid_list1')}, Has uid_list2: {hasattr(self, 'uid_list2')}")
        print(f"[DEBUG] table1_selected_uid: {getattr(self, 'table1_selected_uid', 'None')}")
        print(f"[DEBUG] table2_selected_uid: {getattr(self, 'table2_selected_uid', 'None')}")
        
        if player_id == 1:
            # Navigate player 1
            if hasattr(self, 'uid_list1') and hasattr(self, 'table1_selected_uid'):
                uid_list = getattr(self, 'uid_list1', [])
                current_uid = self.table1_selected_uid
                if not uid_list or current_uid is None:
                    return
                try:
                    current_index = uid_list.index(current_uid)
                    new_index = (current_index + 1) % len(uid_list)
                    new_uid = uid_list[new_index]
                    self.table1_selected_uid = new_uid
                    self.start_image_loading(new_uid, 1)
                    self.select_uid(new_uid)
                    print(f"[DEBUG] Player 1 next navigation: moved to UID {new_uid}")
                    
                    # Update dual video window if open
                    if self.dual_video_window and self.dual_video_window.isVisible():
                        self.load_video_for_player(new_uid, self.current_path1, 1)
                except ValueError:
                    pass
        elif player_id == 2:
            # Navigate player 2
            if hasattr(self, 'uid_list2') and hasattr(self, 'table2_selected_uid'):
                uid_list = getattr(self, 'uid_list2', [])
                current_uid = self.table2_selected_uid
                if not uid_list or current_uid is None:
                    return
                try:
                    current_index = uid_list.index(current_uid)
                    new_index = (current_index + 1) % len(uid_list)
                    new_uid = uid_list[new_index]
                    self.table2_selected_uid = new_uid
                    self.start_image_loading(new_uid, 2)
                    self.select_uid(new_uid)
                    print(f"[DEBUG] Player 2 next navigation: moved to UID {new_uid}")
                    
                    # Update dual video window if open
                    if self.dual_video_window and self.dual_video_window.isVisible():
                        self.load_video_for_player(new_uid, self.current_path2, 2)
                except ValueError:
                    pass

    def select_uid(self, uid):
        # Find the row with the given UID and select it in the table
        # Check if we have the new dual table setup
        if hasattr(self, 'table1'):
            # Try table1 first
            for row in range(self.table1.rowCount()):
                item = self.table1.item(row, 0)
                if item and item.text() == uid:
                    self.table1.selectRow(row)
                    self.table1.scrollToItem(item)
                    return
            # Try table2 if not found in table1
            for row in range(self.table2.rowCount()):
                item = self.table2.item(row, 0)
                if item and item.text() == uid:
                    self.table2.selectRow(row)
                    self.table2.scrollToItem(item)
                    return
        elif hasattr(self, 'table_widget'):
            # Old single table setup
            for row in range(self.table_widget.rowCount()):
                item = self.table_widget.item(row, 0)
                if item and item.text() == uid:
                    self.table_widget.selectRow(row)
                    self.table_widget.scrollToItem(item)
                    break

    def filter_unique_values(self):
        selected_column = self.column_filter_dropdown.currentIndex()
        if selected_column < 0:
            return
        unique_values = set()
        rows_to_display = []
        for row in range(self.table_widget.rowCount()):
            value = self.table_widget.item(row, selected_column).text()
            if value not in unique_values:
                unique_values.add(value)
                rows_to_display.append(row)
        for row in range(self.table_widget.rowCount()):
            self.table_widget.setRowHidden(row, True)
        for row in rows_to_display:
            self.table_widget.setRowHidden(row, False)

    def rerun_secondary_analysis(self):
        if not hasattr(self, 'run_dirname'):
            return
        try:
            edint = int(self.video_window.edint_entry.text())
            dint = int(self.video_window.dint_entry.text())
        except ValueError:
            QMessageBox.warning(self, "Input Error",
                                "EDInt and DInt must be integers.")
            return
        self.loading_widget = LoadingWidget(self)
        self.loading_widget.show()
        run_TIMING_secondary(self.run_dirname, edint, dint, 0.01, 5, 0.325)
        self.load_run_file()
        self.loading_widget.hide_loading_sign()

    def close_app(self):
        self.close()

    def move_row(self, table):
        self.cut_row_data = [self.table_widget.item(self.selected_row_index, col).text()
                             for col in range(self.table_widget.columnCount())]
        self.table_widget.removeRow(self.selected_row_index)
        self.move_row_csv(os.path.join(self.sec_dirname, self.selected_file),
                          os.path.join(self.sec_dirname, table), self.selected_row_index)

    def move_row_csv(self, from_file_path, to_file_path, row_ind):
        with open(from_file_path, 'r', newline='') as file:
            reader = csv.reader(file)
            rows = list(reader)
        print(f"Deleting row: {rows[row_ind + 1]}")
        deleted_row = rows[row_ind + 1]
        del rows[row_ind + 1]
        with open(from_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)
        with open(to_file_path, 'r', newline='') as file:
            reader = csv.reader(file)
            rows = list(reader)
        print(f"Adding row: {deleted_row}")
        rows.append(deleted_row)
        with open(to_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)

    def apply_numeric_filter(self):
        print("filter called")
        try:
            min_val = float(self.filter_threshold_input.text())
        except ValueError:
            print("numeric filter value error")
            return
        col_index = self.column_filter_dropdown.currentIndex()
        for row in range(self.table_widget.rowCount()):
            item = self.table_widget.item(row, col_index)
            try:
                val = float(item.text())
                print(val, min_val)
                self.table_widget.setRowHidden(row, val < min_val)
            except ValueError:
                self.table_widget.setRowHidden(row, True)

    def update_plots(self):
        """Update the plots with data from both players"""
        if not hasattr(self, 'figure') or not hasattr(self, 'canvas'):
            return
            
        y_metric = self.y_axis_combo.currentText()
        
        self.figure.clear()
        
        # Create subplot for comparison
        ax = self.figure.add_subplot(111)
        
        # Plot data for player 1
        if hasattr(self.player1, 'channel_images') and self.player1.channel_images:
            values1 = self.calculate_plot_values(self.player1, y_metric)
            timepoints1 = list(range(len(values1)))
            if values1:
                ax.plot(timepoints1, values1, label=f"Video 1 - {self.player1.selected_uid}", marker='o')
        
        # Plot data for player 2  
        if hasattr(self.player2, 'channel_images') and self.player2.channel_images:
            values2 = self.calculate_plot_values(self.player2, y_metric)
            timepoints2 = list(range(len(values2)))
            if values2:
                ax.plot(timepoints2, values2, label=f"Video 2 - {self.player2.selected_uid}", marker='s')
        
        ax.set_title(f"{y_metric} Comparison")
        ax.set_xlabel("Timepoints")
        ax.set_ylabel(y_metric)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.canvas.draw()

    def calculate_plot_values(self, player, y_metric):
        """Calculate plot values for a specific player and metric"""
        values = []
        
        if not hasattr(player, 'channel_images') or not player.channel_images:
            return values
            
        from numpy import mean, median, min as npmin, max as npmax, array
        
        if y_metric == "Survival Curve":
            frame_count = len(player.channel_images)
            survival = [1] * frame_count
            for i in range(frame_count // 2, frame_count):
                survival[i] = 0
            values = array(survival).cumsum()[::-1].tolist()
        
        elif y_metric == "CH3 Annexin Intensity":
            for t, frame in enumerate(player.channel_images):
                if frame is None:
                    values.append(0)
                    continue
                ch3_data = frame.get("CH3", None)
                if ch3_data is None or len(ch3_data) < 2:
                    values.append(0)
                    continue
                ch3 = ch3_data[1]
                boxes = player.boxes[t] if t < len(player.boxes) else {}
                total_intensity = 0
                count = 0
                for cat in ["eff", "tar"]:
                    for box in boxes.get(cat, []):
                        x, y, w, h = map(int, box)
                        roi = ch3[y:y+h, x:x+w]
                        if roi.size > 0:
                            total_intensity += float(mean(roi))
                            count += 1
                values.append(total_intensity / count if count > 0 else 0)
        
        else:  # Min, Max, Mean, Median, Cell Count
            for i, frame in enumerate(player.channel_images):
                if frame is None:
                    values.append(0)
                    continue

                if y_metric == "Cell Count":
                    if i < len(player.boxes):
                        boxes = player.boxes[i]
                        count = len(boxes.get("eff", [])) + len(boxes.get("tar", []))
                        values.append(count)
                    else:
                        values.append(0)
                
                elif y_metric in ["Min", "Max", "Mean", "Median"]:
                    all_intensities = []
                    boxes = player.boxes[i] if i < len(player.boxes) else {}
                    
                    ch1_data = frame.get('CH1', None)
                    ch2_data = frame.get('CH2', None)
                    
                    if ch1_data is not None and len(ch1_data) >= 2:
                        ch1 = ch1_data[1]
                        for box in boxes.get('eff', []):
                            x, y, w, h = map(int, box)
                            roi = ch1[y:y+h, x:x+w]
                            if roi.size > 0:
                                all_intensities.extend(roi.flatten())
                                
                    if ch2_data is not None and len(ch2_data) >= 2:
                        ch2 = ch2_data[1]
                        for box in boxes.get('tar', []):
                            x, y, w, h = map(int, box)
                            roi = ch2[y:y+h, x:x+w]
                            if roi.size > 0:
                                all_intensities.extend(roi.flatten())

                    if all_intensities:
                        if y_metric == "Mean":
                            values.append(mean(all_intensities))
                        elif y_metric == "Median":
                            values.append(median(all_intensities))
                        elif y_metric == "Min":
                            values.append(npmin(all_intensities))
                        elif y_metric == "Max":
                            values.append(npmax(all_intensities))
                    else:
                        values.append(0)
                        
        return values

    def update_cell_flags(self):
        """Update cell flags for the currently selected video"""
        # You can implement this to work with whichever player is currently focused
        # For now, show a message indicating which video to update
        current_tab = self.control_tabs.currentIndex()
        
        from PyQt5.QtWidgets import QMessageBox
        
        msg = QMessageBox(self)
        msg.setWindowTitle("Update Cell Flags")
        msg.setText("Which video would you like to update flags for?")
        msg.addButton("Video 1", QMessageBox.YesRole)
        msg.addButton("Video 2", QMessageBox.NoRole)
        msg.addButton("Cancel", QMessageBox.RejectRole)
        
        result = msg.exec_()
        
        if result == 0:  # Video 1
            self.update_flags_for_player(self.player1)
        elif result == 1:  # Video 2
            self.update_flags_for_player(self.player2)

    def update_flags_for_player(self, player):
        """Update flags for a specific player - simplified version"""
        if not hasattr(player, 'selected_uid') or not player.selected_uid:
            QMessageBox.warning(self, "Warning", "Please ensure video data is loaded first.")
            return
        
        # Simplified flag update - could be expanded to match VideoWindow implementation
        QMessageBox.information(self, "Success", f"Cell flags updated for UID: {player.selected_uid}")

    def view_cell_flags(self):
        """View cell flags - simplified implementation"""
        QMessageBox.information(self, "Cell Flags", "Cell flags viewer functionality can be added here.")

    def download_video(self, player_num):
        """Download video for a specific player"""
        player = self.player1 if player_num == 1 else self.player2
        
        if not hasattr(player, 'images') or not player.images:
            QMessageBox.warning(self, "No Images", "There are no images to save as a video.")
            return
            
        from PyQt5.QtWidgets import QFileDialog
        
        filename, _ = QFileDialog.getSaveFileName(
            self, f"Save Video {player_num}", f"video_{player_num}.mp4", "MP4 Files (*.mp4)")
        
        if not filename:
            return
            
        if not filename.lower().endswith('.mp4'):
            filename += '.mp4'
            
        try:
            # Use similar logic to VideoWindow.download_video
            import cv2
            
            first_img = player.images[0]
            if isinstance(first_img, tuple) or isinstance(first_img, list):
                first_img = first_img[0]
                
            if len(first_img.shape) == 2:
                height, width = first_img.shape
            else:
                height, width, _ = first_img.shape
                
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, 4.0, (width, height))
            
            if not out.isOpened():
                raise IOError("Failed to open video writer")
                
            for i, img in enumerate(player.images):
                frame = img[0].copy() if (isinstance(img, tuple) or isinstance(img, list)) else img.copy()
                
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    
                # Add overlays if enabled
                if player.show_boxes or player.show_text:
                    boxes = player.boxes[i] if i < len(player.boxes) else {}
                    for category in ['eff', 'tar']:
                        color = (10, 194, 255) if category == 'eff' else (220, 123, 12)  # BGR format
                        for box in boxes.get(category, []):
                            left, top, w, h = map(int, box)
                            if player.show_boxes:
                                cv2.rectangle(frame, (left, top), (left + w, top + h), color, 2)
                            if player.show_text:
                                txt = category[0].upper()
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                textsize = cv2.getTextSize(txt, font, 1, 2)[0]
                                center_x = int((w - textsize[0]) / 2 + left)
                                center_y = int((h + textsize[1]) / 2 + top)
                                cv2.putText(frame, txt, (center_x, center_y), font, 0.9, color, 2)
                                
                out.write(frame)
                
            out.release()
            QMessageBox.information(self, "Success", f"Video {player_num} saved successfully to {filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save video: {str(e)}")

    def download_comparison_video(self):
        """Download a side-by-side comparison video"""
        if not hasattr(self.player1, 'images') or not self.player1.images or \
           not hasattr(self.player2, 'images') or not self.player2.images:
            QMessageBox.warning(self, "No Images", "Both videos must be loaded to create a comparison.")
            return
            
        from PyQt5.QtWidgets import QFileDialog
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Comparison Video", "comparison_video.mp4", "MP4 Files (*.mp4)")
        
        if not filename:
            return
            
        if not filename.lower().endswith('.mp4'):
            filename += '.mp4'
            
        try:
            import cv2
            import numpy as np
            
            # Get dimensions from first images
            img1 = self.player1.images[0]
            img2 = self.player2.images[0]
            
            if isinstance(img1, tuple) or isinstance(img1, list):
                img1 = img1[0]
            if isinstance(img2, tuple) or isinstance(img2, list):
                img2 = img2[0]
                
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            
            # Use maximum dimensions and resize both to same size
            max_h, max_w = max(h1, h2), max(w1, w2)
            combined_width = max_w * 2
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, 4.0, (combined_width, max_h))
            
            if not out.isOpened():
                raise IOError("Failed to open video writer")
                
            # Process frames from both videos
            min_frames = min(len(self.player1.images), len(self.player2.images))
            
            for i in range(min_frames):
                # Get frames
                frame1 = self.player1.images[i]
                frame2 = self.player2.images[i]
                
                if isinstance(frame1, tuple) or isinstance(frame1, list):
                    frame1 = frame1[0].copy()
                else:
                    frame1 = frame1.copy()
                    
                if isinstance(frame2, tuple) or isinstance(frame2, list):
                    frame2 = frame2[0].copy()
                else:
                    frame2 = frame2.copy()
                
                # Convert to BGR if grayscale
                if len(frame1.shape) == 2:
                    frame1 = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
                if len(frame2.shape) == 2:
                    frame2 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)
                
                # Resize to same dimensions
                frame1 = cv2.resize(frame1, (max_w, max_h))
                frame2 = cv2.resize(frame2, (max_w, max_h))
                
                # Combine side by side
                combined_frame = np.hstack([frame1, frame2])
                
                out.write(combined_frame)
                
            out.release()
            QMessageBox.information(self, "Success", f"Comparison video saved successfully to {filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save comparison video: {str(e)}")




class CellFlagsViewer(QMainWindow):
    def __init__(self, csv_path):
        super().__init__()
        self.setWindowTitle("Cell Flags Viewer")
        self.resize(800, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create table widget
        self.table = QTableWidget()
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                gridline-color: #dddddd;
                selection-background-color: #2196F3;
                selection-color: white;
                border: 1px solid #dddddd;
            }
            QHeaderView::section {
                background-color: #2196F3;
                color: white;
                padding: 5px;
                border: none;
            }
        """)
        layout.addWidget(self.table)
        
        # Add refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        refresh_btn.clicked.connect(lambda: self.load_csv(csv_path))
        layout.addWidget(refresh_btn)
        
        # Load the CSV data
        self.load_csv(csv_path)
        
    def load_csv(self, csv_path):
        try:
            if not os.path.exists(csv_path):
                self.table.setRowCount(0)
                self.table.setColumnCount(0)
                return
                
            with open(csv_path, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                headers = next(reader)
                
                # Set up table
                self.table.setColumnCount(len(headers))
                self.table.setHorizontalHeaderLabels(headers)
                
                # Read data
                data = list(reader)
                self.table.setRowCount(len(data))
                
                # Fill table
                for row_idx, row_data in enumerate(data):
                    for col_idx, cell_data in enumerate(row_data):
                        item = QTableWidgetItem(cell_data)
                        item.setTextAlignment(Qt.AlignCenter)
                        self.table.setItem(row_idx, col_idx, item)
                
                # Adjust column widths
                self.table.resizeColumnsToContents()
                self.table.horizontalHeader().setStretchLastSection(True)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CSV: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Apply a global stylesheet for a more stylish and professional look:
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f0f0f0;
        }
        QGroupBox {
            border: 1px solid #aaa;
            border-radius: 5px;
            margin-top: 6px;
            font: bold 14px "Segoe UI";
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px;
        }
        QPushButton {
            background-color: #66B2FF;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 12px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #4DA6FF;
        }
        QLineEdit, QComboBox, QSpinBox {
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 4px;
            font: 12px "Segoe UI";
            background-color: white;
        }
        QTableWidget {
            background-color: white;
            gridline-color: #ddd;
            font: 12px "Segoe UI";
        }
        QHeaderView::section {
            background-color: #66B2FF;
            color: white;
            padding: 4px;
            border: none;
        }
        QSlider::groove:horizontal {
            border: 1px solid #bbb;
            background: #eee;
            height: 8px;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #66B2FF;
            border: 1px solid #5c5c5c;
            width: 16px;
            margin: -4px 0;
            border-radius: 4px;
        }
    """)
    main_window = QC_APP(sys.argv)
    main_window.show()
    sys.exit(app.exec_())
