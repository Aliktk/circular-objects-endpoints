import sys
import cv2
import numpy as np
import logging
import warnings
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QGroupBox, QRadioButton, QButtonGroup, QSlider,
    QMessageBox, QScrollArea, QSizePolicy, QFormLayout, QStackedWidget
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

# --------------------- Suppress Deprecation Warnings ---------------------
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --------------------- Logging Configuration ---------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# --------------------- CoinDetectorApp Class ---------------------
class CoinDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Coin Detector Application')
        self.setGeometry(100, 100, 1600, 900)
        self.image = None
        self.processed_image = None
        self.method = 'Contour Detection'
        self.init_ui()
        # Initialize models if necessary
        # self.yolo_model = YOLO('yolov8_coins.pt')  # Example for YOLOv8
        # self.sam_model = ...  # Initialize SAM model

    def init_ui(self):
        # Main horizontal layout
        main_layout = QHBoxLayout()

        # -------------------- Column 1: Control Panel --------------------
        control_layout = QVBoxLayout()
        control_layout.setAlignment(Qt.AlignTop)
        control_layout.setContentsMargins(10, 10, 10, 10)
        control_layout.setSpacing(20)

        # Load Image Button
        self.load_button = QPushButton('Load Image')
        self.load_button.setFixedHeight(40)
        control_layout.addWidget(self.load_button)
        self.load_button.clicked.connect(self.load_image)

        # Detection Methods
        self.method_groupbox = QGroupBox('Detection Methods')
        methods = ['Contour Detection', 'Watershed Algorithm', 'YOLOv8 Segmentation', 'SAM Segmentation']
        self.method_button_group = QButtonGroup()
        method_layout = QVBoxLayout()
        for method in methods:
            rb = QRadioButton(method)
            if method == 'Contour Detection':
                rb.setChecked(True)
            self.method_button_group.addButton(rb)
            method_layout.addWidget(rb)
        self.method_groupbox.setLayout(method_layout)
        control_layout.addWidget(self.method_groupbox)
        self.method_button_group.buttonClicked.connect(self.change_method)

        # Parameter Controls using QStackedWidget
        self.param_groupbox = QGroupBox('Parameters')
        self.param_layout = QVBoxLayout()

        self.stacked_widget = QStackedWidget()
        self.param_layout.addWidget(self.stacked_widget)
        self.param_groupbox.setLayout(self.param_layout)
        control_layout.addWidget(self.param_groupbox)

        # Initialize parameter widgets for each method
        self.parameter_controls = {
            'Contour Detection': {},
            'Watershed Algorithm': {},
            'YOLOv8 Segmentation': {},
            'SAM Segmentation': {}
        }

        self.create_contour_parameters()
        self.create_watershed_parameters()
        self.create_yolo_parameters()
        self.create_sam_parameters()

        # Process Button
        self.process_button = QPushButton('Process')
        self.process_button.setFixedHeight(40)
        control_layout.addWidget(self.process_button)
        self.process_button.clicked.connect(self.process_image)

        # Save Result Button
        self.save_button = QPushButton('Save Result')
        self.save_button.setFixedHeight(40)
        control_layout.addWidget(self.save_button)
        self.save_button.clicked.connect(self.save_result)
        self.save_button.setEnabled(False)

        # Spacer to push controls to the top
        control_layout.addStretch()

        # Add control layout to main layout
        main_layout.addLayout(control_layout, 1)  # Weight 1 for narrow column

        # -------------------- Column 2: Input Image --------------------
        input_image_layout = QVBoxLayout()
        input_image_layout.setAlignment(Qt.AlignTop)
        input_image_layout.setContentsMargins(10, 10, 10, 10)
        input_image_layout.setSpacing(10)

        input_label = QLabel('Input Image')
        input_label.setAlignment(Qt.AlignCenter)
        input_label.setStyleSheet("font-weight: bold;")
        input_image_layout.addWidget(input_label)

        self.input_image_label = QLabel()
        self.input_image_label.setAlignment(Qt.AlignCenter)
        self.input_image_label.setStyleSheet("border: 1px solid black;")
        self.input_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        input_image_layout.addWidget(self.input_image_label)

        main_layout.addLayout(input_image_layout, 3)  # Weight 3 for wider column

        # -------------------- Column 3: Segmented Image with Overlay --------------------
        segmented_overlay_layout = QVBoxLayout()
        segmented_overlay_layout.setAlignment(Qt.AlignTop)
        segmented_overlay_layout.setContentsMargins(10, 10, 10, 10)
        segmented_overlay_layout.setSpacing(10)

        overlay_label = QLabel('Segmented Image with Overlay')
        overlay_label.setAlignment(Qt.AlignCenter)
        overlay_label.setStyleSheet("font-weight: bold;")
        segmented_overlay_layout.addWidget(overlay_label)

        self.segmented_image_label = QLabel()
        self.segmented_image_label.setAlignment(Qt.AlignCenter)
        self.segmented_image_label.setStyleSheet("border: 1px solid black;")
        self.segmented_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        segmented_overlay_layout.addWidget(self.segmented_image_label)

        main_layout.addLayout(segmented_overlay_layout, 3)  # Weight 3

        # -------------------- Column 4: Segmented Masks --------------------
        segmented_masks_layout = QVBoxLayout()
        segmented_masks_layout.setAlignment(Qt.AlignTop)
        segmented_masks_layout.setContentsMargins(10, 10, 10, 10)
        segmented_masks_layout.setSpacing(10)

        masks_label = QLabel('Segmented Masks')
        masks_label.setAlignment(Qt.AlignCenter)
        masks_label.setStyleSheet("font-weight: bold;")
        segmented_masks_layout.addWidget(masks_label)

        # Scroll Area for segmented coins
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area_widget = QWidget()
        self.scroll_layout = QHBoxLayout()
        self.scroll_layout.setAlignment(Qt.AlignLeft)
        self.scroll_area_widget.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_area_widget)
        segmented_masks_layout.addWidget(self.scroll_area)

        main_layout.addLayout(segmented_masks_layout, 3)  # Weight 3

        # Set central widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def create_contour_parameters(self):
        """Create parameter widget for Contour Detection."""
        try:
            widget = QWidget()
            layout = QFormLayout()

            # dp: Inverse ratio of the accumulator resolution to the image resolution.
            dp_slider = QSlider(Qt.Horizontal)
            dp_slider.setRange(1, 5)  # dp: 1 to 5
            dp_slider.setValue(1)
            dp_slider.setTickInterval(1)
            dp_slider.setTickPosition(QSlider.TicksBelow)
            dp_label = QLabel('dp: 1')
            dp_slider.valueChanged.connect(lambda value: dp_label.setText(f'dp: {value}'))

            # minDist: Minimum distance between the centers of the detected circles.
            minDist_slider = QSlider(Qt.Horizontal)
            minDist_slider.setRange(10, 300)  # minDist: 10 to 300
            minDist_slider.setValue(30)
            minDist_slider.setTickInterval(10)
            minDist_slider.setTickPosition(QSlider.TicksBelow)
            minDist_label = QLabel('minDist: 30')
            minDist_slider.valueChanged.connect(lambda value: minDist_label.setText(f'minDist: {value}'))

            # param1: Higher threshold for the Canny edge detector.
            param1_slider = QSlider(Qt.Horizontal)
            param1_slider.setRange(10, 200)  # param1: 10 to 200
            param1_slider.setValue(50)
            param1_slider.setTickInterval(10)
            param1_slider.setTickPosition(QSlider.TicksBelow)
            param1_label = QLabel('param1: 50')
            param1_slider.valueChanged.connect(lambda value: param1_label.setText(f'param1: {value}'))

            # param2: Accumulator threshold for the circle centers at the detection stage.
            param2_slider = QSlider(Qt.Horizontal)
            param2_slider.setRange(10, 200)  # param2: 10 to 200
            param2_slider.setValue(30)
            param2_slider.setTickInterval(10)
            param2_slider.setTickPosition(QSlider.TicksBelow)
            param2_label = QLabel('param2: 30')
            param2_slider.valueChanged.connect(lambda value: param2_label.setText(f'param2: {value}'))

            # minRadius: Minimum circle radius.
            minRadius_slider = QSlider(Qt.Horizontal)
            minRadius_slider.setRange(10, 100)  # minRadius: 10 to 100
            minRadius_slider.setValue(20)
            minRadius_slider.setTickInterval(10)
            minRadius_slider.setTickPosition(QSlider.TicksBelow)
            minRadius_label = QLabel('minRadius: 20')
            minRadius_slider.valueChanged.connect(lambda value: minRadius_label.setText(f'minRadius: {value}'))

            # maxRadius: Maximum circle radius.
            maxRadius_slider = QSlider(Qt.Horizontal)
            maxRadius_slider.setRange(10, 200)  # maxRadius: 10 to 200
            maxRadius_slider.setValue(50)
            maxRadius_slider.setTickInterval(10)
            maxRadius_slider.setTickPosition(QSlider.TicksBelow)
            maxRadius_label = QLabel('maxRadius: 50')
            maxRadius_slider.valueChanged.connect(lambda value: maxRadius_label.setText(f'maxRadius: {value}'))

            # Add to layout
            layout.addRow('dp:', dp_slider)
            layout.addRow('', dp_label)
            layout.addRow('minDist:', minDist_slider)
            layout.addRow('', minDist_label)
            layout.addRow('param1:', param1_slider)
            layout.addRow('', param1_label)
            layout.addRow('param2:', param2_slider)
            layout.addRow('', param2_label)
            layout.addRow('minRadius:', minRadius_slider)
            layout.addRow('', minRadius_label)
            layout.addRow('maxRadius:', maxRadius_slider)
            layout.addRow('', maxRadius_label)

            widget.setLayout(layout)
            self.stacked_widget.addWidget(widget)

            # Store references to sliders and labels
            self.parameter_controls['Contour Detection'] = {
                'dp': (dp_slider, dp_label),
                'minDist': (minDist_slider, minDist_label),
                'param1': (param1_slider, param1_label),
                'param2': (param2_slider, param2_label),
                'minRadius': (minRadius_slider, minRadius_label),
                'maxRadius': (maxRadius_slider, maxRadius_label)
            }
        except Exception as e:
            logging.error(f'Error creating contour parameters: {e}')
            QMessageBox.critical(self, 'Error', f'An error occurred while creating contour parameters:\n{e}')

    def create_watershed_parameters(self):
        """Create parameter widget for Watershed Algorithm."""
        try:
            widget = QWidget()
            layout = QFormLayout()

            # markers: Number of markers for watershed
            markers_slider = QSlider(Qt.Horizontal)
            markers_slider.setRange(1, 100)  # markers: 1 to 100
            markers_slider.setValue(50)
            markers_slider.setTickInterval(10)
            markers_slider.setTickPosition(QSlider.TicksBelow)
            markers_label = QLabel('Markers: 50')
            markers_slider.valueChanged.connect(lambda value: markers_label.setText(f'Markers: {value}'))

            # Add to layout
            layout.addRow('Markers:', markers_slider)
            layout.addRow('', markers_label)

            widget.setLayout(layout)
            self.stacked_widget.addWidget(widget)

            # Store references to sliders and labels
            self.parameter_controls['Watershed Algorithm'] = {
                'markers': (markers_slider, markers_label)
            }
        except Exception as e:
            logging.error(f'Error creating watershed parameters: {e}')
            QMessageBox.critical(self, 'Error', f'An error occurred while creating watershed parameters:\n{e}')

    def create_yolo_parameters(self):
        """Create parameter widget for YOLOv8 Segmentation."""
        try:
            widget = QWidget()
            layout = QFormLayout()

            # Confidence Threshold
            conf_slider = QSlider(Qt.Horizontal)
            conf_slider.setRange(10, 100)  # Confidence Threshold: 10% to 100%
            conf_slider.setValue(50)
            conf_slider.setTickInterval(10)
            conf_slider.setTickPosition(QSlider.TicksBelow)
            conf_label = QLabel('Confidence Threshold: 50%')
            conf_slider.valueChanged.connect(lambda value: conf_label.setText(f'Confidence Threshold: {value}%'))

            # NMS Threshold
            nms_slider = QSlider(Qt.Horizontal)
            nms_slider.setRange(10, 100)  # NMS Threshold: 10% to 100%
            nms_slider.setValue(50)
            nms_slider.setTickInterval(10)
            nms_slider.setTickPosition(QSlider.TicksBelow)
            nms_label = QLabel('NMS Threshold: 50%')
            nms_slider.valueChanged.connect(lambda value: nms_label.setText(f'NMS Threshold: {value}%'))

            # Add to layout
            layout.addRow('Confidence Threshold:', conf_slider)
            layout.addRow('', conf_label)
            layout.addRow('NMS Threshold:', nms_slider)
            layout.addRow('', nms_label)

            widget.setLayout(layout)
            self.stacked_widget.addWidget(widget)

            # Store references to sliders and labels
            self.parameter_controls['YOLOv8 Segmentation'] = {
                'confidence': (conf_slider, conf_label),
                'nms': (nms_slider, nms_label)
            }
        except Exception as e:
            logging.error(f'Error creating YOLOv8 parameters: {e}')
            QMessageBox.critical(self, 'Error', f'An error occurred while creating YOLOv8 parameters:\n{e}')

    def create_sam_parameters(self):
        """Create parameter widget for SAM Segmentation."""
        try:
            widget = QWidget()
            layout = QFormLayout()

            # Similarity Threshold
            sim_slider = QSlider(Qt.Horizontal)
            sim_slider.setRange(10, 100)  # Similarity Threshold: 10% to 100%
            sim_slider.setValue(50)
            sim_slider.setTickInterval(10)
            sim_slider.setTickPosition(QSlider.TicksBelow)
            sim_label = QLabel('Similarity Threshold: 50%')
            sim_slider.valueChanged.connect(lambda value: sim_label.setText(f'Similarity Threshold: {value}%'))

            # Add to layout
            layout.addRow('Similarity Threshold:', sim_slider)
            layout.addRow('', sim_label)

            widget.setLayout(layout)
            self.stacked_widget.addWidget(widget)

            # Store references to sliders and labels
            self.parameter_controls['SAM Segmentation'] = {
                'similarity': (sim_slider, sim_label)
            }
        except Exception as e:
            logging.error(f'Error creating SAM parameters: {e}')
            QMessageBox.critical(self, 'Error', f'An error occurred while creating SAM parameters:\n{e}')

    def load_image(self):
        try:
            options = QFileDialog.Options()
            filename, _ = QFileDialog.getOpenFileName(
                self, 'Open Image', '', 'Image Files (*.png *.jpg *.jpeg *.bmp)', options=options)
            if filename:
                logging.info(f'Loading image: {filename}')
                self.image = cv2.imread(filename)
                if self.image is not None:
                    self.display_image(self.image, self.input_image_label)
                    self.segmented_image_label.clear()
                    # Clear segmented masks display
                    while self.scroll_layout.count():
                        child = self.scroll_layout.takeAt(0)
                        if child.widget():
                            child.widget().deleteLater()
                    self.processed_image = None
                    self.save_button.setEnabled(False)
                    logging.info('Image loaded successfully.')
                else:
                    logging.error('Failed to load image.')
                    QMessageBox.warning(self, 'Error', 'Failed to load image.')
        except Exception as e:
            logging.error(f'Error loading image: {e}')
            QMessageBox.critical(self, 'Error', f'An error occurred while loading the image:\n{e}')

    def change_method(self):
        try:
            selected_button = self.method_button_group.checkedButton()
            self.method = selected_button.text()
            logging.info(f'Selected detection method: {self.method}')

            # Determine the index based on method
            method_indices = {
                'Contour Detection': 0,
                'Watershed Algorithm': 1,
                'YOLOv8 Segmentation': 2,
                'SAM Segmentation': 3
            }
            index = method_indices.get(self.method, 0)
            self.stacked_widget.setCurrentIndex(index)
            logging.info(f'Parameters updated for method: {self.method}')
        except Exception as e:
            logging.error(f'Error changing method: {e}')
            QMessageBox.critical(self, 'Error', f'An error occurred while changing the method:\n{e}')

    def process_image(self):
        try:
            if self.image is None:
                logging.warning('Process attempted without loading an image.')
                QMessageBox.warning(self, 'Error', 'Please load an image first.')
                return

            logging.info(f'Starting image processing using method: {self.method}')

            if self.method == 'Contour Detection':
                self.process_with_contour_detection()
            elif self.method == 'Watershed Algorithm':
                self.process_with_watershed()
            elif self.method == 'YOLOv8 Segmentation':
                self.process_with_yolo()
            elif self.method == 'SAM Segmentation':
                self.process_with_sam()

            if self.processed_image is not None:
                self.display_image(self.processed_image, self.segmented_image_label)
                self.save_button.setEnabled(True)
                logging.info('Image processing completed successfully.')
            else:
                logging.warning('No processed image to display.')
        except Exception as e:
            logging.error(f'Error during image processing: {e}')
            QMessageBox.critical(self, 'Error', f'An error occurred during image processing:\n{e}')

    def process_with_contour_detection(self):
        try:
            # Retrieve Contour Detection parameters from parameter_controls
            dp_slider, dp_label = self.parameter_controls['Contour Detection']['dp']
            minDist_slider, minDist_label = self.parameter_controls['Contour Detection']['minDist']
            param1_slider, param1_label = self.parameter_controls['Contour Detection']['param1']
            param2_slider, param2_label = self.parameter_controls['Contour Detection']['param2']
            minRadius_slider, minRadius_label = self.parameter_controls['Contour Detection']['minRadius']
            maxRadius_slider, maxRadius_label = self.parameter_controls['Contour Detection']['maxRadius']

            dp = dp_slider.value()
            minDist = minDist_slider.value()
            param1 = param1_slider.value()
            param2 = param2_slider.value()
            minRadius = minRadius_slider.value()
            maxRadius = maxRadius_slider.value()

            logging.info(f'Contour Detection Parameters - dp: {dp}, minDist: {minDist}, param1: {param1}, param2: {param2}, minRadius: {minRadius}, maxRadius: {maxRadius}')

            # Convert to grayscale
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            logging.info('Converted image to grayscale.')

            # Apply Gaussian Blur
            blurred = cv2.GaussianBlur(gray, (11, 11), 0)
            logging.info('Applied Gaussian Blur.')

            # Detect circles using Hough Circle Transform
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=dp,
                minDist=minDist,
                param1=param1,
                param2=param2,
                minRadius=minRadius,
                maxRadius=maxRadius
            )

            logging.info('Applied Hough Circle Transform.')

            # Create empty images for overlay and semantic segmentation
            overlay_img = self.image.copy()
            semantic_img = np.zeros_like(self.image)

            count = 0

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                logging.info(f'Found {len(circles)} circles.')

                for idx, (x, y, r) in enumerate(circles, start=1):
                    # Draw green circles on the overlay image
                    cv2.circle(overlay_img, (x, y), r, (0, 255, 0), 4)
                    cv2.circle(overlay_img, (x, y), 2, (0, 0, 255), 3)  # center point

                    # Create a mask for each coin
                    mask = np.zeros_like(gray)
                    cv2.circle(mask, (x, y), r, 255, -1)
                    segmented = cv2.bitwise_and(self.image, self.image, mask=mask)
                    semantic_img[mask == 255] = segmented[mask == 255]

                    # Logging each detected coin
                    logging.info(f'Detected coin #{idx}: Centroid=({x}, {y}), Radius={r}')
                    count += 1
            else:
                logging.warning('No circles detected.')

            # Overlay the count on the image
            cv2.putText(overlay_img, f'Count: {count}', (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
            logging.info(f'Number of coins detected: {count}')

            self.processed_image = overlay_img
            self.display_segmented_coins(semantic_img, count)
        except Exception as e:
            logging.error(f'Error in contour detection: {e}')
            QMessageBox.critical(self, 'Error', f'An error occurred during contour detection:\n{e}')

    def process_with_watershed(self):
        try:
            # Retrieve Watershed parameters from parameter_controls
            markers_slider, markers_label = self.parameter_controls['Watershed Algorithm']['markers']
            markers = markers_slider.value()

            logging.info(f'Watershed Parameters - markers: {markers}')

            # Convert to grayscale
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            logging.info('Converted image to grayscale.')

            # Apply threshold
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            logging.info('Applied thresholding.')

            # Noise removal using Morphology (Opening)
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            logging.info('Performed morphological opening.')

            # Sure background area
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            logging.info('Dilated to find sure background.')

            # Finding sure foreground area
            sure_fg = cv2.erode(opening, kernel, iterations=3)
            logging.info('Eroded to find sure foreground.')

            # Finding unknown region
            unknown = cv2.subtract(sure_bg, sure_fg)
            logging.info('Identified unknown regions.')

            # Marker labeling
            ret, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            logging.info('Connected components labeled.')

            # Apply Watershed
            markers = cv2.watershed(self.image, markers)
            img_overlay = self.image.copy()
            img_overlay[markers == -1] = [255, 0, 0]  # boundary marking in red
            logging.info('Applied Watershed Algorithm and marked boundaries.')

            # Count coins
            unique_markers = np.unique(markers)
            num_coins = len(unique_markers) - 2  # subtract 2 to exclude background and borders
            logging.info(f'Number of objects detected: {num_coins}')

            # Create a mask for the coins
            segmented_mask = np.zeros_like(self.image)
            for mark in unique_markers:
                if mark <= 1:
                    continue  # Background or border
                mask = np.where(markers == mark, 255, 0).astype('uint8')
                coin = cv2.bitwise_and(self.image, self.image, mask=mask)
                segmented_mask[mask == 255] = coin[mask == 255]

            # Overlay the count on the image
            cv2.putText(img_overlay, f'Count: {num_coins}', (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
            logging.info(f'Overlayed coin count on the image: {num_coins}')

            self.processed_image = img_overlay
            self.display_segmented_coins(segmented_mask, num_coins)
        except Exception as e:
            logging.error(f'Error in watershed algorithm: {e}')
            QMessageBox.critical(self, 'Error', f'An error occurred during watershed processing:\n{e}')

    def process_with_yolo(self):
        try:
            # Placeholder implementation for YOLOv8 Segmentation
            # Retrieve YOLOv8 parameters from parameter_controls
            current_widget = self.stacked_widget.currentWidget()
            layout = current_widget.layout()

            conf_slider, conf_label = self.parameter_controls['YOLOv8 Segmentation']['confidence']
            nms_slider, nms_label = self.parameter_controls['YOLOv8 Segmentation']['nms']

            confidence = conf_slider.value() / 100.0
            nms = nms_slider.value() / 100.0

            logging.info(f'YOLOv8 Parameters - Confidence Threshold: {confidence}, NMS Threshold: {nms}')

            # Example: Load YOLOv8 model (pseudo-code)
            # net = cv2.dnn.readNet('yolov8.weights', 'yolov8.cfg')
            # layers = net.getLayerNames()
            # output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]

            # For demonstration, we'll skip actual YOLOv8 implementation
            QMessageBox.information(self, 'Info', 'YOLOv8 Segmentation is not implemented yet.')
            logging.info('YOLOv8 Segmentation is not implemented yet.')

            # Example processed image
            self.processed_image = self.image.copy()
            count = 0  # Placeholder count

            # Overlay the count on the image
            cv2.putText(self.processed_image, f'Count: {count}', (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)

            # For semantic segmentation, create a dummy mask
            semantic_img = np.zeros_like(self.image)

            self.display_segmented_coins(semantic_img, count)
        except Exception as e:
            logging.error(f'Error in YOLOv8 segmentation: {e}')
            QMessageBox.critical(self, 'Error', f'An error occurred during YOLOv8 segmentation:\n{e}')

    def process_with_sam(self):
        try:
            # Placeholder implementation for SAM Segmentation
            # Retrieve SAM parameters from parameter_controls
            current_widget = self.stacked_widget.currentWidget()
            layout = current_widget.layout()

            sim_slider, sim_label = self.parameter_controls['SAM Segmentation']['similarity']
            similarity = sim_slider.value() / 100.0

            logging.info(f'SAM Parameters - Similarity Threshold: {similarity}')

            # Example: Apply SAM segmentation (pseudo-code)
            # sam = SAMModel()
            # masks = sam.segment(self.image, similarity=similarity)

            # For demonstration, we'll skip actual SAM implementation
            QMessageBox.information(self, 'Info', 'SAM Segmentation is not implemented yet.')
            logging.info('SAM Segmentation is not implemented yet.')

            # Example processed image
            self.processed_image = self.image.copy()
            count = 0  # Placeholder count

            # Overlay the count on the image
            cv2.putText(self.processed_image, f'Count: {count}', (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)

            # For semantic segmentation, create a dummy mask
            semantic_img = np.zeros_like(self.image)

            self.display_segmented_coins(semantic_img, count)
        except Exception as e:
            logging.error(f'Error in SAM segmentation: {e}')
            QMessageBox.critical(self, 'Error', f'An error occurred during SAM segmentation:\n{e}')

    def display_segmented_coins(self, semantic_img, count):
        try:
            # Clear previous segmented masks
            while self.scroll_layout.count():
                child = self.scroll_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

            if count > 0:
                # Resize semantic_img to match segmented image with overlay size
                # Get size of segmented_image_label
                overlay_pixmap = self.segmented_image_label.pixmap()
                if overlay_pixmap:
                    overlay_width = overlay_pixmap.width()
                    overlay_height = overlay_pixmap.height()
                else:
                    overlay_width = 400  # Default size
                    overlay_height = 400

                semantic_img_resized = cv2.resize(semantic_img, (overlay_width, overlay_height))

                # Convert BGR to RGB for display
                rgb_image = cv2.cvtColor(semantic_img_resized, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(
                    rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)

                # Create QLabel for segmented mask
                label = QLabel()
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet("border: 1px solid black;")
                label.setPixmap(pixmap)
                label.setFixedSize(overlay_width, overlay_height)
                label.setScaledContents(True)  # Ensures the image fits the label size

                self.scroll_layout.addWidget(label)
                logging.info('Displayed segmented mask.')
            else:
                logging.info('No segmented masks to display.')
        except Exception as e:
            logging.error(f'Error displaying segmented masks: {e}')
            QMessageBox.critical(self, 'Error', f'An error occurred while displaying segmented masks:\n{e}')

    def save_result(self):
        try:
            if self.processed_image is None:
                logging.warning('Save attempted without a processed image.')
                QMessageBox.warning(self, 'Error', 'No processed image to save.')
                return
            options = QFileDialog.Options()
            filename, _ = QFileDialog.getSaveFileName(
                self, 'Save Image', '', 'PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;Bitmap Files (*.bmp)', options=options)
            if filename:
                cv2.imwrite(filename, self.processed_image)
                logging.info(f'Processed image saved as: {filename}')
                QMessageBox.information(self, 'Success', 'Image saved successfully.')
        except Exception as e:
            logging.error(f'Error saving image: {e}')
            QMessageBox.critical(self, 'Error', f'An error occurred while saving the image:\n{e}')

    def display_image(self, cv_img, label):
        """Convert and display an image in a QLabel."""
        try:
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(
                rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            # Scale the pixmap to fit the label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled_pixmap)
        except Exception as e:
            logging.error(f'Error displaying image: {e}')
            QMessageBox.critical(self, 'Error', f'An error occurred while displaying the image:\n{e}')

    def resizeEvent(self, event):
        """Handle window resizing to adjust image displays."""
        try:
            if self.image is not None:
                self.display_image(self.image, self.input_image_label)
            if self.processed_image is not None:
                self.display_image(self.processed_image, self.segmented_image_label)
            # Update segmented masks
            for i in range(self.scroll_layout.count()):
                item = self.scroll_layout.itemAt(i)
                if item.widget():
                    pixmap = item.widget().pixmap()
                    if pixmap:
                        scaled_pixmap = pixmap.scaled(
                            item.widget().width(), item.widget().height(),
                            Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        item.widget().setPixmap(scaled_pixmap)
        except Exception as e:
            logging.error(f'Error during resize event: {e}')
        super().resizeEvent(event)

# --------------------- Main Execution ---------------------
if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        window = CoinDetectorApp()
        window.show()
        logging.info('Application started.')
        sys.exit(app.exec_())
    except Exception as e:
        logging.critical(f'Application crashed: {e}')
