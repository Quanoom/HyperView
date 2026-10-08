import numpy as np
import cv2  # Ensure installed via: pip install opencv-python
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QMessageBox
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QPointF, QRect, QRectF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QTransform, QLinearGradient, QColor, QFont


def map_temperature_to_color_dynamic(arr, min_temp=None, max_temp=None):
    """
    Map a temperature array (in °C) to an RGBA image array using dynamic thresholds.
    If min_temp and max_temp are given, then:
      - min_temp maps to white (255,255,255),
      - max_temp maps to red (255,0,0),
      - with blue, green, and yellow at 25%, 50%, and 75% of the temperature range, respectively.
    If min_temp and max_temp are not provided, fall back to fixed defaults.

    Returns an array of shape (H, W, 4) with dtype=np.uint8.
    """
    import numpy as np
    # If not provided, use default thresholds (as in your previous logic)
    if min_temp is None or max_temp is None:
        # Default values based on your original fixed mapping.
        min_temp = -15
        max_temp = 45

    # Compute the intermediate thresholds (linearly spaced)
    t0 = min_temp  # should map to white.
    t1 = min_temp + (max_temp - min_temp) * 0.25  # blue
    t2 = min_temp + (max_temp - min_temp) * 0.50  # green
    t3 = min_temp + (max_temp - min_temp) * 0.75  # yellow
    t4 = max_temp  # red

    # Prepare the output array.
    temp = np.array(arr, dtype=np.float32)
    # Create an output array of same spatial dimensions with an extra channel for RGBA.
    rgba = np.zeros(temp.shape + (4,), dtype=np.uint8)
    # Default alpha fully opaque.
    rgba[..., 3] = 255

    valid = ~np.isnan(temp)
    rgba[~valid] = [0, 0, 0, 0]
    t_valid = temp[valid]

    colors = np.zeros((t_valid.size, 3), dtype=np.float32)

    # Interpolate piecewise.
    # For temperatures between t0 and t1: white to blue.
    mask1 = t_valid <= t1
    if np.any(mask1):
        t_part = np.clip(t_valid[mask1], t0, t1)
        ratio = (t_part - t0) / (t1 - t0)
        # White: (255,255,255) to Blue: (0,0,255)
        colors[mask1, 0] = (1 - ratio) * 255  # Red channel fades from 255 to 0.
        colors[mask1, 1] = (1 - ratio) * 255  # Green channel.
        colors[mask1, 2] = 255  # Blue channel stays 255.
    # For temperatures between t1 and t2: blue to green.
    mask2 = (t_valid > t1) & (t_valid <= t2)
    if np.any(mask2):
        t_part = t_valid[mask2]
        ratio = (t_part - t1) / (t2 - t1)
        # Blue (0,0,255) to Green (0,255,0)
        colors[mask2, 0] = 0
        colors[mask2, 1] = ratio * 255
        colors[mask2, 2] = (1 - ratio) * 255
    # For temperatures between t2 and t3: green to yellow.
    mask3 = (t_valid > t2) & (t_valid <= t3)
    if np.any(mask3):
        t_part = t_valid[mask3]
        ratio = (t_part - t2) / (t3 - t2)
        # Green (0,255,0) to Yellow (255,255,0)
        colors[mask3, 0] = ratio * 255
        colors[mask3, 1] = 255
        colors[mask3, 2] = 0
    # For temperatures between t3 and t4: yellow to red.
    mask4 = t_valid > t3
    if np.any(mask4):
        t_part = np.clip(t_valid[mask4], t3, t4)
        ratio = (t_part - t3) / (t4 - t3)
        # Yellow (255,255,0) to Red (255,0,0)
        colors[mask4, 0] = 255
        colors[mask4, 1] = (1 - ratio) * 255
        colors[mask4, 2] = 0

    # Assign computed colors into the RGBA array.
    rgba[valid, 0:3] = np.clip(colors, 0, 255).astype(np.uint8)
    return rgba

from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPainter, QPen, QLinearGradient, QColor, QFont, QFontMetrics

class LegendWidget(QWidget):
    def __init__(self, parent=None, t_low=-15, t_blue=0, t_green=15, t_yellow=30, t_high=45):
        super().__init__(parent)
        self.t_low = t_low
        self.t_blue = t_blue
        self.t_green = t_green
        self.t_yellow = t_yellow
        self.t_high = t_high
        self.setFixedHeight(80)
        self.setSizePolicy(self.sizePolicy().horizontalPolicy(), QSizePolicy.Fixed)
        self.setContentsMargins(0, 0, 0, 0)

    def setLegendLevels(self, t_low, t_blue, t_green, t_yellow, t_high):
        self.t_low = t_low
        self.t_blue = t_blue
        self.t_green = t_green
        self.t_yellow = t_yellow
        self.t_high = t_high
        self.update()

    def paintEvent(self, event):
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing, True)
            full_rect = self.rect()
            margin_left = 20
            margin_right = 20
            margin_top = 10
            margin_bottom = 30
            gradient_rect = full_rect.adjusted(margin_left, margin_top, -margin_right, -margin_bottom)
            painter.fillRect(full_rect, Qt.white)
            total_range = self.t_high - self.t_low
            if total_range == 0:
                total_range = 1
            gradient = QLinearGradient(gradient_rect.topLeft(), gradient_rect.topRight())
            gradient.setColorAt(0.0, QColor(255, 255, 255))
            gradient.setColorAt((self.t_blue - self.t_low) / total_range, QColor(0, 0, 255))
            gradient.setColorAt((self.t_green - self.t_low) / total_range, QColor(0, 255, 0))
            gradient.setColorAt((self.t_yellow - self.t_low) / total_range, QColor(255, 255, 0))
            gradient.setColorAt(1.0, QColor(255, 0, 0))
            painter.setPen(Qt.NoPen)
            painter.setBrush(gradient)
            painter.drawRect(gradient_rect)
            painter.setPen(QPen(Qt.black, 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(gradient_rect)
            font = QFont("Arial", 9)
            painter.setFont(font)
            fm = QFontMetrics(font)
            ticks = [
                (self.t_low, f"{self.t_low:.1f}°C"),
                (self.t_blue, f"{self.t_blue:.1f}°C"),
                (self.t_green, f"{self.t_green:.1f}°C"),
                (self.t_yellow, f"{self.t_yellow:.1f}°C"),
                (self.t_high, f"{self.t_high:.1f}°C")
            ]
            for temp, label in ticks:
                pos = (temp - self.t_low) / total_range
                x = gradient_rect.left() + pos * gradient_rect.width()
                painter.drawLine(int(x), gradient_rect.bottom(), int(x), gradient_rect.bottom() + 5)
                text_width = fm.horizontalAdvance(label)
                text_rect = QRect(int(x - text_width / 2), gradient_rect.bottom() + 8, text_width, 20)
                painter.drawText(text_rect, Qt.AlignCenter, label)
            painter.end()
        except Exception as ex:
            print("Exception in LegendWidget.paintEvent:", ex)




class ImageView(QWidget):
    pixelClicked = pyqtSignal(int, int)
    mouseMoved = pyqtSignal(int, int)
    cropSelected = pyqtSignal(QRect)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.pixmap = None
        self.base_scale = 1.0
        self.zoom_factor = 1.0
        self.rotation_angle = 0.0
        self.pan_offset = QPoint(0, 0)
        self.interaction_mode = "select"
        self.last_mouse_pos = None
        self._cached_pixmap = None
        self._cached_zoom_factor = None
        self._cached_rotation_angle = None
        self.sel_start = None
        self.sel_end = None
        self.marked_pixel = None
        self.last_mouse_image_coord = None

    def setPixmap(self, pixmap):
        self.pixmap = pixmap
        self.zoom_factor = 1.0
        self.rotation_angle = 0.0
        self.pan_offset = QPoint(0, 0)
        self._cached_pixmap = None
        self._cached_zoom_factor = None
        self._cached_rotation_angle = None
        self.sel_start = None
        self.sel_end = None
        self.marked_pixel = None
        self.last_mouse_image_coord = None
        self.computeBaseScale()
        self.update()

    def computeBaseScale(self):
        if self.pixmap:
            w, h = self.pixmap.width(), self.pixmap.height()
            area_w, area_h = self.width(), self.height()
            self.base_scale = min(area_w / w, area_h / h) if (w and h) else 1.0

    def _updateCachedPixmap(self):
        if self.pixmap is None:
            return
        if (self._cached_pixmap is None or
                self._cached_zoom_factor != self.zoom_factor or
                self._cached_rotation_angle != self.rotation_angle):
            transform = QTransform().rotate(self.rotation_angle)
            rotated = self.pixmap.transformed(transform, Qt.SmoothTransformation)
            scale = self.base_scale * self.zoom_factor
            new_width = int(rotated.width() * scale)
            new_height = int(rotated.height() * scale)
            self._cached_pixmap = rotated.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._cached_zoom_factor = self.zoom_factor
            self._cached_rotation_angle = self.rotation_angle

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), Qt.black)
        if self.pixmap:
            self._updateCachedPixmap()
            if self._cached_pixmap is None:
                return
            cached = self._cached_pixmap
            max_offset_x = (cached.width() - self.width()) / 2 if cached.width() > self.width() else 0
            max_offset_y = (cached.height() - self.height()) / 2 if cached.height() > self.height() else 0
            clamped_x = max(-max_offset_x, min(self.pan_offset.x(), max_offset_x))
            clamped_y = max(-max_offset_y, min(self.pan_offset.y(), max_offset_y))
            self.pan_offset = QPoint(int(clamped_x), int(clamped_y))
            x = (self.width() - cached.width()) // 2 + self.pan_offset.x()
            y = (self.height() - cached.height()) // 2 + self.pan_offset.y()
            painter.drawPixmap(x, y, cached)

            if self.interaction_mode == "crop" and self.sel_start and self.sel_end:
                rect = self._getSelectionRect()
                if not rect.isNull():
                    painter.setPen(Qt.red)
                    painter.drawRect(rect)

            if self.marked_pixel and self.pixmap:
                orig_x, orig_y = self.marked_pixel
                scale = self.base_scale * self.zoom_factor
                transform = QTransform().rotate(self.rotation_angle)
                rotated = self.pixmap.transformed(transform, Qt.SmoothTransformation)
                drawn_w = int(rotated.width() * scale)
                drawn_h = int(rotated.height() * scale)
                offset_x = (self.width() - drawn_w) // 2 + self.pan_offset.x()
                offset_y = (self.height() - drawn_h) // 2 + self.pan_offset.y()
                x_image = int(orig_x * scale)
                y_image = int(orig_y * scale)
                widget_x = offset_x + x_image
                widget_y = offset_y + y_image
                pen = QPen(Qt.red)
                pen.setWidth(2)
                painter.setPen(pen)
                painter.setBrush(Qt.red)
                radius = 5
                painter.drawEllipse(widget_x - radius, widget_y - radius, radius * 2, radius * 2)

            if self.interaction_mode == "select" and self.last_mouse_image_coord is not None:
                orig_x, orig_y = self.last_mouse_image_coord
                scale = self.base_scale * self.zoom_factor
                transform = QTransform().rotate(self.rotation_angle)
                rotated = self.pixmap.transformed(transform, Qt.SmoothTransformation)
                drawn_w = int(rotated.width() * scale)
                drawn_h = int(rotated.height() * scale)
                offset_x = (self.width() - drawn_w) // 2 + self.pan_offset.x()
                offset_y = (self.height() - drawn_h) // 2 + self.pan_offset.y()
                widget_x = offset_x + int(orig_x * scale)
                widget_y = offset_y + int(orig_y * scale)
                pen = QPen(Qt.green)
                pen.setWidth(1)
                painter.setPen(pen)
                painter.drawLine(offset_x, widget_y, offset_x + drawn_w, widget_y)
                painter.drawLine(widget_x, offset_y, widget_x, offset_y + drawn_h)
                painter.drawText(widget_x + 5, widget_y - 5, f"({orig_x}, {orig_y})")

    def _getSelectionRect(self):
        if self.sel_start and self.sel_end:
            return QRect(self.sel_start, self.sel_end).normalized()
        return QRect()

    def resizeEvent(self, event):
        self.computeBaseScale()
        self._cached_pixmap = None
        self.update()
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if self.interaction_mode == "crop":
            self.sel_start = event.pos()
            self.sel_end = event.pos()
        else:
            if event.buttons() & Qt.LeftButton:
                self.last_mouse_pos = event.pos()
        event.accept()

    def mouseMoveEvent(self, event):
        if self.interaction_mode == "crop":
            self.sel_end = event.pos()
            self.update()
        else:
            if self.interaction_mode == "pan":
                if event.buttons() & Qt.LeftButton:
                    delta = event.pos() - self.last_mouse_pos
                    self.pan_offset += delta
                    self.last_mouse_pos = event.pos()
                    self.update()
                else:
                    self.last_mouse_pos = event.pos()
            elif self.interaction_mode == "rotate":
                if event.buttons() & Qt.LeftButton:
                    delta = event.pos() - self.last_mouse_pos
                    rotation_change = delta.x() * 0.5
                    self.rotation_angle = (self.rotation_angle + rotation_change) % 360
                    self.last_mouse_pos = event.pos()
                    self._cached_pixmap = None
                    self.update()
                else:
                    self.last_mouse_pos = event.pos()
            elif self.interaction_mode == "select" and self.pixmap:
                scale = self.base_scale * self.zoom_factor
                transform = QTransform().rotate(self.rotation_angle)
                rotated = self.pixmap.transformed(transform, Qt.SmoothTransformation)
                drawn_w = int(rotated.width() * scale)
                drawn_h = int(rotated.height() * scale)
                offset_x = (self.width() - drawn_w) // 2 + self.pan_offset.x()
                offset_y = (self.height() - drawn_h) // 2 + self.pan_offset.y()
                rel = event.pos() - QPoint(offset_x, offset_y)
                if 0 <= rel.x() < drawn_w and 0 <= rel.y() < drawn_h:
                    orig_x = int(rel.x() / scale)
                    orig_y = int(rel.y() / scale)
                    self.last_mouse_image_coord = (orig_x, orig_y)
                    self.mouseMoved.emit(orig_x, orig_y)
                else:
                    self.last_mouse_image_coord = None
                self.last_mouse_pos = event.pos()
                self.update()
            else:
                self.last_mouse_pos = event.pos()
        event.accept()

    def mouseReleaseEvent(self, event):
        if self.interaction_mode == "crop" and self.sel_start and self.sel_end:
            rect = self._getSelectionRect()
            self.cropSelected.emit(rect)
            self.sel_start = None
            self.sel_end = None
            self.interaction_mode = "select"
        elif self.interaction_mode == "select" and self.pixmap:
            scale = self.base_scale * self.zoom_factor
            transform = QTransform().rotate(self.rotation_angle)
            rotated = self.pixmap.transformed(transform, Qt.SmoothTransformation)
            drawn_w = int(rotated.width() * scale)
            drawn_h = int(rotated.height() * scale)
            offset_x = (self.width() - drawn_w) // 2 + self.pan_offset.x()
            offset_y = (self.height() - drawn_h) // 2 + self.pan_offset.y()
            rel = event.pos() - QPoint(offset_x, offset_y)
            if 0 <= rel.x() < drawn_w and 0 <= rel.y() < drawn_h:
                orig_x = int(rel.x() / scale)
                orig_y = int(rel.y() / scale)
                if 0 <= orig_x < self.pixmap.width() and 0 <= orig_y < self.pixmap.height():
                    self.marked_pixel = (orig_x, orig_y)
                    self.pixelClicked.emit(orig_x, orig_y)
        else:
            self.last_mouse_pos = None
        event.accept()

    def setInteractionMode(self, mode):
        if mode in ["select", "pan", "rotate", "crop"]:
            self.interaction_mode = mode
            if mode == "crop":
                self.sel_start = None
                self.sel_end = None

    def zoomIn(self):
        self.zoom_factor *= 1.25
        if self.zoom_factor > 60:
            self.zoom_factor = 60
        self._cached_pixmap = None
        self.update()

    def zoomOut(self):
        self.zoom_factor /= 1.25
        if self.zoom_factor < 0.1:
            self.zoom_factor = 0.1
        self._cached_pixmap = None
        self.update()


from PyQt5.QtWidgets import QSizePolicy

class ImageDisplayWidget(QWidget):
    pixelClicked = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        band_layout = QHBoxLayout()
        self.bandSlider = QSlider(Qt.Horizontal)
        self.bandSlider.setMinimum(0)
        self.bandSlider.setSingleStep(1)
        self.bandSlider.setTickPosition(QSlider.TicksBelow)
        self.bandSlider.valueChanged.connect(self.onBandChange)
        self.bandLabel = QLabel("Band: 1")
        band_layout.addWidget(self.bandSlider)
        band_layout.addWidget(self.bandLabel)
        layout.addLayout(band_layout)

        self.temperatureLabel = QLabel("Temperature: N/A")
        layout.addWidget(self.temperatureLabel)

        # Legend widget for LST images.
        self.legendWidget = LegendWidget()
        self.legendWidget.setVisible(False)
        layout.addWidget(self.legendWidget)

        # Add a fixed spacing to separate the legend from the image.
        layout.addSpacing(5)

        self.imageView = ImageView(self)
        self.imageView.setMinimumSize(200, 200)
        self.imageView.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.imageView, 1)

        self.imageView.pixelClicked.connect(lambda x, y: self.pixelClicked.emit(x, y))
        self.imageView.cropSelected.connect(self.cropImage)
        self.imageView.mouseMoved.connect(self.updateTemperatureDisplay)

        self.image_loader = None
        self.saved_path = None
        self.current_band_index = 0
        self.undo_stack = []
        self.redo_stack = []

    def setContentMode(self):
        if self.image_loader:
            if self.image_loader.num_bands == 1:
                self.bandSlider.setEnabled(False)
            else:
                self.bandSlider.setEnabled(True)
                self.bandSlider.setMinimum(0)
                self.bandSlider.setMaximum(self.image_loader.num_bands - 1)
                self.bandSlider.setSingleStep(1)
        self._updateBandLabel()
        self.displayBand(self.current_band_index)
        if self.imageView.pixmap:
            self.undo_stack = [self.imageView.pixmap.copy()]
            self.redo_stack = []
        # Show the legend if the metadata indicates that this is an LST image.
        if self.image_loader and (
                "LST Calculation" in self.image_loader.metadata or "LST Timelapse" in self.image_loader.metadata):
            self.legendWidget.setVisible(True)
        else:
            self.legendWidget.setVisible(False)

    def loadImage(self, image_loader):
        self.image_loader = image_loader
        data = self.image_loader.load()
        self.saved_path = self.image_loader.filepath
        self.undo_stack = []
        self.redo_stack = []
        if self.image_loader.num_bands > 1:
            self.bandSlider.setEnabled(True)
            self.bandSlider.setMaximum(self.image_loader.num_bands - 1)
        else:
            self.bandSlider.setEnabled(False)
        self.current_band_index = 0
        self._updateBandLabel()
        self.displayBand(self.current_band_index)
        if self.imageView.pixmap:
            self.undo_stack.append(self.imageView.pixmap.copy())
        if self.image_loader and "LST Calculation" in self.image_loader.metadata:
            self.legendWidget.setVisible(True)
        else:
            self.legendWidget.setVisible(False)

    def _updateBandLabel(self):
        # Show “Frame: X (YYYY-MM)” instead of just band number
        idx = self.current_band_index
        label = f"Frame: {idx + 1}"
        dates = self.image_loader.metadata.get("frame_dates", [])
        if 0 <= idx < len(dates) and dates[idx]:
            label += f" ({dates[idx]})"
        self.bandLabel.setText(label)

    def onBandChange(self, value):
        self.current_band_index = value
        self._updateBandLabel()
        self.displayBand(value)

    def numpy_to_qimage(self, arr):
        arr = np.nan_to_num(arr, nan=0.0)
        # If the array has more than 2 dimensions, squeeze it.
        if arr.ndim > 2:
            arr = np.squeeze(arr)
        if arr.dtype != np.uint8:
            with np.errstate(divide='ignore', invalid='ignore'):
                arr = (255 * ((arr - np.min(arr)) / (np.ptp(arr) + 1e-8))).astype(np.uint8)
        try:
            height, width = arr.shape  # This now expects arr to be 2D.
        except ValueError as e:
            raise ValueError(f"Expected a 2D array for QImage conversion, got shape {arr.shape}") from e
        bytes_per_line = width
        qimg = QImage(arr.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        return qimg.copy()

    def numpy_to_qimage_color(self, arr):
        import numpy as np
        from PyQt5.QtGui import QImage

        # 1) Cast to float and ignore NaNs
        arr = np.array(arr, dtype=np.float32)

        if np.nanmax(arr) > 150:
            arr = arr - 273.15

        # 2) Grab metadata
        meta = self.image_loader.metadata or {}
        mode = meta.get("color_mapping", "dynamic")  # "fixed" or "dynamic"

        # 3) Decide range in °C
        if mode == "fixed":
            # User‐entered fixed range (already in °C)
            dynamic_min = float(meta.get("fixed_min", 0.0))
            dynamic_max = float(meta.get("fixed_max", 1.0))
        else:
            # Dynamic
            if "LST Timelapse" in meta:
                # Timelapse: use stored global range (must be in °C)
                dynamic_min = float(meta.get("global_min", np.nanmin(arr)))
                dynamic_max = float(meta.get("global_max", np.nanmax(arr)))
            else:
                # Single‐image: use this band’s own min/max
                dynamic_min = float(np.nanmin(arr))
                dynamic_max = float(np.nanmax(arr))

        # 4) Fallback if bad
        if np.isnan(dynamic_min) or np.isnan(dynamic_max) or dynamic_max <= dynamic_min:
            dynamic_min = float(np.nanmin(arr))
            dynamic_max = float(np.nanmax(arr))
            if dynamic_max <= dynamic_min:
                dynamic_min, dynamic_max = 0.0, 1.0

        # 5) Compute legend thresholds
        t0 = dynamic_min
        t1 = dynamic_min + 0.25 * (dynamic_max - dynamic_min)
        t2 = dynamic_min + 0.50 * (dynamic_max - dynamic_min)
        t3 = dynamic_min + 0.75 * (dynamic_max - dynamic_min)
        t4 = dynamic_max

        # 6) Update legend
        if self.legendWidget:
            self.legendWidget.setLegendLevels(t0, t1, t2, t3, t4)

        # 7) Clip data to range
        arr = np.clip(arr, dynamic_min, dynamic_max)

        # 8) Map to color
        rgba = map_temperature_to_color_dynamic(arr, min_temp=dynamic_min, max_temp=dynamic_max)

        # 9) Build QImage
        h, w, c = rgba.shape
        bytes_per_line = w * c
        qimg = QImage(rgba.data, w, h, bytes_per_line, QImage.Format_RGBA8888)
        return qimg.copy()

    def displayBand(self, band_index):
        if self.image_loader is None:
            return
        try:
            # Use the LST color conversion if metadata indicates an LST image,
            # regardless of the number of bands.
            if ("LST Calculation" in self.image_loader.metadata or
                    "LST Timelapse" in self.image_loader.metadata):
                qimg = self.numpy_to_qimage_color(self.image_loader.get_band(band_index))
            else:
                qimg = self.numpy_to_qimage(self.image_loader.get_band(band_index))
        except Exception as conv_err:
            print("Error during image conversion:", conv_err)
            return

        if qimg is None:
            return
        pixmap = QPixmap.fromImage(qimg)
        self.imageView.setPixmap(pixmap)
        # Reset the undo/redo stacks (optional)
        self.undo_stack = [pixmap.copy()]
        self.redo_stack = []

    def updateTemperatureDisplay(self, orig_x, orig_y):
        """
        Update the temperatureLabel based on the pixel under the cursor.
        Works for both standard single-band LST images and timelapse (multiband) images.
        """
        # Check for LST metadata regardless of number of bands.
        if (self.image_loader is None or self.image_loader.image_data is None or
                ("LST Calculation" not in self.image_loader.metadata and
                 "LST Timelapse" not in self.image_loader.metadata)):
            self.temperatureLabel.setText("Temperature: N/A")
            return

        try:
            band = self.image_loader.get_band(self.current_band_index)
        except Exception as e:
            print("Error extracting band:", e)
            self.temperatureLabel.setText("Temperature: N/A")
            return

        # Convert band data to float32
        data = band.astype(np.float32)
        # Assume values >150 are in Kelvin; convert to Celsius.
        if np.nanmax(data) > 150:
            data = data - 273.15

        try:
            h, w = data.shape  # Expect a 2D array.
        except Exception as shape_err:
            print("Error getting shape from band data:", shape_err)
            self.temperatureLabel.setText("Temperature: N/A")
            return

        if 0 <= orig_x < w and 0 <= orig_y < h:
            temp_value = data[orig_y, orig_x]
            self.temperatureLabel.setText(f"Temperature: {temp_value:.2f} °C")
        else:
            self.temperatureLabel.setText("Temperature: N/A")

    def cropImage(self, widget_rect):
        if not self.imageView.pixmap:
            return
        try:
            # Compute widget→image transform
            widget_width = self.imageView.width()
            widget_height = self.imageView.height()
            effective_scale = self.imageView.base_scale * self.imageView.zoom_factor
            disp_tl = QPointF(
                (widget_width - self.imageView.pixmap.width() * effective_scale) / 2,
                (widget_height - self.imageView.pixmap.height() * effective_scale) / 2
            ) + QPointF(self.imageView.pan_offset)

            sel_rect = QRectF(widget_rect).normalized()
            # Convert widget coords → image coords
            x1 = int((sel_rect.left() - disp_tl.x()) / effective_scale)
            y1 = int((sel_rect.top() - disp_tl.y()) / effective_scale)
            x2 = int((sel_rect.right() - disp_tl.x()) / effective_scale)
            y2 = int((sel_rect.bottom() - disp_tl.y()) / effective_scale)

            # Clip to image bounds
            crop_rect = QRect(x1, y1, x2 - x1, y2 - y1).intersected(self.imageView.pixmap.rect())
            if crop_rect.isNull() or crop_rect.width() <= 0 or crop_rect.height() <= 0:
                return

            # Save current pixmap for undo
            self.undo_stack.append(self.imageView.pixmap.copy())
            self.redo_stack = []

            # Crop the displayed pixmap
            new_pixmap = self.imageView.pixmap.copy(crop_rect)
            self.imageView.setPixmap(new_pixmap)

            # --- NEW: also crop the underlying numpy data and record it ---
            if self.image_loader is not None and self.image_loader.image_data is not None:
                x, y, w, h = crop_rect.x(), crop_rect.y(), crop_rect.width(), crop_rect.height()
                data = self.image_loader.image_data
                if data.ndim == 2:
                    new_data = data[y:y + h, x:x + w]
                else:
                    new_data = data[y:y + h, x:x + w, :]
                self.image_loader.image_data = new_data

                # **This line** records the crop window for later LST calculations:
                self.image_loader.metadata['crop_rect'] = (x, y, w, h)

        except Exception as e:
            print("Error in cropImage:", e)

    def undo(self):
        if len(self.undo_stack) > 1:
            current = self.undo_stack.pop()
            self.redo_stack.append(current)
            new_state = self.undo_stack[-1]
            self.imageView.setPixmap(new_state)
        else:
            print("Nothing to undo.")

    def redo(self):
        if self.redo_stack:
            state = self.redo_stack.pop()
            self.undo_stack.append(state)
            self.imageView.setPixmap(state)
        else:
            print("Nothing to redo.")

    def displayColorComposite(self):
        if self.image_loader is None:
            return
        data = self.image_loader.image_data
        if data is None:
            return
        if data.ndim < 3 or data.shape[2] < 3:
            QMessageBox.warning(self, "Insufficient Bands",
                                "The image does not have enough bands for a color composite.")
            return
        wavelengths = self.image_loader.get_wavelengths()
        if wavelengths is None or len(wavelengths) < 3:
            QMessageBox.warning(self, "No Wavelength Metadata", "No or insufficient wavelength metadata available.")
            return
        wavelengths = np.array(wavelengths, dtype=np.float32)
        targets = {"red": 650, "green": 550, "blue": 450}
        tolerance = 20
        channels = {}
        for color, target in targets.items():
            indices = np.where((wavelengths >= target - tolerance) & (wavelengths <= target + tolerance))[0]
            if indices.size == 0:
                differences = np.abs(wavelengths - target)
                indices = np.array([differences.argmin()])
            channels[color] = np.mean(data[:, :, indices], axis=2)
        rgb = np.stack([channels["red"], channels["green"], channels["blue"]], axis=-1)
        rgb_normalized = np.zeros_like(rgb, dtype=np.uint8)
        for i in range(3):
            channel = rgb[..., i].astype(np.float32)
            min_val = channel.min()
            ptp_val = np.ptp(channel) + 1e-8
            norm = ((channel - min_val) / ptp_val * 255).astype(np.uint8)
            rgb_normalized[..., i] = norm
        height, width, _ = rgb_normalized.shape
        bytes_per_line = 3 * width
        qimg = QImage(rgb_normalized.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)
        self.imageView.setPixmap(pixmap)
        self.undo_stack.append(pixmap.copy())
        self.redo_stack = []

    def displayLandsatColorComposite(self):
        if self.image_loader is None:
            return

        data = self.image_loader.image_data
        if data is None:
            return

        metadata = getattr(self.image_loader, "metadata", {})
        sc = metadata.get("SPACECRAFT_ID", "").upper()

        if data.ndim < 3 or data.shape[2] < 4:
            QMessageBox.warning(self, "Insufficient Bands",
                                "The image does not have enough bands for a Landsat color composite.")
            return

        # Select bands based on Landsat version
        if sc in ("LANDSAT_8", "LANDSAT_9"):
            # Use bands 1,2,3 (indices 0,1,2)
            red = data[:, :, 3].astype(np.float32)
            green = data[:, :, 2].astype(np.float32)
            blue = data[:, :, 1].astype(np.float32)
        else:
            # Use bands 3,2,1 (indices 2,1,0)
            red = data[:, :, 2].astype(np.float32)
            green = data[:, :, 1].astype(np.float32)
            blue = data[:, :, 0].astype(np.float32)

        def normalize(arr):
            arr_min = arr.min()
            arr_max = arr.max()
            return ((arr - arr_min) / (arr_max - arr_min + 1e-8) * 255).astype(np.uint8)

        red_norm = normalize(red)
        green_norm = normalize(green)
        blue_norm = normalize(blue)

        rgb = np.stack([red_norm, green_norm, blue_norm], axis=-1)
        height, width, _ = rgb.shape
        bytes_per_line = 3 * width
        qimg = QImage(rgb.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)

        self.imageView.setPixmap(pixmap)
        self.undo_stack.append(pixmap.copy())
        self.redo_stack = []