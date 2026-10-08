from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QToolBar,
    QAction, QTabWidget, QFileDialog, QMessageBox, QPushButton, QInputDialog
)
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QIcon, QImage, QPixmap
from image_view import ImageDisplayWidget
from dialogs import MetadataDialog, PixelSpectrumDialog, StartupDialog
from CNMF import CNMF_fusion, vca, vd, estR, gaussian_down_sample, zoom_nn, lsqnonneg
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QInputDialog, QMessageBox, QSlider
from full_ref import psnr, rmse, ergas, uqi, sam
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from NMF import LSMU, HALS, NNLS
import subprocess
import os
import numpy as np
from srf_definitions import *


try:
    import cv2
except ImportError:
    cv2 = None
try:
    from scipy.signal import convolve2d
except ImportError:
    convolve2d = None
try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None



# ------------------ Raw Save Helper Functions ------------------
def _get_envi_data_type_code(dtype):
    if dtype == np.uint8:
        return 1
    elif dtype == np.int16:
        return 2
    elif dtype == np.int32:
        return 3
    elif dtype == np.float32:
        return 4
    elif dtype == np.float64:
        return 5
    elif dtype == np.uint16:
        return 12
    elif dtype == np.uint32:
        return 13
    else:
        return 4


def _get_byte_order():
    # Little-endian systems use 0.
    return 0


def _write_envi_header(header_filename, samples, lines, bands, data_type, wavelengths):
    header_text = (
        "ENVI\n"
        "description = {Saved by HyperView}\n"
        f"samples = {samples}\n"
        f"lines   = {lines}\n"
        f"bands   = {bands}\n"
        "header offset = 0\n"
        "file type = ENVI Standard\n"
        f"data type = {data_type}\n"
        "interleave = bsq\n"
        f"byte order = {_get_byte_order()}\n"
    )
    if wavelengths is not None and len(wavelengths) == bands:
        wl_str = ", ".join(f"{wl:.2f}" for wl in wavelengths)
        header_text += f"wavelength = {{{wl_str}}}\n"
    with open(header_filename, "w") as hdr_file:
        hdr_file.write(header_text)


def _saveRawImageWithHeader(image_data, raw_path, metadata):
    # Determine image dimensions.
    if image_data.ndim == 2:
        lines, samples = image_data.shape
        bands = 1
    elif image_data.ndim == 3:
        lines, samples, bands = image_data.shape
    else:
        raise ValueError("Image data has unsupported dimensions.")
    data_type_code = _get_envi_data_type_code(image_data.dtype)
    wavelengths = None
    if metadata and "wavelength" in metadata:
        try:
            wl = metadata["wavelength"]
            if isinstance(wl, (list, tuple)):
                wavelengths = [float(val) for val in wl]
            elif isinstance(wl, str):
                wl = wl.strip("{}")
                wavelengths = [float(val.strip()) for val in wl.split(",")]
        except Exception:
            wavelengths = None
    header_path = raw_path + ".hdr"
    _write_envi_header(header_path, samples, lines, bands, data_type_code, wavelengths)
    if bands == 1:
        bsq_data = image_data
    else:
        bsq_data = image_data.transpose(2, 0, 1)
    bsq_data.tofile(raw_path)

def gaussian_kernel(size, sigma):
    """Generate a 2D Gaussian kernel of given size and sigma."""
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return kernel / np.sum(kernel)


def gaussian_down_sample(data, w, mask=0):
    # Assume no mask (mask=0)
    xdata, ydata, band = data.shape
    hx = int(np.floor(xdata / w))
    hy = int(np.floor(ydata / w))
    HSI = np.zeros((hx, hy, band), dtype=data.dtype)
    sig = w / 2.35482
    H1 = gaussian_kernel(w, sig).reshape(w, w, 1)
    if w % 2 == 0:
        H2 = gaussian_kernel(w * 2, sig).reshape(w * 2, w * 2, 1)
        for x in range(hx):
            for y in range(hy):
                if x == 0 or x == hx - 1 or y == 0 or y == hy - 1:
                    patch = data[x * w:(x + 1) * w, y * w:(y + 1) * w, :]
                    HSI[x, y, :] = (patch.astype(np.float64) * np.tile(H1, (1, 1, band))).sum(axis=(0, 1))
                else:
                    patch = data[int(x * w - w // 2): int((x + 1) * w + w // 2),
                            int(y * w - w // 2): int((y + 1) * w + w // 2), :]
                    HSI[x, y, :] = (patch.astype(np.float64) * np.tile(H2, (1, 1, band))).sum(axis=(0, 1))
    else:
        H2 = gaussian_kernel(w * 2 - 1, sig).reshape(w * 2 - 1, w * 2 - 1, 1)
        for x in range(hx):
            for y in range(hy):
                if x == 0 or x == hx - 1 or y == 0 or y == hy - 1:
                    patch = data[x * w:(x + 1) * w, y * w:(y + 1) * w, :]
                    HSI[x, y, :] = (patch.astype(np.float64) * np.tile(H1, (1, 1, band))).sum(axis=(0, 1))
                else:
                    patch = data[int(x * w - (w - 1) / 2): int((x + 1) * w + (w - 1) / 2),
                            int(y * w - (w - 1) / 2): int((y + 1) * w + (w - 1) / 2), :]
                    HSI[x, y, :] = (patch.astype(np.float64) * np.tile(H2, (1, 1, band))).sum(axis=(0, 1))
    return HSI


# ---------------- End Gaussian Downsampling Functions ----------------



class FusedImageViewer(QWidget):
    def __init__(self, fused_image):
        super().__init__()
        self.fused_image = fused_image
        self.num_bands = fused_image.shape[2]

        # Layout
        self.layout = QVBoxLayout()

        # Matplotlib Figure
        self.figure, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        # Slider for band selection
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.num_bands - 1)
        self.slider.setValue(0)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self.update_band)
        self.layout.addWidget(self.slider)

        # Label for selected band
        self.label = QLabel(f"Selected Band: 0")
        self.layout.addWidget(self.label)

        self.setLayout(self.layout)

        # Show the initial band
        self.update_band(0)

    def update_band(self, band):
        """Updates the displayed band image dynamically."""
        self.label.setText(f"Selected Band: {band}")
        self.ax.clear()
        self.ax.imshow(self.fused_image[:, :, band], cmap="gray")
        self.ax.set_title(f"Fused Image - Band {band}")
        self.ax.axis("off")
        self.canvas.draw()

class MainWindow(QMainWindow):
    def __init__(self, file_to_open=None):
        super().__init__()
        self.setWindowTitle("HyperView")
        self.setWindowIcon(QIcon("hyperview_icon.ico"))
        self.draw_spectrum_mode = False
        self.endmember_data = None  # To store endmember data after VCA
        self.abundances = None  # To store abundances after VCA
        self.M = 0  # To store the number of endmembers after VCA
        self.HSI = None
        self.HSI_RESHAPED = None
        self.W_hyper = None
        self.W_hals = None
        self.H_hals = None
        self.W_nmf = None
        self.H_nmf = None
        self.shape = None
        self.A_LMSU = None
        self.A_HALS = None
        self.A_NNLS = None
        self.H_NNLS = None
        self.W_NNLS = None
        self.vca_results = []  # List to store VCA results for each processed subset


        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        main_layout = QVBoxLayout(centralWidget)

        # Navigation toolbar on the left
        nav_toolbar = QToolBar("Navigation")
        nav_toolbar.setOrientation(Qt.Vertical)
        nav_toolbar.setIconSize(QSize(40, 40))
        self.addToolBar(Qt.LeftToolBarArea, nav_toolbar)
        select_action = QAction(QIcon("select.png"), "Select", self)
        pan_action = QAction(QIcon("pan.png"), "Pan", self)
        zoom_in_action = QAction(QIcon("zoom_in.png"), "Zoom In", self)
        zoom_out_action = QAction(QIcon("zoom_out.png"), "Zoom Out", self)
        rotate_action = QAction(QIcon("rotate.png"), "Rotate", self)
        nav_toolbar.addAction(select_action)
        nav_toolbar.addAction(pan_action)
        nav_toolbar.addAction(zoom_in_action)
        nav_toolbar.addAction(zoom_out_action)
        nav_toolbar.addAction(rotate_action)
        select_action.triggered.connect(lambda: self.getCurrentImageDisplay().imageView.setInteractionMode("select"))
        pan_action.triggered.connect(lambda: self.getCurrentImageDisplay().imageView.setInteractionMode("pan"))
        zoom_in_action.triggered.connect(lambda: self.getCurrentImageDisplay().imageView.zoomIn())
        zoom_out_action.triggered.connect(lambda: self.getCurrentImageDisplay().imageView.zoomOut())
        rotate_action.triggered.connect(lambda: self.getCurrentImageDisplay().imageView.setInteractionMode("rotate"))

        self.tabWidget = QTabWidget()
        self.tabWidget.setTabsClosable(True)
        self.tabWidget.tabCloseRequested.connect(self.closeTab)
        self.tabWidget.currentChanged.connect(self.onTabChanged)
        main_layout.addWidget(self.tabWidget)

        bottom_layout = QHBoxLayout()
        self.fileNameLabel = QLabel("No image loaded")
        bottom_layout.addWidget(self.fileNameLabel)
        self.drawSpectrumButton = QPushButton("Draw Pixel Spectrum")
        self.drawSpectrumButton.clicked.connect(self.activateSpectrumMode)
        bottom_layout.addWidget(self.drawSpectrumButton)
        self.showMetadataButton = QPushButton("Show Metadata")
        self.showMetadataButton.clicked.connect(self.showMetadata)
        bottom_layout.addWidget(self.showMetadataButton)

        main_layout.addLayout(bottom_layout)
        self.createMenus()

        if file_to_open:
            self.addNewTab(file_to_open)

    def runSubsetVca(self):
        if self.shape is None:
            QMessageBox.warning(self, "No Data", "Run Vca First To extract shape.")
            return
        default_x, default_y = 50, 50  # Default subset dimensions
        x, ok_x = QInputDialog.getInt(self, "Subset Dimensions", "Enter subset width (x):", default_x)
        y, ok_y = QInputDialog.getInt(self, "Subset Dimensions", "Enter subset height (y):", default_y)
        if not ok_x or not ok_y:
            return

        if x > self.shape[1] or y > self.shape[0]:
            QMessageBox.warning(self, "Error", "Subset dimensions exceed image size.")
            return

        # Choose a random starting coordinate within the image bounds
        x0 = np.random.randint(0, self.shape[1] - x + 1)
        y0 = np.random.randint(0, self.shape[0] - y + 1)
        subset = self.HSI[y0:y0 + y, x0:x0 + x, :]

        # Display the subset (first band as grayscale)
        plt.figure(figsize=(6, 6))
        plt.imshow(subset[:, :, 0], cmap='gray')
        plt.title("Selected Subset")
        plt.axis("off")
        plt.show()

        # Reshape the subset and run VCA
        R, C, B = subset.shape
        subset_reshaped = subset.reshape(R * C, B).T  # Reshape to (Bands, pixels)
        try:
            W_hyper, _ = vca(subset_reshaped, self.M)
        except Exception as e:
            QMessageBox.critical(self, "VCA Error", f"Error during VCA process:\n{e}")
            return

        # Store the VCA result
        self.vca_results.append(W_hyper)
        QMessageBox.information(self, "VCA Finished", f"VCA processed for subset {len(self.vca_results)} successfully.")

    def visualizeSubsetVcaResults(self):
        """Visualize all stored VCA results for the processed subsets in one window."""
        if not self.vca_results:
            QMessageBox.warning(self, "No VCA Results",
                                "No VCA results stored. Please process at least one subset first.")
            return

        num_results = len(self.vca_results)
        fig, axes = plt.subplots(num_results, 1, figsize=(10, 3 * num_results), sharex=True)
        if num_results == 1:
            axes = [axes]  # Ensure we have an iterable list of axes

        colors = plt.get_cmap('tab10')  # Use tab10 colormap for up to 10 distinct colors

        for idx, (ax, W_hyper) in enumerate(zip(axes, self.vca_results)):
            for i in range(W_hyper.shape[1]):
                ax.plot(W_hyper[:, i], color=colors(i % 10), label=f"Endmember {i + 1}")
            ax.set_title(f"Subset {idx + 1} - VCA Endmembers")
            ax.set_ylabel("Spectral Intensity")
            ax.legend()
            ax.grid(True)

        plt.xlabel("Spectral Bands")
        plt.tight_layout()
        plt.show()
    def addFusedImageTab(self, fused_image):
        fused_widget = FusedImageViewer(fused_image)
        tab_index = self.tabWidget.addTab(fused_widget, "Fused Image")
        self.tabWidget.setCurrentIndex(tab_index)

    def runVcaAnalysis(self):
        """Runs VCA and displays the extracted endmembers and their abundances in separate windows."""
        current = self.getCurrentImageDisplay()
        if current is None or not hasattr(current, 'image_loader') or current.image_loader is None:
            QMessageBox.warning(self, "No Image", "Load an image first before running VCA.")
            return

        try:
            HSI = current.image_loader.image_data
            if HSI is None:
                raise ValueError("No image data found.")
            default_y = 7
            M, ok_y = QInputDialog.getInt(self, "Number of Endmembers", "Enter Number of Endmembers (y):", default_y)
            if ok_y is None :
                return
            R, C, B = HSI.shape
            HSI_reshaped = HSI.reshape(R * C, B).T
            self.shape = (R, C)


            W_hyper, indices = vca(HSI_reshaped, M)

            self.endmember_data = W_hyper
            self.M = M
            self.HSI = HSI
            self.HSI_RESHAPED = HSI_reshaped
            self.W_hyper = W_hyper

            QMessageBox.information(self, "VCA Finished", "VCA successfully finished.")

        except Exception as e:
            QMessageBox.critical(self, "VCA Error", f"VCA process failed:\n{e}")

    def visualizeAllEndmembersSpectra(self):
        """Displays the spectral signatures of all endmembers combined."""
        if self.endmember_data is None:
            QMessageBox.warning(self, "No Endmembers", "Run VCA to extract endmembers before visualizing.")
            return

        # Plot spectral signatures of all endmembers together
        plt.figure(figsize=(10, 6))
        for i in range(self.M):  # Loop over all endmembers
            plt.plot(self.endmember_data[:, i], label=f"Endmember {i + 1}")

        plt.xlabel("Spectral Bands")
        plt.ylabel("Spectral Intensity")
        plt.title("Combined Spectral Signatures of All Endmembers")
        plt.legend()
        plt.grid()
        plt.show()


    def SingleHALSAbundanceMap(self):

        if self.A_HALS is None:
            QMessageBox.warning(self, "No Data", "Run HALS first.")
            return

        num_endmembers = self.A_HALS.shape[0]

        endmember_number, ok = QInputDialog.getInt(self, "Select Endmember",
                                                   f"Enter the number of the endmember (1-{num_endmembers}):",
                                                   min=1, max=num_endmembers, value=1)
        if not ok:
            return
        endmember_number -= 1

        A_reshaped = self.A_HALS.reshape(num_endmembers, self.shape[0], self.shape[1])

        plt.figure(figsize=(6, 6))
        plt.imshow(A_reshaped[endmember_number], cmap="viridis")
        plt.colorbar(label="Abundance")
        plt.title(f"Abundance Map for Endmember {endmember_number + 1}")
        plt.axis("off")
        plt.show()

    def SingleLMSEAbundanceMap(self):

        if self.A_LMSU is None:
            QMessageBox.warning(self, "No Data", "Run LMSU first.")
            return

        num_endmembers = self.A_LMSU.shape[0]

        # Ask the user to input the endmember number they want to visualize
        endmember_number, ok = QInputDialog.getInt(self, "Select Endmember",
                                                   f"Enter the number of the endmember (1-{num_endmembers}):",
                                                   min=1, max=num_endmembers, value=1)
        if not ok:
            return
        endmember_number -= 1
        A_reshaped = self.A_LMSU.reshape(num_endmembers, self.shape[0], self.shape[1])

        plt.figure(figsize=(6, 6))
        plt.imshow(A_reshaped[endmember_number], cmap="viridis")
        plt.colorbar(label="Abundance")
        plt.title(f"Abundance Map for Endmember {endmember_number + 1}")
        plt.axis("off")
        plt.show()

    def SingleNNLSAbundanceMap(self):

        if self.A_NNLS is None:
            QMessageBox.warning(self, "No Data", "Run NNLS first.")
            return

        num_endmembers = self.A_NNLS.shape[0]


        endmember_number, ok = QInputDialog.getInt(self, "Select Endmember",
                                                   f"Enter the number of the endmember (1-{num_endmembers}):",
                                                   min=1, max=num_endmembers, value=1)
        if not ok:
            return

        endmember_number -= 1

        A_reshaped = self.A_NNLS.reshape(num_endmembers, self.shape[0], self.shape[1])

        plt.figure(figsize=(6, 6))
        plt.imshow(A_reshaped[endmember_number], cmap="viridis")
        plt.colorbar(label="Abundance")
        plt.title(f"Abundance Map for Endmember {endmember_number + 1}")
        plt.axis("off")
        plt.show()

    def visualiseAbundanceLMSUMaps(self):

        if self.A_LMSU is None:
            QMessageBox.warning(self, "No Data", "Run LMSU first.")
            return
        num_endmembers = self.A_LMSU.shape[0]
        A_reshaped = self.A_LMSU.reshape(num_endmembers, self.shape[0], self.shape[1])  # Reshape to 2D

        fig, axes = plt.subplots(1, num_endmembers, figsize=(15, 5))
        for i in range(num_endmembers):
            ax = axes[i] if num_endmembers > 1 else axes
            ax.imshow(A_reshaped[i], cmap="viridis")
            ax.set_title(f"Map {i + 1}")
            ax.axis("off")

        plt.show()

    def visualiseAbundanceNNLSMaps(self):

        if self.A_NNLS is None:
            QMessageBox.warning(self, "No Data", "Run NNLS first.")
            return
        num_endmembers = self.A_NNLS.shape[0]
        A_reshaped = self.A_NNLS.reshape(num_endmembers, self.shape[0], self.shape[1])  # Reshape to 2D

        fig, axes = plt.subplots(1, num_endmembers, figsize=(15, 5))
        for i in range(num_endmembers):
            ax = axes[i] if num_endmembers > 1 else axes
            ax.imshow(A_reshaped[i], cmap="viridis")
            ax.set_title(f"Map {i + 1}")
            ax.axis("off")

        plt.show()

    def visualiseAbundanceHALSMaps(self):

        if self.A_HALS is None:
            QMessageBox.warning(self, "No Data", "Run HALS first.")
            return
        num_endmembers = self.A_HALS.shape[0]
        A_reshaped = self.A_HALS.reshape(num_endmembers, self.shape[0], self.shape[1])  # Reshape to 2D

        fig, axes = plt.subplots(1, num_endmembers, figsize=(15, 5))
        for i in range(num_endmembers):
            ax = axes[i] if num_endmembers > 1 else axes
            ax.imshow(A_reshaped[i], cmap="viridis")
            ax.set_title(f"Map {i + 1}")
            ax.axis("off")

        plt.show()

    def visualiseAllAbundanceMapsNormalNmf(self):

        if self.abundances is None:
            QMessageBox.warning(self, "No Abundances", "Run NMF_1 to extract abundances before visualizing.")
            return

        num_endmembers = self.abundances.shape[0]
        A_reshaped = self.abundances.reshape(num_endmembers, self.shape[0], self.shape[1])

        fig, axes = plt.subplots(1, num_endmembers, figsize=(15, 5))
        for i in range(num_endmembers):
            ax = axes[i] if num_endmembers > 1 else axes
            ax.imshow(A_reshaped[i], cmap="viridis")
            ax.set_title(f"Map {i + 1}")
            ax.axis("off")
        plt.show()

    def visualiseAbundanceMapNormalNmf(self):

        endmember_number, ok = QInputDialog.getInt(self, "Select Endmember",
                                                   f"Enter the number of the endmember (1-{self.M}):",
                                                   min=1, max=self.M, value=1)
        if not ok:
            return

        endmember_number -= 1

        if endmember_number < 0 or endmember_number >= self.M:
            QMessageBox.warning(self, "Invalid Endmember",
                                f"Invalid endmember number: {endmember_number + 1}. Please choose a valid number between 1 and {self.M}.")
            return

        if self.abundances is None:
            QMessageBox.warning(self, "No Abundances", "Run NMF_1 to extract abundances before visualizing.")
            return
        # Display abundance map for the selected endmember
        plt.figure(figsize=(6, 6))
        plt.imshow(self.abundances[endmember_number], cmap="viridis")
        plt.colorbar(label="Abundance")
        plt.title(f"Map {endmember_number + 1}")
        plt.show()


    def PlotSpectralResultLMSU(self):

        if self.W_nmf is None:
            QMessageBox.warning(self, "No Data", "Run LMSE first.")
            return
        # Plot NMF extracted endmembers
        plt.figure(figsize=(10, 6))
        for i in range(self.W_nmf.shape[1]):
            plt.plot(self.W_nmf[:, i], label=f"Endmember {i + 1}")

        plt.xlabel("Spectral Bands")
        plt.ylabel("Spectral Intensity")
        plt.title("LMSU(NMF) Extracted Endmembers")
        plt.legend()
        plt.grid()
        plt.show()

    def PlotSpectralResultHals(self):
        if self.W_hals is None:
            QMessageBox.warning(self, "No Data", "Run HALS first.")
            return
        # Plot NMF extracted endmembers
        plt.figure(figsize=(10, 6))
        for i in range(self.W_hals.shape[1]):
            plt.plot(self.W_hals[:, i], label=f"Endmember {i + 1}")

        plt.xlabel("Spectral Bands")
        plt.ylabel("Spectral Intensity")
        plt.title("HALS(NMF) Extracted Endmembers")
        plt.legend()
        plt.grid()
        plt.show()

    def HalsRunProcess(self):

        if self.W_hyper is None:
            QMessageBox.warning(self, "No Data", "Run VCA first.")
            return
        # Prepare for HALS
        bands, pixels = self.HSI_RESHAPED.shape
        A_init = np.abs(np.random.rand(self.W_hyper.shape[0], self.M) * 0.1)  # Small values
        S_init = np.abs(np.random.rand(self.M, pixels) * 0.1)

        A2S_iteraRatioVector = [3, 3]  # Reduce iterations to speed up

        # Apply HALS
        result = HALS(self.HSI_RESHAPED, A_init, S_init, 25, A2S_iteraRatioVector)  # 20 iterations

        if isinstance(result, int):
            print("Error: HALS failed due to incorrect dimensions.")
            return


        W_hals, H_hals = result  # Unpack result properly
        self.W_hals = W_hals
        self.H_hals = H_hals

        QMessageBox.information(self, "HALS Finished", "HALS successfully finished.")

        self.A_HALS = self.H_hals.reshape(self.M, self.shape[0], self.shape[1])  # Ensure correct shape

    def PlotSpectralResultNNLS(self):
        if self.W_NNLS is None:
            QMessageBox.warning(self, "No Data", "Run NNLS first.")
            return
        plt.figure(figsize=(10, 6))
        for i in range(self.W_NNLS.shape[1]):
            plt.plot(self.W_NNLS[:, i], label=f"Endmember {i + 1}")

        plt.xlabel("Spectral Bands")
        plt.ylabel("Spectral Intensity")
        plt.title("NNLS(NMF) Extracted Endmembers")
        plt.legend()
        plt.grid()
        plt.show()


    def nnlsRunProcess(self):

        if self.W_hyper is None:
            QMessageBox.warning(self, "No Data", "Run VCA first.")
            return
        bands, pixels = self.HSI_RESHAPED.shape
        A_init = np.abs(np.random.rand(self.W_hyper.shape[0], self.M) * 0.1)  # Small values
        S_init = np.abs(np.random.rand(self.M, pixels) * 0.1)

        # Apply NNLS
        result = NNLS(self.HSI_RESHAPED,  A_init, S_init, 25, 1)  # 25 iterations, update S first

        if isinstance(result, int):
            print("Error: NNLS failed due to incorrect dimensions.")
            return

        W_NNLS, H_NNLS = result  # Unpack result properly

        self.W_NNLS = W_NNLS
        self.H_NNLS = H_NNLS

        QMessageBox.information(self, "NNLS Finished", "NNLS successfully finished.")

        self.A_NNLS = self.H_NNLS.reshape(self.M, self.shape[0], self.shape[1])  # Ensure correct shape


    def LmsuRunProcess(self):

        if self.W_hyper is None:
            QMessageBox.warning(self, "No Data", "Run VCA first.")
            return
        bands, pixels = self.HSI_RESHAPED.shape

        H_init = np.random.rand(self.M, pixels)  # (endmembers, pixels)

        # Apply NMF (LSMU)
        result = LSMU(self.HSI_RESHAPED, self.W_hyper, H_init, 100)

        if isinstance(result, int):  # If LSMU returns -1 due to an error
            print("Error: LSMU failed due to incorrect dimensions.")
            return

        W_nmf, H_nmf = result  # Unpack result properly
        self.W_nmf = W_nmf
        self.H_nmf = H_nmf

        QMessageBox.information(self, "LMSU Finished", "LMSU successfully finished.")

        self.A_LMSU = self.H_nmf.reshape(self.M, self.shape[0], self.shape[1])  # Ensure correct shape

    def showSpectralSignature(self):
        """Displays the spectral signature of the selected endmember."""
        endmember_number, ok = QInputDialog.getInt(self, "Select Endmember",
                                                   f"Enter the number of the endmember (1-{self.M}):",
                                                   min=1, max=self.M, value=1)
        if not ok:
            return  # If the user cancels or doesn't select a valid number, exit the function

        # Adjust for 0-based index
        endmember_number -= 1

        # Ensure the endmember number is valid
        if endmember_number < 0 or endmember_number >= self.M:
            QMessageBox.warning(self, "Invalid Endmember",
                                f"Invalid endmember number: {endmember_number + 1}. Please choose a valid number between 1 and {self.M}.")
            return

        # Plot the spectral signature for the selected endmember
        plt.figure(figsize=(10, 6))
        plt.plot(self.endmember_data[:, endmember_number], label=f"Endmember {endmember_number + 1}")
        plt.xlabel("Spectral Bands")
        plt.ylabel("Spectral Intensity")
        plt.title(f"Spectral Signature for Endmember {endmember_number + 1}")
        plt.legend()
        plt.grid()
        plt.show()

    def getCurrentImageDisplay(self):
        widget = self.tabWidget.currentWidget()
        return widget if widget else None

    def addNewTab(self, file_path):
        widget = ImageDisplayWidget()
        try:
            from image_loader import ImageLoader
            loader = ImageLoader(file_path)
            widget.loadImage(loader)
            widget.setContentMode()
            widget.pixelClicked.connect(self.onImagePixelClicked)
            self.tabWidget.addTab(widget, os.path.basename(file_path))
            self.tabWidget.setCurrentWidget(widget)
            self.fileNameLabel.setText(file_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{e}")

    def activateSpectrumMode(self):
        self.draw_spectrum_mode = True
        QMessageBox.information(self, "Draw Spectrum", "Click on a pixel to draw its spectrum.")


    def showMetadata(self):
        current = self.getCurrentImageDisplay()
        if current and current.image_loader:
            metadata = current.image_loader.metadata
            if metadata:
                if isinstance(metadata, dict):
                    meta_str = "\n".join(f"{k}: {v}" for k, v in metadata.items())
                else:
                    meta_str = str(metadata)
            else:
                meta_str = "No metadata available."
        else:
            meta_str = "No image loaded."
        dlg = MetadataDialog(meta_str, self)
        dlg.exec_()

    def HSBSV(self):
        subprocess.Popen(["python", "test_3.py"])

    def createMenus(self):
        menuBar = self.menuBar()
        # File menu
        fileMenu = menuBar.addMenu("&File")
        openAction = QAction("Open", self)
        openAction.triggered.connect(self.openFile)
        fileMenu.addAction(openAction)
        saveAction = QAction("Save", self)
        saveAction.triggered.connect(self.saveFile)
        fileMenu.addAction(saveAction)
        saveAsAction = QAction("Save As", self)
        saveAsAction.triggered.connect(self.saveAsFile)
        fileMenu.addAction(saveAsAction)
        closeAction = QAction("Close", self)
        closeAction.triggered.connect(self.close)
        fileMenu.addAction(closeAction)
        # Edit menu
        editMenu = menuBar.addMenu("&Edit")
        undoAction = QAction("Undo", self)
        redoAction = QAction("Redo", self)
        cropAction = QAction("Crop", self)
        editMenu.addAction(undoAction)
        editMenu.addAction(redoAction)
        editMenu.addAction(cropAction)
        undoAction.triggered.connect(
            lambda: self.getCurrentImageDisplay().undo() if self.getCurrentImageDisplay() else None)
        redoAction.triggered.connect(
            lambda: self.getCurrentImageDisplay().redo() if self.getCurrentImageDisplay() else None)
        cropAction.triggered.connect(lambda: self.getCurrentImageDisplay().imageView.setInteractionMode("crop"))

        fusionMenu = menuBar.addMenu("&Fusion")
        fusionAction = QAction("Fusion Process", self)
        fusionAction.triggered.connect(self.runFusionProcess)
        fusionMenu.addAction(fusionAction)

        SpectralUnmixingMenu = menuBar.addMenu("&Spectral Unmixing")

        # Create "Vertex Components Analysis" submenu inside "Spectral Unmixing"
        VertexComponentsAnalysisMenu = SpectralUnmixingMenu.addMenu("Vertex Components Analysis")
        VertexComponentsAnalysisAction = QAction("VCA Process", self)
        VertexComponentsAnalysisAction.triggered.connect(self.runVcaAnalysis)

        # Create "NMF" submenu inside "Spectral Unmixing"
        nmfMenu = SpectralUnmixingMenu.addMenu("Non-Negative Matrix Factorization")

        # Create "LMSU" submenu inside "NMF"
        nmf_2Menu = nmfMenu.addMenu("LMSU")
        nmf_2_Action = QAction("LMSU", self)
        nmf_2_Action.triggered.connect(self.LmsuRunProcess)

        nmf_3_Action = QAction("Plot Spectral Result", self)
        nmf_3_Action.triggered.connect(self.PlotSpectralResultLMSU)  # <- Assign correct method

        nmf_4_Action = QAction("Visualize Abundance Map For Single EndMember", self)
        nmf_4_Action.triggered.connect(self.SingleLMSEAbundanceMap)  # <- Assign correct method

        nmf_6_Action = QAction("Visualize Abundance Maps", self)
        nmf_6_Action.triggered.connect(self.visualiseAbundanceLMSUMaps)  # <- Assign correct method

        HALSMenu = nmfMenu.addMenu("HALS")
        HALS_1_Action = QAction("HALS Process", self)
        HALS_1_Action.triggered.connect(self.HalsRunProcess)

        HALS_2_Action = QAction("Plot Spectral Result", self)
        HALS_2_Action.triggered.connect(self.PlotSpectralResultHals)  # <- Assign correct method

        HALS_3_Action = QAction("Visualize Abundance Map For Single EndMember", self)
        HALS_3_Action.triggered.connect(self.SingleHALSAbundanceMap)  # <- Assign correct method

        HALS_4_Action = QAction("Visualize Abundance Maps", self)
        HALS_4_Action.triggered.connect(self.visualiseAbundanceHALSMaps)  # <- Assign correct method

        HALSMenu.addAction(HALS_1_Action)  # Correct placement under "Normal NMF"
        HALSMenu.addAction(HALS_2_Action)  # Also correctly placed under "Normal NMF"
        HALSMenu.addAction(HALS_3_Action)  # Also correctly placed under "Normal NMF"
        HALSMenu.addAction(HALS_4_Action)  # Also correctly placed under "Normal NMF"

        NNLSMenu = nmfMenu.addMenu("NNLS")
        NNLS_1_Action = QAction("NNLS Process", self)
        NNLS_1_Action.triggered.connect(self.nnlsRunProcess)

        NNLS_2_Action = QAction("Plot Spectral Result", self)
        NNLS_2_Action.triggered.connect(self.PlotSpectralResultNNLS)  # <- Assign correct method

        NNLS_3_Action = QAction("Visualize Abundance Map For Single EndMember", self)
        NNLS_3_Action.triggered.connect(self.SingleNNLSAbundanceMap)  # <- Assign correct method

        NNLS_4_Action = QAction("Visualize Abundance Maps", self)
        NNLS_4_Action.triggered.connect(self.visualiseAbundanceNNLSMaps)  # <- Assign correct method

        NNLSMenu.addAction(NNLS_1_Action)  # Correct placement under "Normal NMF"
        NNLSMenu.addAction(NNLS_2_Action)  # Also correctly placed under "Normal NMF"
        NNLSMenu.addAction(NNLS_3_Action)  # Also correctly placed under "Normal NMF"
        NNLSMenu.addAction(NNLS_4_Action)  # Also correctly placed under "Normal NMF"

        # Additional Actions
        VisualizeAllEndMembersAction = QAction("Spectral of All Endmembers", self)
        VisualizeAllEndMembersAction.triggered.connect(self.visualizeAllEndmembersSpectra)

        singleEndMemberAction = QAction("Spectral of a Single Endmember", self)
        singleEndMemberAction.triggered.connect(self.showSpectralSignature)

        subSetMenu = menuBar.addMenu("&Spectral Library")

        # Action to enter subset dimensions
        enterSubsetAction = QAction("HSBSV", self)
        enterSubsetAction.triggered.connect(self.HSBSV)
        subSetMenu.addAction(enterSubsetAction)

        subsetVcaRun = QAction("Input Subset Dimension", self)
        subsetVcaRun.triggered.connect(self.runSubsetVca)
        subSetMenu.addAction(subsetVcaRun)

        subsetVcaRun = QAction("Visualize Subset Vca", self)
        subsetVcaRun.triggered.connect(self.visualizeSubsetVcaResults)
        subSetMenu.addAction(subsetVcaRun)

        # Add actions to "Vertex Components Analysis" submenu
        VertexComponentsAnalysisMenu.addAction(VertexComponentsAnalysisAction)
        VertexComponentsAnalysisMenu.addAction(VisualizeAllEndMembersAction)
        VertexComponentsAnalysisMenu.addAction(singleEndMemberAction)


        # Add actions to "NMF" submenu
        nmf_2Menu.addAction(nmf_2_Action)  # <- Now correctly inside LMSU submenu
        nmf_2Menu.addAction(nmf_3_Action)
        nmf_2Menu.addAction(nmf_6_Action)  # Also correctly placed under "Normal NMF"
        nmf_2Menu.addAction(nmf_4_Action)

        cropAction.triggered.connect(lambda: self.getCurrentImageDisplay().imageView.setInteractionMode("crop"))
        hyperspectralMenu = menuBar.addMenu("&Hyperspectral")
        transformAction = QAction("Transform to Multispectral", self)
        transformAction.triggered.connect(self.runTransformToMultispectral)
        hyperspectralMenu.addAction(transformAction)
        downsampleAction = QAction("Downsample Hyperspectral", self)
        downsampleAction.triggered.connect(self.downsampleHyperspectral)
        hyperspectralMenu.addAction(downsampleAction)
        # New action: View Color Composite
        colorCompositeAction = QAction(QIcon("color_icon.png"), "View Color Composite", self)
        colorCompositeAction.triggered.connect(self.viewColorComposite)
        hyperspectralMenu.addAction(colorCompositeAction)


        multispectralMenu = menuBar.addMenu("&Multispectral")
        calculateLSTAction = QAction("Calculate LST", self)
        calculateLSTAction.triggered.connect(self.calculateLST)
        multispectralMenu.addAction(calculateLSTAction)
        landsatcolorcompositeAction = QAction("View Color Composite", self)
        landsatcolorcompositeAction.triggered.connect(self.viewlandsatcolorcomposite)
        multispectralMenu.addAction(landsatcolorcompositeAction)

        # NEW action for timelapse
        lstTimelapseAction = QAction("Calculate LST Timelapse", self)
        lstTimelapseAction.triggered.connect(self.calculateLSTTimelapse)
        multispectralMenu.addAction(lstTimelapseAction)

    def viewlandsatcolorcomposite(self):
        current = self.getCurrentImageDisplay()
        if current and current.image_loader:
            current.displayLandsatColorComposite()
        else:
            QMessageBox.warning(self, "No Image", "There is no image loaded.")


    def openFile(self):
        filePath, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "",
            "Images (*.raw *.l1r *.tif *.tiff *.hdr *.aux *.hdf *.hdf5 *.nc *.jp2 *.img *.bil *.dat *.envi);;All Files (*)"
        )
        if filePath:
            self.addNewTab(filePath)

    def saveFile(self):
        current = self.getCurrentImageDisplay()
        if current and current.imageView.pixmap:
            if current.saved_path:
                base = os.path.splitext(current.saved_path)[0]
                try:
                    if current.image_loader and current.image_loader.image_data is not None:
                        _saveRawImageWithHeader(current.image_loader.image_data, base, current.image_loader.metadata)
                        QMessageBox.information(self, "Saved", f"Image saved as raw data with header ({base}.hdr)")
                    else:
                        QMessageBox.warning(self, "No Image Data", "No image data available for saving.")
                except Exception as e:
                    QMessageBox.warning(self, "Save Error", f"Failed to save raw image:\n{e}")
            else:
                self.saveAsFile()
        else:
            QMessageBox.warning(self, "No Image", "There is no image to save.")

    def saveAsFile(self):
        current = self.getCurrentImageDisplay()
        if current and current.imageView.pixmap:
            filePath, _ = QFileDialog.getSaveFileName(
                self, "Save Image As", "",
                "All Files (*)"
            )
            if filePath:
                base = os.path.splitext(filePath)[0]
                try:
                    if current.image_loader and current.image_loader.image_data is not None:
                        _saveRawImageWithHeader(current.image_loader.image_data, base, current.image_loader.metadata)
                        current.saved_path = base
                        QMessageBox.information(self, "Saved", f"Image saved as raw data with header ({base}.hdr)")
                    else:
                        QMessageBox.warning(self, "No Image Data", "No image data available for saving.")
                except Exception as e:
                    QMessageBox.warning(self, "Save Error", f"Failed to save raw image:\n{e}")
        else:
            QMessageBox.warning(self, "No Image", "There is no image to save.")

    def calculateLST(self):
        """
        PSWA LST, cropped (if you made a crop), with Dynamic/Fixed mapping.
        Patches both get_fmask_cloud_array and get_cwv_as_array so their
        masks are always the same shape as the Band-10 brightness array.
        """
        import os, traceback
        import numpy as np
        from PyQt5.QtWidgets import QMessageBox, QInputDialog

        try:
            # 1) Get current image & loader
            widget = self.getCurrentImageDisplay()
            if not widget or not widget.image_loader:
                QMessageBox.warning(self, "No Image", "Load and (optionally) crop a Landsat scene first.")
                return
            loader = widget.image_loader
            mtl = loader.filepath

            # 2) Ensure it's an MTL.txt
            if not mtl.lower().endswith("_mtl.txt"):
                QMessageBox.warning(self, "Invalid File", "Select a Landsat *_MTL.txt metadata file.")
                return

                # 3) Choose LSE (emissivity) mode
            lse_mode, ok = QInputDialog.getItem(
                self, "Emissivity Mode",
                "Select Land Surface Emissivity (LSE) calculation mode:",
                ["auto-ndvi-raw", "auto-ndvi-dos"], 0, False
            )
            if not ok:
                return

            # 3) Choose Dynamic vs Fixed color mapping
            mode, ok = QInputDialog.getItem(
                self, "Color Mapping", "Choose mapping mode:",
                ["Dynamic", "Fixed"], 0, False
            )
            if not ok:
                return
            fixed_min = fixed_max = None
            if mode == "Fixed":
                fixed_min, ok1 = QInputDialog.getDouble(
                    self, "Fixed Scale", "Min temperature (°C):", 0.0, -100, 1000, 2
                )
                if not ok1:
                    return
                fixed_max, ok2 = QInputDialog.getDouble(
                    self, "Fixed Scale", "Max temperature (°C):", 50.0, -100, 1000, 2
                )
                if not ok2:
                    return

            # 4) Initialize the PSWA retriever
            from L8LST_PSWA.LSTRetriever import LSTRetriever
            temp_dir = os.path.join(os.getcwd(), "lst_temp")
            os.makedirs(temp_dir, exist_ok=True)
            retriever = LSTRetriever(
                metadata_file=mtl,
                LSE_mode=lse_mode,
                temp_dir=temp_dir,
                window_size=15
            )

            # ─── Monkey‐patch PSWA internals ───────────────────────────────────────
            # a) Capture b10_bt shape
            orig_bt_func = retriever.get_b10_b11_brightness_temp_arrays

            def patched_bt():
                b10_bt, b11_bt = orig_bt_func()
                retriever._last_bt = b10_bt
                return b10_bt, b11_bt

            retriever.get_b10_b11_brightness_temp_arrays = patched_bt

            # b) Patch cloud‐mask function to zeros of correct shape
            retriever.get_fmask_cloud_array = lambda: np.zeros_like(retriever._last_bt)

            # c) Patch CWV function to zeros of correct shape
            orig_cwv = retriever.get_cwv_as_array

            def patched_cwv(cloud_arr, b10_bt, b11_bt):
                # resize cloud_arr to b10_bt shape
                rows, cols = b10_bt.shape
                ca = cloud_arr
                ca2 = np.zeros((rows, cols), dtype=ca.dtype)
                r = min(rows, ca.shape[0]);
                c = min(cols, ca.shape[1])
                ca2[:r, :c] = ca[:r, :c]
                return orig_cwv(ca2, b10_bt, b11_bt)

            retriever.get_cwv_as_array = patched_cwv

            # 5) If cropped, crop the TIFFs for band10/11
            crop = loader.metadata.get("crop_rect")
            if crop:
                x, y, w, h = crop
                from osgeo import gdal
                b10c = os.path.join(temp_dir, "crop_b10.tif")
                gdal.Translate(b10c, retriever.band_10_path,
                               srcWin=[x, y, w, h], format="GTiff")
                b11c = os.path.join(temp_dir, "crop_b11.tif")
                gdal.Translate(b11c, retriever.band_11_path,
                               srcWin=[x, y, w, h], format="GTiff")
                retriever.band_10_path, retriever.band_11_path = b10c, b11c
                # rebuild dummy cloud mask file of matching size
                maskp = os.path.join(temp_dir, "latest_cloud.tif")
                if os.path.exists(maskp):
                    os.remove(maskp)
                drv = gdal.GetDriverByName("GTiff")
                out = drv.Create(maskp, w, h, 1, gdal.GDT_Byte)
                out.GetRasterBand(1).Fill(0)
                out = None

            # 6) Run PSWA LST (Kelvin→Celsius)
            with np.errstate(divide='ignore', invalid='ignore'):
                lst_k = retriever.get_lst_array()
            if lst_k is None:
                raise RuntimeError("PSWA returned no LST array")
            finite = np.isfinite(lst_k)
            fill = np.min(lst_k[finite]) if finite.any() else 0.0
            lst_k = np.where(finite, lst_k, fill)
            lst_c = lst_k - 273.15
            if mode == "Dynamic":
                p1 = np.nanpercentile(lst_c, 1)
                p99 = np.nanpercentile(lst_c, 99)
                lst_c = np.clip(lst_c, p1, p99)

            # 7) **Optional**: mask to a polygon vector
            vec_path, _ = QFileDialog.getOpenFileName(
                self, "Mask to City Vector (optional)", "",
                "GeoJSON (*.geojson);;Shapefile (*.shp);;All Files (*)"
            )
            if vec_path:
                import fiona
                import rasterio.features
                from shapely.geometry import shape
                # load your polygon(s)
                with fiona.open(vec_path) as src:
                    geoms = [feature["geometry"] for feature in src]
                # read the transform from the (possibly cropped) band10 file
                import rasterio
                with rasterio.open(retriever.band_10_path) as src_ds:
                    transform = src_ds.transform
                # build a mask: True INSIDE your city
                mask = rasterio.features.geometry_mask(
                    geoms,
                    out_shape=lst_c.shape,
                    transform=transform,
                    invert=True
                )
                # apply the mask
                lst_c = np.where(mask, lst_c, np.nan)

            # 7) Build legend metadata
            meta = {"LST Calculation": "PSWA", "color_mapping": mode.lower()}
            if mode == "Fixed":
                meta.update({"fixed_min": fixed_min, "fixed_max": fixed_max})

            # 8) Display in a new tab
            from image_view import ImageDisplayWidget
            from image_loader import ImageLoader

            w = ImageDisplayWidget()
            loader2 = ImageLoader("")  # dummy
            loader2.image_data = lst_c.astype('float32')
            loader2.num_bands = 1
            loader2.metadata = meta

            w.image_loader = loader2
            w.setContentMode()
            w.saved_path = ""

            self.tabWidget.addTab(w, "LST_PSWA")
            self.tabWidget.setCurrentWidget(w)

            QMessageBox.information(self, "Done", "LST calculated on your cropped region!")

        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "calculateLST Error", f"{e}\n\n{tb}")
            return
    def transformToMultispectral(self, hyper, sensor):
        if sensor == "Landsat 7":
            # Use your original Landsat 7 SRF function
            srf = get_landsat7_srf()
            band_names = ["Band 1", "Band 2", "Band 3", "Band 4", "Band 5", "Band 7", "Band 8"]
            interp_kwargs = {}
        elif sensor == "Landsat 8":
            # Use the interpolation approach (just like Landsat 7).
            # Make sure you have a function that returns a dict with
            # { "Band 1": {"wavelength": [...], "response": [...]}, ... }
            # for all Landsat 8 bands.
            srf = get_landsat8_srf()  # You must define this function
            band_names = list(srf.keys())
            interp_kwargs = {}
        elif sensor == "Sentinel 2A":
            # Keep the old bounding approach as you originally had
            srf = get_sentinel2a_srf()
            band_names = list(srf.keys())
            interp_kwargs = {}
        elif sensor == "Sentinel 2B":
            # Keep the old bounding approach as you originally had
            srf = get_sentinel2b_srf()
            band_names = list(srf.keys())
            interp_kwargs = {}
        else:
            raise ValueError("Unknown sensor SRF specified.")

        hyper_wl = self.getCurrentImageDisplay().image_loader.metadata.get("wavelength", None)
        if hyper_wl is None:
            raise ValueError("No wavelength information found in image metadata.")
        hyper_wl = np.array(hyper_wl, dtype=np.float32)
        H, W, B = hyper.shape
        m = len(band_names)
        ms = np.zeros((H, W, m), dtype=np.float32)
        self.new_wavelengths = []

        for i, bn in enumerate(band_names):

            if sensor == "Landsat 7":
                # --- Landsat 7 interpolation approach (unchanged) ---
                xp = np.array(srf[bn]["wavelength"], dtype=np.float32)
                fp = np.array(srf[bn]["response"], dtype=np.float32)
                L = min(len(xp), len(fp))
                xp = xp[:L]
                fp = fp[:L]
                response = np.interp(hyper_wl, xp, fp, **interp_kwargs)
                weight_sum = np.sum(response)
                if weight_sum == 0:
                    weight_sum = 1e-6
                effective_wl = np.sum(hyper_wl * response) / weight_sum
                self.new_wavelengths.append(effective_wl)
                weighted_sum = np.zeros((H, W), dtype=np.float32)
                for j in range(B):
                    weighted_sum += hyper[:, :, j] * response[j]
                ms[:, :, i] = weighted_sum / weight_sum

            elif sensor == "Landsat 8":
                # --- Landsat 8 now uses the SAME interpolation approach as Landsat 7 ---
                xp = np.array(srf[bn]["wavelength"], dtype=np.float32)
                fp = np.array(srf[bn]["response"], dtype=np.float32)
                L = min(len(xp), len(fp))
                xp = xp[:L]
                fp = fp[:L]
                response = np.interp(hyper_wl, xp, fp, **interp_kwargs)
                weight_sum = np.sum(response)
                if weight_sum == 0:
                    weight_sum = 1e-6
                effective_wl = np.sum(hyper_wl * response) / weight_sum
                self.new_wavelengths.append(effective_wl)
                weighted_sum = np.zeros((H, W), dtype=np.float32)
                for j in range(B):
                    weighted_sum += hyper[:, :, j] * response[j]
                ms[:, :, i] = weighted_sum / weight_sum

            else:
                xp = np.array(srf[bn]["wavelength"], dtype=np.float32)
                fp = np.array(srf[bn]["response"], dtype=np.float32)
                L = min(len(xp), len(fp))
                xp = xp[:L]
                fp = fp[:L]
                response = np.interp(hyper_wl, xp, fp, **interp_kwargs)
                weight_sum = np.sum(response)
                if weight_sum == 0:
                    weight_sum = 1e-6
                effective_wl = np.sum(hyper_wl * response) / weight_sum
                self.new_wavelengths.append(effective_wl)
                weighted_sum = np.zeros((H, W), dtype=np.float32)
                for j in range(B):
                    weighted_sum += hyper[:, :, j] * response[j]
                ms[:, :, i] = weighted_sum / weight_sum

        return ms

    def onImagePixelClicked(self, x, y):
        current = self.getCurrentImageDisplay()
        if self.draw_spectrum_mode and current and current.image_loader:
            try:
                spectrum = current.image_loader.get_pixel_spectrum(x, y)
                dlg = PixelSpectrumDialog(spectrum, band_slider=current.bandSlider, parent=self)
                dlg.bandSelected.connect(lambda band: current.bandSlider.setValue(band))
                dlg.exec_()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not get pixel spectrum:\n{e}")
            finally:
                self.draw_spectrum_mode = False

    def onTabChanged(self, index):
        current = self.getCurrentImageDisplay()
        if hasattr(current, 'image_loader') and current.image_loader:
            self.fileNameLabel.setText(current.image_loader.filepath)
        else:
            self.fileNameLabel.setText("No image loaded")

    def closeTab(self, index):
        if self.tabWidget.count() > 1:
            self.tabWidget.removeTab(index)
        else:
            QMessageBox.warning(self, "Warning", "At least one tab must remain open.")

    # --- Fusion Process Methods ---

    def runFusionProcess(self):
        # Select a reference image from the open tabs.
        if self.tabWidget.count() == 0:
            QMessageBox.warning(self, "No Images", "No images are open.")
            return

        items = [self.tabWidget.tabText(i) for i in range(self.tabWidget.count())]
        item_1, ok_1= QInputDialog.getItem(self, "Select MSI Image",
                                        "Choose MSI Images for fusion:", items, 0, False)
        if not ok_1 or not item_1:
            return
        msi_widget = None
        for i in range(self.tabWidget.count()):
            if self.tabWidget.tabText(i) == item_1:
                msi_widget = self.tabWidget.widget(i)
                break
        if msi_widget is None:
            QMessageBox.warning(self, "Selection Error", "MSI image not found.")
            return

        item_2, ok_2 = QInputDialog.getItem(self, "Select LOW_HSI Image",
                                        "Choose LOW_HSI Images for fusion:", items, 0, False)
        if not ok_2 or not item_2:
            return
        low_hsi_widget = None
        for i in range(self.tabWidget.count()):
            if self.tabWidget.tabText(i) == item_2:
                low_hsi_widget  = self.tabWidget.widget(i)
                break
        if low_hsi_widget  is None:
            QMessageBox.warning(self, "Selection Error", "MSI image not found.")
            return


        item_3, ok_3 = QInputDialog.getItem(self, "Select ORIGINAL Image",
                                        "Choose ORIGINAL Image for Calculating Quality Metrics:", items, 0, False)
        if not ok_3 or not item_3:
            return
        original_widget = None
        for i in range(self.tabWidget.count()):
            if self.tabWidget.tabText(i) == item_3:
                original_widget   = self.tabWidget.widget(i)
                break
        if  original_widget   is None:
            QMessageBox.warning(self, "Selection Error", "MSI image not found.")
            return


        try:
            LOW_HSI_IMAGE = low_hsi_widget.image_loader.image_data
            MSI_IMAGE = msi_widget.image_loader.image_data
            ORIGINAL_IMAGE = original_widget.image_loader.image_data

            if LOW_HSI_IMAGE is None:
                raise ValueError("LOW_HSI IMAGE data not available.")
            if MSI_IMAGE is None:
                raise ValueError("MSI_IMAGE IMAGE data not available.")

            fused = CNMF_fusion(LOW_HSI_IMAGE, MSI_IMAGE)

            fused = fused.astype(np.float32)

            psnr_value = psnr(ORIGINAL_IMAGE, fused)
            rmse_value = rmse(ORIGINAL_IMAGE, fused)
            ergas_value = ergas(ORIGINAL_IMAGE, fused)
            uqi_value = uqi(ORIGINAL_IMAGE, fused)
            sam_value = sam(ORIGINAL_IMAGE, fused)

            QMessageBox.information(
                self,
                "Fusion Complete",
                f"Fusion complete.\n\n"
                f"PSNR = {psnr_value:.6f}\n"
                f"UQI  = {uqi_value:.6f}\n"
                f"RMSE = {rmse_value:.6f}\n"
                f"SAM  = {sam_value:.6f}\n"
                f"ERGAS = {ergas_value:.6f}"
            )

            self.addFusedImageTab(fused)


        except Exception as e:
            QMessageBox.critical(self, "Fusion Error", f"Fusion process failed:\n{e}")


    def viewColorComposite(self):
        current = self.getCurrentImageDisplay()
        if current and current.image_loader:
            current.displayColorComposite()
        else:
            QMessageBox.warning(self, "No Image", "There is no image loaded.")

    def calculateLSTTimelapse(self):
        """
        LST Timelapse over multiple folders, fully patched to avoid
        boolean‐mask shape errors, with:
          - Emissivity mode choice
          - Dynamic/Fixed mapping (1st–99th pct for Dynamic)
          - Optional polygon mask
        """
        import os, traceback
        import numpy as np
        from PyQt5.QtWidgets import QMessageBox, QFileDialog, QListView, QTreeView, QInputDialog

        try:
            # 1) Emissivity mode
            lse_mode, ok = QInputDialog.getItem(
                self, "Emissivity Mode",
                "Select LSE calculation mode:",
                ["auto-ndvi-raw", "auto-ndvi-dos"], 0, False
            )
            if not ok:
                return

            # 2) Color mapping
            cmap_mode, ok = QInputDialog.getItem(
                self, "Color Mapping",
                "Choose mapping mode:", ["Dynamic", "Fixed"], 0, False
            )
            if not ok:
                return
            fixed_min = fixed_max = None
            if cmap_mode == "Fixed":
                fixed_min, ok1 = QInputDialog.getDouble(
                    self, "Fixed Scale", "Min temperature (°C):", 0.0, -100, 1000, 2
                )
                if not ok1:
                    return
                fixed_max, ok2 = QInputDialog.getDouble(
                    self, "Fixed Scale", "Max temperature (°C):", 50.0, -100, 1000, 2
                )
                if not ok2:
                    return

            # 3) Select folders
            dlg = QFileDialog(self, "Select Folders for LST Timelapse")
            dlg.setOption(QFileDialog.DontUseNativeDialog, True)
            dlg.setFileMode(QFileDialog.Directory)
            dlg.setOption(QFileDialog.ShowDirsOnly, True)
            for view in dlg.findChildren((QListView, QTreeView)):
                view.setSelectionMode(view.ExtendedSelection)
            if not dlg.exec_():
                return
            folders = dlg.selectedFiles()
            if not folders:
                return

            # 4) Prepare
            from L8LST_PSWA.LSTRetriever import LSTRetriever
            from image_loader import LandsatMetadataReader
            from osgeo import gdal
            temp_dir = os.path.join(os.getcwd(), "lst_temp")
            os.makedirs(temp_dir, exist_ok=True)

            lst_frames = []
            frame_dates = []

            # 5) Loop over each folder
            for fld in folders:
                mtl = next(
                    (os.path.join(fld, f) for f in os.listdir(fld) if f.upper().endswith("_MTL.TXT")),
                    None
                )
                if not mtl:
                    continue

                reader = LandsatMetadataReader(mtl)
                raw_date = reader.metadata.get("DATE_ACQUIRED", "")
                if raw_date:
                    parts = raw_date.split("-")
                    date_str = f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else raw_date
                else:
                    date_str = "9999-99"
                frame_dates.append(date_str)

                retriever = LSTRetriever(
                    metadata_file=mtl,
                    LSE_mode=lse_mode,
                    temp_dir=temp_dir,
                    window_size=15
                )

                # ─── PATCH 1: capture BT shape ────────────────────────────────
                orig_bt = retriever.get_b10_b11_brightness_temp_arrays

                def patched_bt():
                    b10_bt, b11_bt = orig_bt()
                    retriever._last_bt = b10_bt
                    return b10_bt, b11_bt

                retriever.get_b10_b11_brightness_temp_arrays = patched_bt

                # ─── PATCH 2: cloud mask zeros of correct shape ──────────────
                retriever.get_fmask_cloud_array = lambda: np.zeros_like(retriever._last_bt)

                # ─── PATCH 3: CWV resized to BT shape ─────────────────────────
                orig_cwv = retriever.get_cwv_as_array

                def patched_cwv(cloud_array, b10_bt, b11_bt):
                    rows, cols = b10_bt.shape
                    ca = cloud_array
                    ca2 = np.zeros((rows, cols), dtype=ca.dtype)
                    r, c = min(rows, ca.shape[0]), min(cols, ca.shape[1])
                    ca2[:r, :c] = ca[:r, :c]
                    return orig_cwv(ca2, b10_bt, b11_bt)

                retriever.get_cwv_as_array = patched_cwv

                # Build dummy cloud-mask file
                ds10 = gdal.Open(retriever.band_10_path)
                maskp = os.path.join(temp_dir, f"mask_{os.path.basename(fld)}.tif")
                if os.path.exists(maskp):
                    os.remove(maskp)
                drv = gdal.GetDriverByName("GTiff")
                out = drv.Create(maskp, ds10.RasterXSize, ds10.RasterYSize, 1, gdal.GDT_Byte)
                out.SetGeoTransform(ds10.GetGeoTransform())
                out.SetProjection(ds10.GetProjection())
                out.GetRasterBand(1).Fill(0)
                out = None

                # Compute LST (Kelvin → Celsius)
                with np.errstate(divide='ignore', invalid='ignore'):
                    lst_k = retriever.get_lst_array()
                if lst_k is None:
                    continue
                valid = np.isfinite(lst_k)
                fill = np.min(lst_k[valid]) if valid.any() else 0.0
                lst_k = np.where(valid, lst_k, fill)
                lst_c = lst_k - 273.15

                # Percentile clip for Dynamic
                if cmap_mode == "Dynamic":
                    p1 = np.nanpercentile(lst_c, 1)
                    p99 = np.nanpercentile(lst_c, 99)
                    lst_c = np.clip(lst_c, p1, p99)

                lst_frames.append(lst_c.astype("float32"))

                # 6) Sort by date
            combined = sorted(zip(frame_dates, lst_frames), key=lambda x: x[0])
            frame_dates, lst_frames = zip(*combined)

            if not lst_frames:
                QMessageBox.warning(self, "No Frames", "No LST frames computed.")
                return

            # 6) Optional polygon mask
            vec_path, _ = QFileDialog.getOpenFileName(
                self, "Mask to City Vector (optional)", "",
                "GeoJSON (*.geojson);;Shapefile (*.shp);;All Files (*)"
            )
            if vec_path:
                import fiona, rasterio, rasterio.features
                from shapely.geometry import shape
                with fiona.open(vec_path) as src:
                    geoms = [feature["geometry"] for feature in src]
                import rasterio as rio
                with rio.open(retriever.band_10_path) as src_ds:
                    transform = src_ds.transform

                masked = []
                for arr in lst_frames:
                    mask = rasterio.features.geometry_mask(
                        geoms, out_shape=arr.shape, transform=transform, invert=True
                    )
                    masked.append(np.where(mask, arr, np.nan).astype("float32"))
                lst_frames = masked

            # --- after you have filled lst_frames with arrays ---

            # 7) Ensure all frames share the same shape by cropping to the intersection:
            #    find min rows and cols across frames
            shapes = [arr.shape for arr in lst_frames]
            min_h = min(h for h, w in shapes)
            min_w = min(w for h, w in shapes)
            cropped_frames = [arr[:min_h, :min_w] for arr in lst_frames]

            # Stack & metadata
            cube = np.stack(cropped_frames, axis=-1)  # now all frames are same shape

            meta = {
                "LST Timelapse": f"{cube.shape[2]} frames",
                "LSE_mode": lse_mode,
                "color_mapping": cmap_mode.lower(),
                "frame_dates": list(frame_dates)
            }
            if cmap_mode == "Fixed":
                meta.update({"fixed_min": fixed_min, "fixed_max": fixed_max})
            else:
                meta["global_min"] = float(np.nanpercentile(cube, 1))
                meta["global_max"] = float(np.nanpercentile(cube, 99))

            # 8) Display in a new tab
            from image_view import ImageDisplayWidget
            from image_loader import ImageLoader

            w = ImageDisplayWidget()
            loader2 = ImageLoader("")
            loader2.image_data = cube
            loader2.num_bands = cube.shape[2]
            loader2.metadata = meta

            w.image_loader = loader2
            w.setContentMode()
            w.saved_path = ""

            self.tabWidget.addTab(w, "LST Timelapse")
            self.tabWidget.setCurrentWidget(w)

            QMessageBox.information(self, "Done", "LST timelapse ready—use the slider to scroll through time!")

        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "calculateLSTTimelapse Error", f"{e}\n\n{tb}")
            return

    def downsampleHyperspectral(self):
        current = self.getCurrentImageDisplay()
        if not current or not current.image_loader:
            QMessageBox.warning(self, "No Image", "No image loaded.")
            return
        w, ok = QInputDialog.getInt(self, "Downsample", "Enter downsampling factor (w):", 2, 1, 100, 1)
        if not ok:
            return
        data = current.image_loader.image_data
        try:
            downsampled = gaussian_down_sample(data, w)
        except Exception as e:
            QMessageBox.critical(self, "Downsampling Error", f"Error during downsampling:\n{e}")
            return
        new_metadata = dict(current.image_loader.metadata)
        new_metadata["downsample_factor"] = w
        new_metadata["original_shape"] = data.shape
        new_metadata["downsampled_shape"] = downsampled.shape
        ds_widget = ImageDisplayWidget()
        from image_loader import ImageLoader
        ds_widget.image_loader = ImageLoader("")
        ds_widget.image_loader.image_data = downsampled
        if downsampled.ndim == 2:
            ds_widget.image_loader.num_bands = 1
        else:
            ds_widget.image_loader.num_bands = downsampled.shape[2]
        ds_widget.image_loader.metadata = new_metadata
        ds_widget.setContentMode()
        ds_widget.saved_path = ""
        self.tabWidget.addTab(ds_widget, f"Downsampled (w={w})")
        self.tabWidget.setCurrentWidget(ds_widget)
        QMessageBox.information(self, "Downsampling Complete", f"Downsampled image generated with factor {w}.")

    def runTransformToMultispectral(self):
        if self.tabWidget.count() == 0:
            QMessageBox.warning(self, "No Images", "No images are open.")
            return
        items = [self.tabWidget.tabText(i) for i in range(self.tabWidget.count())]
        img_item, ok1 = QInputDialog.getItem(self, "Select Image",
                                             "Select an image to transform:", items, 0, False)
        if not ok1 or not img_item:
            return
        ref_widget = None
        for i in range(self.tabWidget.count()):
            if self.tabWidget.tabText(i) == img_item:
                ref_widget = self.tabWidget.widget(i)
                break
        if ref_widget is None:
            QMessageBox.warning(self, "Selection Error", "Image not found.")
            return
        sensors = ["Landsat 7", "Landsat 8", "Sentinel 2A", "Sentinel 2B"]
        sensor, ok2 = QInputDialog.getItem(self, "Select Sensor",
                                           "Select SRF to use:", sensors, 0, False)
        if not ok2 or not sensor:
            return
        hyper = ref_widget.image_loader.image_data
        if hyper is None:
            QMessageBox.warning(self, "Error", "Image data not available.")
            return
        try:
            hyper = hyper.astype(np.float32)
            ms = self.transformToMultispectral(hyper, sensor)
        except Exception as e:
            QMessageBox.critical(self, "Transformation Error", f"Error during transformation:\n{e}")
            return

        new_metadata = dict(ref_widget.image_loader.metadata)
        new_metadata["transformation"] = f"Transformed to multispectral using {sensor} SRF"
        new_metadata["bands"] = ms.shape[2]
        new_metadata["wavelength"] = self.new_wavelengths

        ms_widget = ImageDisplayWidget()
        from image_loader import ImageLoader
        ms_widget.image_loader = ImageLoader("")
        ms_widget.image_loader.image_data = ms
        ms_widget.image_loader.num_bands = ms.shape[2]
        ms_widget.image_loader.metadata = new_metadata
        ms_widget.setContentMode()
        ms_widget.saved_path = ""
        self.tabWidget.addTab(ms_widget, f"MS {img_item} ({sensor})")
        self.tabWidget.setCurrentWidget(ms_widget)
        QMessageBox.information(self, "Transformation Complete", f"Multispectral image generated using {sensor} SRF.")


    def transformToMultispectral(self, hyper, sensor):
        if sensor == "Landsat 7":
            # Use your original Landsat 7 SRF function
            srf = get_landsat7_srf()
            band_names = ["Band 1", "Band 2", "Band 3", "Band 4", "Band 5", "Band 7", "Band 8"]
            interp_kwargs = {}
        elif sensor == "Landsat 8":
            # Use the interpolation approach (just like Landsat 7).
            # Make sure you have a function that returns a dict with
            # { "Band 1": {"wavelength": [...], "response": [...]}, ... }
            # for all Landsat 8 bands.
            srf = get_landsat8_srf()  # You must define this function
            band_names = list(srf.keys())
            interp_kwargs = {}
        elif sensor == "Sentinel 2A":
            # Keep the old bounding approach as you originally had
            srf = get_sentinel2a_srf()
            band_names = list(srf.keys())
            interp_kwargs = {}
        elif sensor == "Sentinel 2B":
            # Keep the old bounding approach as you originally had
            srf = get_sentinel2b_srf()
            band_names = list(srf.keys())
            interp_kwargs = {}
        else:
            raise ValueError("Unknown sensor SRF specified.")

        hyper_wl = self.getCurrentImageDisplay().image_loader.metadata.get("wavelength", None)
        if hyper_wl is None:
            raise ValueError("No wavelength information found in image metadata.")
        hyper_wl = np.array(hyper_wl, dtype=np.float32)
        H, W, B = hyper.shape
        m = len(band_names)
        ms = np.zeros((H, W, m), dtype=np.float32)
        self.new_wavelengths = []

        for i, bn in enumerate(band_names):

            if sensor == "Landsat 7":
                # --- Landsat 7 interpolation approach (unchanged) ---
                xp = np.array(srf[bn]["wavelength"], dtype=np.float32)
                fp = np.array(srf[bn]["response"], dtype=np.float32)
                L = min(len(xp), len(fp))
                xp = xp[:L]
                fp = fp[:L]
                response = np.interp(hyper_wl, xp, fp, **interp_kwargs)
                weight_sum = np.sum(response)
                if weight_sum == 0:
                    weight_sum = 1e-6
                effective_wl = np.sum(hyper_wl * response) / weight_sum
                self.new_wavelengths.append(effective_wl)
                weighted_sum = np.zeros((H, W), dtype=np.float32)
                for j in range(B):
                    weighted_sum += hyper[:, :, j] * response[j]
                ms[:, :, i] = weighted_sum / weight_sum

            elif sensor == "Landsat 8":
                # --- Landsat 8 now uses the SAME interpolation approach as Landsat 7 ---
                xp = np.array(srf[bn]["wavelength"], dtype=np.float32)
                fp = np.array(srf[bn]["response"], dtype=np.float32)
                L = min(len(xp), len(fp))
                xp = xp[:L]
                fp = fp[:L]
                response = np.interp(hyper_wl, xp, fp, **interp_kwargs)
                weight_sum = np.sum(response)
                if weight_sum == 0:
                    weight_sum = 1e-6
                effective_wl = np.sum(hyper_wl * response) / weight_sum
                self.new_wavelengths.append(effective_wl)
                weighted_sum = np.zeros((H, W), dtype=np.float32)
                for j in range(B):
                    weighted_sum += hyper[:, :, j] * response[j]
                ms[:, :, i] = weighted_sum / weight_sum

            else:
                xp = np.array(srf[bn]["wavelength"], dtype=np.float32)
                fp = np.array(srf[bn]["response"], dtype=np.float32)
                L = min(len(xp), len(fp))
                xp = xp[:L]
                fp = fp[:L]
                response = np.interp(hyper_wl, xp, fp, **interp_kwargs)
                weight_sum = np.sum(response)
                if weight_sum == 0:
                    weight_sum = 1e-6
                effective_wl = np.sum(hyper_wl * response) / weight_sum
                self.new_wavelengths.append(effective_wl)
                weighted_sum = np.zeros((H, W), dtype=np.float32)
                for j in range(B):
                    weighted_sum += hyper[:, :, j] * response[j]
                ms[:, :, i] = weighted_sum / weight_sum

        return ms

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())