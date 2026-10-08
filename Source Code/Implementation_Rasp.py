from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton,
                             QInputDialog, QLabel, QHBoxLayout, QFileDialog)
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from osgeo import gdal
from CNMF import vca  # Import VCA function from HSBSV module
import matplotlib as mpl  # For better color mapping
from scipy.ndimage import zoom
import scipy.linalg as splin
from sunsal import sunsal  # Import the sunsal function

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


# --- Function to Read Images ---
def read_image(file_path):
    """Reads an image using GDAL and returns a NumPy array."""
    if not os.path.isfile(file_path):
        print(f"Error: File not found - {file_path}")
        return None

    dataset = gdal.Open(file_path)
    if dataset is None:
        print(f"Error: Unable to open {file_path}")
        return None

    width, height, num_bands = dataset.RasterXSize, dataset.RasterYSize, dataset.RasterCount
    print(f"\nFile: {file_path}\nDimensions: {width} x {height}\nNumber of Bands: {num_bands}")

    image_data = np.zeros((height, width, num_bands), dtype=np.float32)
    for band in range(num_bands):
        raster_band = dataset.GetRasterBand(band + 1)
        image_data[:, :, band] = raster_band.ReadAsArray().astype(np.float32)

    return image_data


def gaussian_down_sample(I, ratio, mask=None):
    """
    Gaussian downsampling of an image

    Parameters:
    -----------
    I : ndarray
        Input image with shape (height, width, bands)
    ratio : int
        Downsampling ratio
    mask : ndarray, optional
        Binary mask for processing

    Returns:
    --------
    ndarray
        Downsampled image
    """
    if mask is None:
        mask = np.ones((I.shape[0], I.shape[1]))

    rows, cols, bands = I.shape
    rows_lr = rows // ratio
    cols_lr = cols // ratio

    # Initialize low-resolution image
    I_lr = np.zeros((rows_lr, cols_lr, bands))

    # Gaussian kernel parameters
    sigma = ratio / 3.0
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)

    # Create Gaussian kernel
    x = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
    y = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
    X, Y = np.meshgrid(x, y)
    kernel = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    kernel = kernel / np.sum(kernel)

    # Apply convolution and downsample
    from scipy.ndimage import convolve
    for b in range(bands):
        I_convolved = convolve(I[:, :, b], kernel, mode='reflect')
        I_lr[:, :, b] = I_convolved[::ratio, ::ratio]

    return I_lr


def zoom_nn(img, ratio):
    """
    Nearest neighbor upsampling/downsampling of an image

    Parameters:
    -----------
    img : ndarray
        Input image
    ratio : int or float
        Zoom ratio

    Returns:
    --------
    ndarray
        Zoomed image
    """
    return zoom(img, ratio, order=0)


def lsqnonneg(A, C):
    """
    Solve non-negative least squares problem using scipy

    Parameters:
    -----------
    A : ndarray
        Design matrix
    C : ndarray
        Target vector

    Returns:
    --------
    ndarray
        Solution vector
    """
    from scipy.optimize import nnls

    n = A.shape[1]
    x = np.zeros((n, C.shape[1]))

    for i in range(C.shape[1]):
        result, _ = nnls(A, C[:, i])
        x[:, i] = result

    return x


def estR(HS, MS, mask=0):
    '''
    Estimation of relative spectral response functions (SRFs)
    via the nonnegative least squares method

    USAGE
        R = estR(HS,MS,mask)

    INPUT
        HS  : Low-spatial-resolution HS image (rows2,cols2,bands2)
        MS  : MS image (rows1,cols1,bands1)
        mask: (optional) Binary mask for processing (rows2,cols2) (mainly
              for real data)

    OUTPUT
        R   : Relative SRFs
              without mask (bands1,bands2)
              with mask    (bands1,bands2+1) (consider offset)
    '''
    print(f"HS shape: {HS.shape}, MS shape: {MS.shape}")

    rows1 = MS.shape[0]
    cols1 = MS.shape[1]
    bands1 = MS.shape[2]
    rows2 = HS.shape[0]
    cols2 = HS.shape[1]
    bands2 = HS.shape[2]

    # masking mode
    if np.isscalar(mask):
        masking = 0
        mask = np.ones((rows2, cols2))
    else:
        masking = 1

    # Add a column of ones to handle offset
    HS_reshaped = HS.reshape(rows2 * cols2, bands2)
    HS_with_ones = np.hstack((HS_reshaped, mask.reshape(rows2 * cols2, 1)))
    bands2_plus_1 = bands2 + 1

    R = np.zeros((bands1, bands2_plus_1))

    # downgrade spatial resolution
    w = int(rows1 / rows2)  # Ensure w is an integer
    mask2 = zoom_nn(mask, w)

    Y = gaussian_down_sample(MS, w, mask2).reshape(rows2 * cols2, bands1)

    A = HS_with_ones

    if masking == 1:
        Y = Y[mask.reshape(rows2 * cols2) == 1, :]
        A = A[mask.reshape(rows2 * cols2) == 1, :]

    print(f"A shape: {A.shape}, Y shape: {Y.shape}")

    # solve nonnegative least squares problems
    for b in range(bands1):
        y = Y[:, b].reshape(-1, 1)
        r = lsqnonneg(A, y)
        R[b, :] = r.T

    print(f"Estimated R shape: {R.shape}")
    return R


# --- GUI Class ---
class HSBSVProcessor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HSBSV Processing")
        self.setGeometry(100, 100, 800, 500)

        # Initialize variables
        self.HSI = None
        self.MSI = None
        self.vca_results = []
        self.subsets = []

        # Main layout
        self.layout = QVBoxLayout()

        # Image selection area
        img_layout = QHBoxLayout()

        # HSI selection
        hsi_layout = QVBoxLayout()
        self.hsi_label = QLabel("No HSI selected")
        self.hsi_button = QPushButton("Load HSI Image")
        self.hsi_button.clicked.connect(self.load_hsi)
        hsi_layout.addWidget(self.hsi_label)
        hsi_layout.addWidget(self.hsi_button)

        # MSI selection
        msi_layout = QVBoxLayout()
        self.msi_label = QLabel("No MSI selected")
        self.msi_button = QPushButton("Load MSI Image")
        self.msi_button.clicked.connect(self.load_msi)
        msi_layout.addWidget(self.msi_label)
        msi_layout.addWidget(self.msi_button)

        img_layout.addLayout(hsi_layout)
        img_layout.addLayout(msi_layout)
        self.layout.addLayout(img_layout)

        # Processing buttons
        process_layout = QVBoxLayout()

        self.subset_button = QPushButton("Generate and Process Subsets")
        self.subset_button.clicked.connect(self.process_subsets)
        process_layout.addWidget(self.subset_button)

        self.apply_HSBSV_button = QPushButton("Apply HSBSV and Visualize")
        self.apply_HSBSV_button.clicked.connect(self.apply_HSBSV)
        self.apply_HSBSV_button.setEnabled(False)  # Disabled until subsets are processed
        process_layout.addWidget(self.apply_HSBSV_button)



        self.layout.addLayout(process_layout)

        # Status
        self.status_label = QLabel("Load HSI and MSI images to begin")
        self.layout.addWidget(self.status_label)
        self.setLayout(self.layout)

    def load_hsi(self):
        """Load a hyperspectral image file"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open HSI Image", "", "All Files (*)")
        if file_path:
            self.HSI = read_image(file_path)
            if self.HSI is not None:
                self.hsi_label.setText(f"HSI: {os.path.basename(file_path)}\n{self.HSI.shape}")
                self.status_label.setText("HSI loaded successfully!")
                self.update_button_states()

    def load_msi(self):
        """Load a multispectral image file"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open MSI Image", "", "All Files (*)")
        if file_path:
            self.MSI = read_image(file_path)
            if self.MSI is not None:
                self.msi_label.setText(f"MSI: {os.path.basename(file_path)}\n{self.MSI.shape}")
                self.status_label.setText("MSI loaded successfully!")
                self.update_button_states()

    def update_button_states(self):
        """Update the enabled state of buttons based on loaded data"""
        if self.HSI is not None and self.MSI is not None:
            self.subset_button.setEnabled(True)
            self.status_label.setText("Ready to process subsets")
        else:
            self.subset_button.setEnabled(False)

    def process_subsets(self):
        """Process multiple subsets based on user input"""
        if self.HSI is None:
            self.status_label.setText("Error: HSI not loaded")
            return

        # Get user inputs
        num_subsets, ok1 = QInputDialog.getInt(self, "Input", "Number of subsets:", 5, 1, 100)
        if not ok1:
            return

        subset_width, ok2 = QInputDialog.getInt(self, "Input", "Subset width:", 50, 10, self.HSI.shape[1])
        if not ok2:
            return

        subset_height, ok3 = QInputDialog.getInt(self, "Input", "Subset height:", 50, 10, self.HSI.shape[0])
        if not ok3:
            return

        num_endmembers, ok4 = QInputDialog.getInt(self, "Input", "Number of endmembers per subset:", 7, 1, 20)
        if not ok4:
            return

        # Clear previous results
        self.vca_results = []
        self.subsets = []

        # Process each subset
        for i in range(num_subsets):
            # Generate random coordinates
            x0 = np.random.randint(0, self.HSI.shape[1] - subset_width + 1)
            y0 = np.random.randint(0, self.HSI.shape[0] - subset_height + 1)

            # Extract subset
            subset = self.HSI[y0:y0 + subset_height, x0:x0 + subset_width, :]
            self.subsets.append((subset, (y0, x0, subset_height, subset_width)))

            # Apply VCA
            subset_reshaped = subset.reshape(subset_height * subset_width, self.HSI.shape[2]).T
            W_hyper, _ = vca(subset_reshaped, num_endmembers)
            self.vca_results.append(W_hyper)

            self.status_label.setText(f"Processed subset {i + 1}/{num_subsets}")

        # Consolidate all VCA results
        self.consolidated_vca = np.hstack(self.vca_results)
        print(f"Consolidated VCA shape: {self.consolidated_vca.shape}")

        self.apply_HSBSV_button.setEnabled(True)
        self.status_label.setText(f"All {num_subsets} subsets processed successfully")

    def apply_HSBSV(self):
        """Apply the HSBSV algorithm using estR and sunsal"""
        if self.MSI is None or len(self.vca_results) == 0:
            self.status_label.setText("Error: MSI or VCA results not available")
            return

        # Apply estR
        try:
            # Run estR to get spectral response functions
            R = estR(self.HSI, self.MSI)

            # Get only the spectral response part (without the offset column)
            R_spectral = R[:, :-1]

            # Apply R to the consolidated VCA results
            EM_msi = np.dot(R_spectral, self.consolidated_vca)


            # Reshape MSI for sunsal
            rows1, cols1, bands1 = self.MSI.shape
            Y = self.MSI.reshape(rows1 * cols1, bands1).T

            # Run sunsal to get abundance maps
            A, res_p, res_d, _ = sunsal(EM_msi, Y, positivity=True, addone=True)

            fused_image = np.dot(self.consolidated_vca, A)  # Shape: (144, 10000)
            fused_image = fused_image.reshape((144, 100, 100))  # Shape: (144, 100, 100)
            fused_image = np.transpose(fused_image, (1, 2, 0))  # Final shape: (100, 100, 144)


            self.visualize_fused_image(fused_image)

            # Visualize results (non-blocking)
            self.visualize_results(A, EM_msi)

            self.status_label.setText("HSBSV processing completed successfully")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error during HSBSV processing: {e}")
            self.status_label.setText(f"Error during HSBSV processing: {str(e)}")

    def visualize_results(self, abundance_maps, endmembers_msi):
        """Visualize the results of the HSBSV algorithm"""
        num_maps = abundance_maps.shape[0]
        rows = int(np.ceil(np.sqrt(num_maps)))
        cols = int(np.ceil(num_maps / rows))

        # Create separate visualization window to avoid event loop issues
        fig = plt.figure(figsize=(15, 10))

        # Get MSI dimensions for reshaping abundance maps
        rows1, cols1 = self.MSI.shape[0], self.MSI.shape[1]

        for i in range(num_maps):
            if i < abundance_maps.shape[0]:
                ax = fig.add_subplot(rows, cols, i + 1)

                # Check if the abundance map is already flattened
                if len(abundance_maps.shape) == 2:
                    # Reshape the flattened abundance map
                    im = ax.imshow(abundance_maps[i].reshape(rows1, cols1), cmap='viridis')
                else:
                    im = ax.imshow(abundance_maps[i], cmap='viridis')

                ax.set_title(f"Abundance Map {i + 1}")
                fig.colorbar(im, ax=ax)

        # Show the figure without blocking
        fig.tight_layout(pad=2.0)
        plt.show(block=False)

    def displayColorComposite(self, image_data, wavelengths=None):
        """
        Displays a color composite by averaging bands within a spectral window
        around the target wavelengths for red, green, and blue.

        Parameters:
        -----------
        image_data : ndarray
            The hyperspectral image data with shape (height, width, bands)
        wavelengths : ndarray, optional
            Array of wavelengths corresponding to each band
        """
        if image_data is None:
            self.status_label.setText("Error: No image data available.")
            return

        # Ensure the image has multiple bands.
        if image_data.ndim < 3 or image_data.shape[2] < 3:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Insufficient Bands",
                                "The image does not have enough bands for a color composite.")
            return

        # If wavelengths are not provided, create artificial ones
        if wavelengths is None:
            num_bands = image_data.shape[2]
            wavelengths = np.linspace(400, 900, num_bands)  # Assume range from 400nm to 900nm

        wavelengths = np.array(wavelengths, dtype=np.float32)

        # Define target wavelengths and tolerance (in nm) for averaging.
        targets = {"red": 650, "green": 550, "blue": 450}
        tolerance = 20  # ±20 nm window

        channels = {}
        for color, target in targets.items():
            # Get indices of bands within the window.
            indices = np.where((wavelengths >= target - tolerance) & (wavelengths <= target + tolerance))[0]
            # If no bands fall in the window, use the single band with the minimum difference.
            if indices.size == 0:
                differences = np.abs(wavelengths - target)
                indices = np.array([differences.argmin()])
            # Average the bands along the third dimension.
            channels[color] = np.mean(image_data[:, :, indices], axis=2)

        # Stack into an RGB image.
        rgb = np.stack([channels["red"], channels["green"], channels["blue"]], axis=-1)

        # Normalize each channel to the range [0, 255] and convert to uint8.
        rgb_normalized = np.zeros_like(rgb, dtype=np.uint8)
        for i in range(3):
            channel = rgb[..., i].astype(np.float32)
            min_val = channel.min()
            # Use np.ptp() instead of the removed .ptp() method
            max_min_diff = np.ptp(channel) + 1e-8  # Avoid division by zero
            norm = ((channel - min_val) / max_min_diff * 255).astype(np.uint8)
            rgb_normalized[..., i] = norm

        # Display using matplotlib
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(rgb_normalized)
        ax.set_title("RGB Color Composite")
        ax.axis('off')
        plt.tight_layout()
        plt.show(block=False)

        return rgb_normalized

    def visualize_fused_image(self, fused_image):
        """Visualize the fused image as RGB."""
        if fused_image is None:
            self.status_label.setText("Error: No fused image available.")
            return

        try:
            # Use the displayColorComposite method to create an RGB visualization
            rgb_image = self.displayColorComposite(fused_image)

            self.status_label.setText("Fused image displayed as RGB composite")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in RGB visualization: {e}")
            self.status_label.setText(f"Error in RGB visualization: {str(e)}")



# --- Main Script Execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HSBSVProcessor()
    window.show()
    sys.exit(app.exec_())
