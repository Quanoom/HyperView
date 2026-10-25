# HyperView Platform : 
### Introduction : 
<b>HyperView</b>, a comprehensive software platform created for the advanced analysis of multispectral and hyperspectral remote sensing imagery. 
The research presented systematically explores several key domains of image processing. 
A significant focus is placed on the estimation of Land Surface Temperature (LST), for which the Practical Split-Window Algorithm (PSWA) and its SWCVR extension are implemented and validated using Landsat 8 data.
This methodology is practically applied to monitor and analyze the dynamics of urban heat islands in the coastal city of Oran, Algeria. Furthermore, the study conducts a comparative investigation of two sophisticated image fusion techniques: Coupled Nonnegative Matrix Factorization (CNMF) and Hyperspectral Super-resolution with Spectra Bundles dealing with Spectral Variability (HSB-SV). 
The goal of this fusion is to synergistically combine data to enhance the spatial resolution of hyperspectral images while rigorously preserving their spectral fidelity. The memoir also provides an in-depth treatment of spectral unmixing, 
employing Vertex Component Analysis (VCA) for the robust extraction of pure spectral signatures (endmembers) and subsequently applying various Non-negative Matrix Factorization (NMF) methods including HALS, NNLS, and LMSU to accurately estimate the fractional abundance of these materials in each pixel.
In a final, forward-looking chapter, the work evaluates the real-world feasibility of deploying the computationally intensive HSB-SV algorithm on resource-constrained embedded systems.
This is achieved by implementing and exhaustively testing the algorithm on Raspberry Pi 3B/3B+ platforms, assessing critical performance metrics and reconstruction quality to validate its potential for on-board, real-time applications such as drone-based remote sensing.

### Graphic User Interface : 
#### Main Window and Layout :
The main window of HyperView, defined in the `main_window.py` module by the `MainWindow` class, serves as the central container for all the software's features. It follows a standard layout: a menu bar at the top; a navigation toolbar on the left for quick access to image-interaction tools; a central area dedicated to displaying images via a tabbed system; and a status bar at the bottom that shows contextual information, such as the name of the loaded file. The overall layout is designed to maximize the space available for image display while keeping essential tools easily accessible.

<img width="783" height="465" alt="Screenshot_25-10-2026_164011_" src="https://github.com/user-attachments/assets/81405f96-4c2d-44a1-9495-47ddfc1adcb3" />

#### Managing multiple images via a tabbed interface :
To allow the user to work on several images simultaneously, HyperView uses a `QTabWidget` from PyQt5, managed within `main_window.py`. Each opened image is displayed in a separate tab, with the tab title corresponding to the image filename. This enables the user to easily switch between images. Each tab contains an instance of the `ImageDisplayWidget`, which handles the display and interactions specific to that image. Tabs can be closed individually, provided that at least one tab remains open to keep the work session active.

<img width="939" height="604" alt="Screenshot_25-10-2026_164022_" src="https://github.com/user-attachments/assets/7fe26cbf-f339-49d2-a642-b91b28a656c5" />

#### Navigation Toolbar :
A vertical toolbar, located on the left side of the main window, provides quick access to 
essential tools for interacting with the image displayed in the active tab. These 
tools—featuring icons chosen for their clarity—include: 
• Selection/Pixel Information: Allows you to click on a pixel to retrieve information 
about it (such as its coordinates or, via a specific action, its spectrum). 
• Pan: Enables navigation within the image when it is zoomed in beyond the 
boundaries of the display area. 
• Zoom In/Out: Allows you to increase or decrease the magnification level of the image. 
• Rotation: Enables rotation of the displayed image. Activating any of these tools 
changes how the mouse interacts with the `ImageView` widget contained within the 
current tab.

<img width="45" height="204" alt="Screenshot_25-10-2026_164556_" src="https://github.com/user-attachments/assets/54495d4a-fb79-49d3-853b-b77e8780ccd0" />

#### Menus and General Features :
The top menu bar organizes all of HyperView’s features in a hierarchical structure. The main menus include:
• File: Contains standard operations such as Open, Save, Save As, and Close Application.
• Edit: Offers basic editing tools, including Undo/Redo (for display-related changes) and Crop.
• Multispectral and Hyperspectral: These menus group together processing and analysis features specific to these types of imagery, such as generating color composites, performing spectral transformations (sensor simulation), sub-sampling, and accessing basic atmospheric corrections. More advanced features (unmixing, LST, fusion) are also listed here, though their algorithmic details are covered in dedicated chapters of this thesis.
• Fusion, Spectral Unmixing, Spectral Library: These menus provide access to specialized software features, which are the focus of other chapters.
• Help: Provides information about the software.

<img width="770" height="155" alt="Screenshot_25-10-2026_164035_" src="https://github.com/user-attachments/assets/9e4f3ce3-519a-4819-8362-eb634ab11f96" />

### Data Visualization and Exploration :
Once images have been loaded, HyperView provides several tools for their visualization and interactive exploration.
#### Single-band and multi-band display (slider) :
The main image display is handled by the `ImageDisplayWidget`, which incorporates an instance of `ImageView`. For multi-band images, a slider (`QSlider`) allows the user to navigate dynamically through the various spectral bands. The selected band is then displayed in grayscale. A label indicates the current band number and, if wavelength metadata is available, the corresponding central wavelength. The image is normalized for display in order to utilize the full dynamic range of grayscale values ​​(0–255).

<img width="937" height="513" alt="Screenshot_25-10-2026_164058_" src="https://github.com/user-attachments/assets/f3f81f04-9e8a-4354-a356-5a9b45ca610f" />

#### Color Composites (RGB, FCC standard) :
HyperView allows you to generate color composites for better visual interpretation of multispectral and hyperspectral images. A "natural" color composite feature (True Color Composite) attempts to assign the bands closest to visible red, green, and blue to the corresponding display channels, based on the wavelength information available in the metadata (displayColorComposite in main_window.py). For Landsat data, a specific option (displayLandsatColorComposite) allows you to quickly create standard color composites, such as natural color composites (Bands 4, 3, 2 for Landsat 8/9 or 3, 2, 1 for Landsat 4/5/7) or false infrared color composites (e.g., Near Infrared, Red, Green). Each channel in the composition is individually normalized before being combined to form the color image.

<img width="939" height="513" alt="Screenshot_25-10-2026_16417_" src="https://github.com/user-attachments/assets/97ba434e-8a37-4037-b77a-afd2de015f14" />

### Basic Image Processing Operations :
Beyond visualization, HyperView includes tools for elementary image manipulation.
#### Cropping :
Users can select a region of interest (ROI) on the displayed image using the crop tool. In "Crop" mode, a selection rectangle can be drawn over the image. Once the selection is confirmed (upon releasing the mouse button), the `cropImage` function of the `ImageDisplayWidget` is invoked. This function converts the selection rectangle's coordinates from the widget space to the image space—accounting for zoom, panning, and rotation—and crops the image in memory (updating both the `QPixmap` used for display and the underlying `numpy.ndarray` data within `ImageLoader`) to the dimensions of the rectangle. Metadata is also updated to reflect the new dimensions and the origin of the cropped subset.
#### Undo/Redo Functions :
To facilitate experimentation and allow for the correction of operational errors (primarily those affecting the display, such as cropping), "Undo" and "Redo" features have been implemented. These functions operate on a stack of states for the `ImageDisplayWidget`, where each state corresponds to a `QPixmap` of the displayed image. Currently, this functionality focuses primarily on changes to the displayed `QPixmap` (e.g., following a crop operation) and does not extend to modifications of the underlying `numpy` data used for more complex processing tasks. 6.4. General Transformations and Corrections 
HyperView also offers basic spectral and spatial transformation and correction features. 
#### Hyperspectral-to-Multispectral Transformation (via SRF) :
A useful feature for data comparison or sensor simulation is the transformation of an hyperspectral image into a multispectral image. The `transformToMultispectral` function in `main_window.py` enables this operation. 
The user selects a loaded hyperspectral image and a target multispectral sensor (e.g., Landsat 7/8, Sentinel-2A/B). The software then uses the corresponding spectral response functions (SRFs)—defined in `srf_definitions.py`—along with the wavelengths of the hyperspectral image (extracted from its metadata) to calculate the band values ​​for the simulated sensor via spectral convolution [26]. The resulting multispectral image, along with its new effective wavelengths, is then displayed in a new tab.
#### Spatial downsampling :
To reduce data size or to simulate a coarser spatial resolution, a spatial downsampling function (`downsampleHyperspectral` in `main_window.py`) is available. The user specifies a reduction factor (w). The image is then downsampled using a Gaussian filter (`gaussian_down_sample` in `main_window.py`, which applies a Gaussian kernel before selecting the pixels) to minimize aliasing effects and preserve a more faithful representation of the spatial information at the lower resolution. The resulting image is displayed in a new tab.
#### Basic atmospheric corrections (TOA, DOS) :
First-level atmospheric correction methods are accessible via the `applyAtmosphericCorrectionAndDisplay` and `processAndSaveAllReflectiveBands` functions in `main_window.py`. These allow for the conversion of raw digital numbers (DN) from Landsat images into top-of-atmosphere (TOA) reflectance. An option for Dark Object Subtraction (DOS)—a simple empirical method for estimating and subtracting the additive atmospheric contribution—is also provided, leveraging the functionality of the `LandsatBandCalibrator` class (assumed to be part of the `LandsatBasicUtils` module, which is referenced but not provided here). It should be noted that more advanced atmospheric correction methods, such as the Simplified Radiative Transfer Model (SREM), are also integrated; however, their details—along with the specific libraries required (SREMPyLandsat)—are discussed in dedicated chapters, given their complexity and frequent association with specific applications, such as the calculation of Land Surface Temperature (LST).
#### Cloud masking (general reference to FMask) :
Although the detailed implementation is not the focus of this chapter, the software includes an entry point (`fmask_usgsLandsatStacked.py`) for cloud-masking functionality, presumably based on the FMask algorithm or a variant thereof. This feature is crucial for the pre-processing of optical imagery, as it enables the identification and exclusion of pixels contaminated by clouds and their shadows from subsequent analyses [27]. 
#### Image Quality Assessment Modules :
The `full_ref.py` module provides a set of functions for the quantitative assessment of image quality, which is particularly useful for comparing images resulting from different processing methods or simulations [28]. The implemented metrics include:
• Peak Signal-to-Noise Ratio (PSNR).
• Root Mean Square Error (RMSE).
• Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS).
• Universal Image Quality Index (UQI).
• Spectral Angle Mapper (SAM). These metrics are accessible via the "Fusion" menu following certain processes, enabling an objective evaluation of the results.
### Conclusion : 
The HyperView software—a tool developed specifically to address the needs for analyzing and manipulating multispectral and hyperspectral remote sensing imagery within the scope of this thesis. We have detailed the motivations behind its development, its modular software architecture, the features of its graphical user interface—designed for intuitive interaction—and its general functionalities. These include a robust multi-format data-loading system with specialized handling for Landsat scenes; interactive visualization tools for exploring spectral bands and pixel profiles; fundamental image-processing operations, such as cropping and spectral transformation via Spectral Response Functions (SRF); and modules for assessing image quality. The software is built upon proven open-source technologies, primarily the Python ecosystem—utilizing PyQt5 for the interface—along with libraries such as NumPy, SciPy, GDAL, and Matplotlib for data processing and visualization.
