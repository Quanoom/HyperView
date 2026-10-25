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
