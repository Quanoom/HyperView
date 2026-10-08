from PyQt5.QtWidgets import QDialog, QVBoxLayout, QScrollArea, QLabel, QDialogButtonBox, QListWidget, QFileDialog, \
    QMessageBox
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class MetadataDialog(QDialog):
    def __init__(self, metadata_str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Metadata")
        self.resize(600, 400)
        layout = QVBoxLayout(self)
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        content = QLabel(metadata_str)
        content.setWordWrap(True)
        scroll_area.setWidget(content)
        layout.addWidget(scroll_area)
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok)
        buttonBox.accepted.connect(self.accept)
        layout.addWidget(buttonBox)


class PixelSpectrumDialog(QDialog):
    bandSelected = pyqtSignal(int)  # Signal emitted when a band is selected via a plot click.

    def __init__(self, spectrum, band_slider=None, parent=None):
        """
        :param spectrum: The spectrum data to plot.
        :param band_slider: Optional reference to a QSlider in the main UI.
        :param parent: The parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Pixel Spectrum")
        self.setWindowIcon(QIcon("hyperview_icon.ico"))
        self.band_slider = band_slider
        outer_layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(4, 3))
        self.canvas = FigureCanvas(self.figure)
        nav_toolbar = NavigationToolbar(self.canvas, self)
        outer_layout.addWidget(nav_toolbar)
        outer_layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)
        self.ax.plot(spectrum, linestyle='-', marker='', color='blue')
        self.ax.set_title("Pixel Spectrum")
        self.ax.set_xlabel("Band")
        self.ax.set_ylabel("Intensity")
        buttonBox = QDialogButtonBox()
        okButton = buttonBox.addButton(QDialogButtonBox.Ok)
        saveButton = buttonBox.addButton("Save", QDialogButtonBox.ActionRole)
        okButton.clicked.connect(self.accept)
        saveButton.clicked.connect(self.savePlot)
        outer_layout.addWidget(buttonBox)
        self.spectrum = spectrum
        # Connect mouse click event on the canvas.
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)

    def on_plot_click(self, event):
        # Only handle clicks inside the axes.
        if event.inaxes != self.ax:
            return

        if event.xdata is not None:
            print("Plot clicked at xdata =", event.xdata)  # Debug output.
            band = int(round(event.xdata))
            # Remove any previous red vertical lines and text annotations.
            for line in list(self.ax.get_lines()):
                if line.get_linestyle() == '--' and line.get_color() == 'red':
                    line.remove()
            for txt in list(self.ax.texts):
                txt.remove()
            # Draw a red dashed vertical line at the selected band.
            self.ax.axvline(x=band, color='red', linestyle='--')
            # Add a text annotation with a white background for better visibility.
            ylim = self.ax.get_ylim()
            y_pos = ylim[0] + 0.05 * (ylim[1] - ylim[0])
            self.ax.text(band, y_pos, str(band), color='red', fontsize=14,
                         ha='center', va='bottom',
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
            self.canvas.draw_idle()
            # Emit the bandSelected signal.
            self.bandSelected.emit(band)
            # If a band slider was provided, update its value.
            if self.band_slider is not None:
                self.band_slider.setValue(band)

    def savePlot(self):
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Plot", "", "PNG Files (*.png);;All Files (*)")
        if filePath:
            self.figure.savefig(filePath)
            QMessageBox.information(self, "Saved", f"Plot saved to {filePath}")


class StartupDialog(QDialog):
    def __init__(self, recent_list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to HyperView")
        self.setWindowIcon(QIcon("hyperview_icon.ico"))
        self.selected_file = None
        layout = QVBoxLayout(self)
        label = QLabel("Choose an option:")
        layout.addWidget(label)
        if recent_list:
            recentLabel = QLabel("Recent Images:")
            layout.addWidget(recentLabel)
            self.recentList = QListWidget()
            for path in recent_list:
                self.recentList.addItem(path)
            layout.addWidget(self.recentList)
        else:
            self.recentList = None
        buttonBox = QDialogButtonBox()
        openRecentButton = buttonBox.addButton("Open Selected", QDialogButtonBox.AcceptRole)
        openFileButton = buttonBox.addButton("Open Image", QDialogButtonBox.ActionRole)
        continueButton = buttonBox.addButton("Continue without Image", QDialogButtonBox.RejectRole)
        layout.addWidget(buttonBox)
        openRecentButton.clicked.connect(self.openRecent)
        openFileButton.clicked.connect(self.openFile)
        continueButton.clicked.connect(self.continueWithout)

    def openRecent(self):
        if self.recentList and self.recentList.currentItem():
            self.selected_file = self.recentList.currentItem().text()
            self.accept()
        else:
            QMessageBox.warning(self, "No Selection", "Please select a recent image from the list.")

    def openFile(self):
        filePath, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "",
            "All Files (*)"
           )
        if filePath:
            self.selected_file = filePath
            self.accept()

    def continueWithout(self):
        self.selected_file = None
        self.accept()




from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QDialogButtonBox, QPushButton
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices


class GeographicLocationDialog(QDialog):
    """
    Displays geographic location information extracted from image metadata.
    """

    def __init__(self, latitude, longitude, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Geographic Location")
        self.resize(300, 150)
        layout = QVBoxLayout(self)

        # Show latitude and longitude
        label = QLabel(f"Latitude: {latitude}\nLongitude: {longitude}")
        layout.addWidget(label)

        # Provide an option to open the location in Google Maps
        btn_open = QPushButton("Open in Google Maps")
        btn_open.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl(f"https://www.google.com/maps/search/?api=1&query={latitude},{longitude}")
        ))
        layout.addWidget(btn_open)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok)
        buttonBox.accepted.connect(self.accept)
        layout.addWidget(buttonBox)
