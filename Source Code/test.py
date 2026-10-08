import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QMenuBar

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PyQt5 Menu Example")
        self.setGeometry(100, 100, 600, 400)

        # Create Menu Bar
        menuBar = self.menuBar()

        # ---- Fusion Menu ----
        fusionMenu = menuBar.addMenu("&Fusion")
        fusionAction = QAction("Fusion Process", self)
        fusionAction.triggered.connect(self.runFusionProcess)
        fusionMenu.addAction(fusionAction)

        # ---- Image Menu ----
        imageMenu = menuBar.addMenu("&Image Processing")

        loadImageAction = QAction("Load Image", self)
        loadImageAction.triggered.connect(self.loadImage)

        saveImageAction = QAction("Save Image", self)
        saveImageAction.triggered.connect(self.saveImage)

        imageMenu.addAction(loadImageAction)
        imageMenu.addAction(saveImageAction)

        # ---- Settings Menu with Submenu ----
        settingsMenu = menuBar.addMenu("&Settings")

        languageMenu = settingsMenu.addMenu("Language")  # Submenu inside Settings
        englishAction = QAction("English", self)
        englishAction.triggered.connect(lambda: self.setLanguage("English"))

        frenchAction = QAction("French", self)
        frenchAction.triggered.connect(lambda: self.setLanguage("French"))

        languageMenu.addAction(englishAction)
        languageMenu.addAction(frenchAction)

        # ---- Help Menu ----
        helpMenu = menuBar.addMenu("&Help")
        aboutAction = QAction("About", self)
        aboutAction.triggered.connect(self.showAbout)
        helpMenu.addAction(aboutAction)

    def runFusionProcess(self):
        print("Fusion Process started")

    def loadImage(self):
        print("Load Image function triggered")

    def saveImage(self):
        print("Save Image function triggered")

    def setLanguage(self, lang):
        print(f"Language set to {lang}")

    def showAbout(self):
        print("About Menu Clicked")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
