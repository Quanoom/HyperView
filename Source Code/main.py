import sys
import os
import time
from PyQt5.QtWidgets import QApplication, QSplashScreen
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt
from main_window import MainWindow
from dialogs import StartupDialog

def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("hyperview_icon.ico"))

    splash_pix = QPixmap("splash.png")
    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.show()
    app.processEvents()
    time.sleep(2)
    splash.finish(None)

    recent = []
    if os.path.exists("recent.txt"):
        with open("recent.txt", "r") as f:
            recent = [line.strip() for line in f if line.strip()]

    startup = StartupDialog(recent)
    if startup.exec_() == StartupDialog.Accepted:
        selected = startup.selected_file
    else:
        selected = None

    if selected:
        if selected not in recent:
            recent.insert(0, selected)
            with open("recent.txt", "w") as f:
                for path in recent[:10]:
                    f.write(path + "\n")

    main_window = MainWindow(selected)
    main_window.resize(1000, 600)
    main_window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()