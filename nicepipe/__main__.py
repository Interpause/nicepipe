import sys
from rich import inspect, print
from PySide6.QtUiTools import QUiLoader
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import QFile, Qt
from PySide6.QtWidgets import QApplication
from nicepipe.utils import enable_fancy_console


# import uvicorn


def main():
    # uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")
    with enable_fancy_console():
        app_ui = QApplication()

        ui_file = QFile("ui/main.ui")
        if not ui_file.open(QFile.ReadOnly):
            print(f"Cannot open {ui_file.fileName()}: {ui_file.errorString()}")
            sys.exit(-1)
        loader = QUiLoader()
        window = loader.load(ui_file)
        inspect(window)
        print("potato")
        ui_file.close()
        if not window:
            print(loader.errorString())
            sys.exit(-1)
        window.show()
        sys.exit(app_ui.exec())


if __name__ == "__main__":
    main()
