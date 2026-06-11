from __future__ import annotations

import sys

from .modules.registry import build_default_registry
from .settings import load_settings


def main(argv: list[str] | None = None) -> int:
    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow

    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("OriginNSFitGJB")
    settings = load_settings()
    window = MainWindow(build_default_registry(), settings)
    window.resize(settings.window_width, settings.window_height)
    window.show()
    return app.exec()
