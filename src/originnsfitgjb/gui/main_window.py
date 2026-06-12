from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .modules.registry import ModuleRegistry
from .settings import GuiSettings


class MainWindow(QMainWindow):
    def __init__(self, registry: ModuleRegistry, settings: GuiSettings) -> None:
        super().__init__()
        self._registry = registry
        self._settings = settings
        self.setWindowTitle("OriginNSFitGJB")
        self._navigation = QListWidget()
        self._navigation.setFixedWidth(180)
        self._pages = QStackedWidget()
        self._build_layout()

    def _build_layout(self) -> None:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.addWidget(self._navigation)
        body_layout.addWidget(self._pages, 1)
        layout.addWidget(body)
        self.setCentralWidget(container)

        for module in self._registry.all_modules():
            item = QListWidgetItem(module.title)
            item.setData(Qt.ItemDataRole.UserRole, module.module_id)
            self._navigation.addItem(item)
            self._pages.addWidget(self._create_module_page(module.title, module.create_page))

        self._navigation.currentRowChanged.connect(self._pages.setCurrentIndex)
        if self._navigation.count():
            self._navigation.setCurrentRow(0)

    def _create_module_page(self, title: str, create_page: Callable[[], object]) -> QWidget:
        page = create_page()
        if not isinstance(page, QWidget):
            raise TypeError(f"GUI module {title} did not create a QWidget.")
        return page
