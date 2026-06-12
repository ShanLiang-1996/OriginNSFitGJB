from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QThread, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...analysis_service import AnalysisConfig, AnalysisProgress, AnalysisRunResult
from ..worker import AnalysisWorker


class Gjb18aPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._thread: QThread | None = None
        self._worker: AnalysisWorker | None = None
        self._output_buttons: list[QPushButton] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(self._build_input_group())
        layout.addWidget(self._build_options_group())

        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        self._run_button = QPushButton("开始全流程分析")
        self._run_button.clicked.connect(self._start_run)
        controls_layout.addWidget(self._run_button)

        self._status_label = QLabel("等待运行")
        controls_layout.addWidget(self._status_label, 1)
        layout.addWidget(controls)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(180)
        layout.addWidget(self._log, 1)

        layout.addWidget(self._build_outputs_group())

    def _build_input_group(self) -> QGroupBox:
        group = QGroupBox("输入与列设置")
        form = QFormLayout(group)

        self._input_dir = QLineEdit("data")
        form.addRow("输入目录", self._path_row(self._input_dir, self._choose_input_dir))

        self._patterns = QLineEdit("*.csv;*.tsv;*.txt;*.xlsx;*.xls")
        form.addRow("文件模式", self._patterns)

        self._output_dir = QLineEdit("output")
        form.addRow("输出目录", self._path_row(self._output_dir, self._choose_output_dir))

        self._life_column = QLineEdit()
        self._life_column.setPlaceholderText("自动识别，或填写 life")
        form.addRow("寿命列", self._life_column)

        self._response_column = QLineEdit()
        self._response_column.setPlaceholderText("自动识别，或填写 strain")
        form.addRow("响应列", self._response_column)

        self._status_column = QLineEdit()
        self._status_column.setPlaceholderText("可选，例如 status")
        form.addRow("状态列", self._status_column)

        self._level_column = QLineEdit()
        self._level_column.setPlaceholderText("可选")
        form.addRow("级别列", self._level_column)
        return group

    def _build_options_group(self) -> QGroupBox:
        group = QGroupBox("分析参数与 Origin 选项")
        form = QFormLayout(group)

        self._confidence = QLineEdit("0.95")
        form.addRow("置信度", self._confidence)

        self._fit_points = QSpinBox()
        self._fit_points.setRange(2, 10000)
        self._fit_points.setValue(300)
        form.addRow("拟合点数", self._fit_points)

        self._outlier_mode = QComboBox()
        self._outlier_mode.addItem("自动剔除", "auto")
        self._outlier_mode.addItem("仅报告", "report-only")
        form.addRow("异常值模式", self._outlier_mode)

        self._audit = QCheckBox("写审计输出")
        form.addRow(self._audit)

        self._audit_workbook = QCheckBox("写审计 workbook")
        form.addRow(self._audit_workbook)

        self._audit_json = QCheckBox("写审计 JSON")
        form.addRow(self._audit_json)

        self._hidden_origin = QCheckBox("隐藏 Origin")
        form.addRow(self._hidden_origin)

        self._linearized_graph = QCheckBox("生成线性化图")
        form.addRow(self._linearized_graph)

        self._no_runout_arrows = QCheckBox("隐藏 runout 箭头")
        form.addRow(self._no_runout_arrows)

        self._project_path = QLineEdit()
        self._project_path.setPlaceholderText("留空则写入 output/gjb_analysis.opj")
        form.addRow("Origin 项目路径", self._path_row(self._project_path, self._choose_project_path))

        self._graph_template_path = QLineEdit()
        self._graph_template = self._graph_template_path
        form.addRow(
            "Origin 图模板",
            self._path_row(self._graph_template_path, self._choose_graph_template),
        )
        return group

    def _build_outputs_group(self) -> QGroupBox:
        group = QGroupBox("输出文件")
        layout = QHBoxLayout(group)
        self._outputs_layout = layout
        layout.addStretch(1)
        return group

    def _path_row(self, line_edit: QLineEdit, slot: Callable[[], None]) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        browse = QPushButton("浏览")
        browse.clicked.connect(slot)
        layout.addWidget(line_edit, 1)
        layout.addWidget(browse)
        return row

    def _choose_input_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输入目录", self._input_dir.text())
        if path:
            self._input_dir.setText(path)

    def _choose_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", self._output_dir.text())
        if path:
            self._output_dir.setText(path)

    def _choose_project_path(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "选择 Origin 项目路径",
            self._project_path.text(),
            "Origin Project (*.opj *.opju)",
        )
        if path:
            self._project_path.setText(path)

    def _choose_graph_template(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 Origin 图模板",
            self._graph_template_path.text(),
            "Origin Template (*.otp *.otpu)",
        )
        if path:
            self._graph_template_path.setText(path)

    def _start_run(self) -> None:
        try:
            config = self._config_from_form()
        except ValueError as exc:
            QMessageBox.warning(self, "配置错误", str(exc))
            return

        self._clear_outputs()
        self._log.clear()
        self._run_button.setEnabled(False)
        self._status_label.setText("正在启动分析")

        thread = QThread(self)
        worker = AnalysisWorker(config)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.log.connect(self._append_log)
        worker.finished.connect(self._on_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_worker_refs)

        self._thread = thread
        self._worker = worker
        thread.start()

    def _config_from_form(self) -> AnalysisConfig:
        input_dir = Path(self._input_dir.text().strip() or "data")
        output_dir = Path(self._output_dir.text().strip() or "output")
        if not input_dir.is_dir():
            raise ValueError(f"输入目录不存在：{input_dir}")

        patterns = tuple(
            item.strip()
            for item in self._patterns.text().replace(",", ";").split(";")
            if item.strip()
        )
        return AnalysisConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            patterns=patterns,
            life_column=self._text_or_none(self._life_column),
            response_column=self._text_or_none(self._response_column),
            status_column=self._text_or_none(self._status_column),
            level_column=self._text_or_none(self._level_column),
            confidence=float(self._confidence.text().strip()),
            fit_points=self._fit_points.value(),
            dry_run=False,
            audit=self._audit.isChecked(),
            audit_workbook=self._audit_workbook.isChecked(),
            audit_json=self._audit_json.isChecked(),
            outlier_mode=str(self._outlier_mode.currentData()),
            hidden_origin=self._hidden_origin.isChecked(),
            project_path=self._path_or_none(self._project_path),
            graph_template_path=self._path_or_none(self._graph_template_path),
            linearized_graph=self._linearized_graph.isChecked(),
            no_runout_arrows=self._no_runout_arrows.isChecked(),
        )

    def _on_progress(self, event: AnalysisProgress) -> None:
        if event.total:
            self._status_label.setText(f"{event.message} ({event.current}/{event.total})")
        else:
            self._status_label.setText(event.message)

    def _append_log(self, message: str) -> None:
        self._log.append(message)

    def _on_finished(self, result: AnalysisRunResult) -> None:
        self._run_button.setEnabled(True)
        self._status_label.setText("分析完成" if result.completed else "分析失败")
        if result.origin_error:
            QMessageBox.warning(
                self,
                "Origin 自动化异常",
                "Python 输出已完成，但 Origin 生成失败。请查看 origin_automation.log。",
            )
        for path in result.output_paths:
            self._add_output_button(path)

    def _add_output_button(self, path: Path) -> None:
        button = QPushButton(path.name)
        button.setToolTip(str(path))
        button.clicked.connect(
            lambda checked=False, output_path=path: QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(output_path.resolve()))
            )
        )
        insert_index = max(0, self._outputs_layout.count() - 1)
        self._outputs_layout.insertWidget(insert_index, button)
        self._output_buttons.append(button)

    def _clear_outputs(self) -> None:
        for button in self._output_buttons:
            self._outputs_layout.removeWidget(button)
            button.deleteLater()
        self._output_buttons.clear()

    def _clear_worker_refs(self) -> None:
        self._thread = None
        self._worker = None

    @staticmethod
    def _text_or_none(line_edit: QLineEdit) -> str | None:
        value = line_edit.text().strip()
        return value or None

    @staticmethod
    def _path_or_none(line_edit: QLineEdit) -> Path | None:
        value = line_edit.text().strip()
        return Path(value) if value else None
