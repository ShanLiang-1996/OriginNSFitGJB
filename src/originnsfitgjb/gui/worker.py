from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from ..analysis_service import AnalysisConfig, AnalysisProgress, AnalysisRunResult, run_analysis


class AnalysisWorker(QObject):
    progress = Signal(object)
    log = Signal(str)
    finished = Signal(object)

    def __init__(self, config: AnalysisConfig) -> None:
        super().__init__()
        self._config = config

    @Slot()
    def run(self) -> None:
        try:
            result: AnalysisRunResult = run_analysis(
                self._config,
                progress_callback=self._emit_progress,
                log_callback=self.log.emit,
            )
        except Exception as exc:
            message = f"Unexpected GUI worker failure: {exc}"
            self.log.emit(message)
            result = AnalysisRunResult(
                completed=False,
                messages=(message,),
                origin_error=str(exc),
            )
        self.finished.emit(result)

    def _emit_progress(self, event: AnalysisProgress) -> None:
        self.progress.emit(event)
