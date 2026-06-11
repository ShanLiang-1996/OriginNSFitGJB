# GUI Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PySide6 Windows desktop workbench for OriginNSFitGJB with a first `GJB/Z 18A 分析` module, while keeping the existing CLI behavior intact.

**Architecture:** Extract the current CLI orchestration into a GUI-independent service layer, then make CLI and GUI call that service. Add a small GUI module registry so the workbench can grow by adding modules instead of rewriting the main window.

**Tech Stack:** Python 3.10+, pandas/numpy/scipy/openpyxl/originpro/originpy/pywin32, PySide6, PyInstaller, unittest/pytest-compatible tests.

---

## File Structure

- Create `src/originnsfitgjb/analysis_service.py`: service dataclasses, progress events, output writing helpers, and `run_analysis`.
- Modify `src/originnsfitgjb/cli.py`: keep argparse and console output, delegate execution to `analysis_service`.
- Create `tests/test_analysis_service.py`: service-level dry-run, progress, failure, and CLI compatibility tests.
- Create `src/originnsfitgjb/gui/__init__.py`: GUI package marker.
- Create `src/originnsfitgjb/gui/settings.py`: JSON settings load/save and default settings path.
- Create `tests/test_gui_settings.py`: settings round-trip tests that do not import PySide6.
- Create `src/originnsfitgjb/gui/modules/__init__.py`: module package marker.
- Create `src/originnsfitgjb/gui/modules/base.py`: module metadata dataclass and protocol-like page factory type.
- Create `src/originnsfitgjb/gui/modules/registry.py`: module registry helpers.
- Create `src/originnsfitgjb/gui/modules/gjb18a.py`: module definition for the first GUI module.
- Create `tests/test_gui_module_registry.py`: registry tests that do not import PySide6.
- Create `src/originnsfitgjb/gui/worker.py`: Qt worker object that runs `run_analysis` off the UI thread.
- Create `src/originnsfitgjb/gui/main_window.py`: main workbench window and navigation.
- Create `src/originnsfitgjb/gui/modules/gjb18a_page.py`: GJB form, run button, progress, log, and output links.
- Create `src/originnsfitgjb/gui/app.py`: Qt application bootstrap.
- Create `src/originnsfitgjb/gui/__main__.py`: `python -m originnsfitgjb.gui` entrypoint.
- Create `scripts/run_originnsfitgjb_gui.py`: PyInstaller GUI script entrypoint.
- Create `requirements-gui.txt`: GUI dependency list.
- Modify `requirements-build.txt`: include GUI dependencies for packaging.
- Create `OriginNSFitGJB-GUI.spec`: PyInstaller spec for a windowed GUI executable.
- Modify `README.md` and `docs/user_manual.md`: document GUI usage and packaging.

## Task 1: Add Service Types And Configuration Helpers

**Files:**
- Create: `src/originnsfitgjb/analysis_service.py`
- Create: `tests/test_analysis_service.py`

- [ ] **Step 1: Write failing tests for service config defaults and path helpers**

Add this initial test file:

```python
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from originnsfitgjb.analysis_service import AnalysisConfig


class AnalysisServiceConfigTests(unittest.TestCase):
    def test_default_patterns_match_cli_defaults(self) -> None:
        config = AnalysisConfig()

        self.assertEqual(
            config.patterns,
            ("*.csv", "*.tsv", "*.txt", "*.xlsx", "*.xls"),
        )

    def test_audit_dir_defaults_under_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            config = AnalysisConfig(output_dir=output_dir)

            self.assertEqual(config.resolved_audit_dir(), output_dir / "audit")

    def test_explicit_audit_dir_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audit_dir = Path(tmp) / "review"
            config = AnalysisConfig(audit_dir=audit_dir)

            self.assertEqual(config.resolved_audit_dir(), audit_dir)

    def test_project_path_defaults_under_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            config = AnalysisConfig(output_dir=output_dir)

            self.assertEqual(config.resolved_project_path(), output_dir / "gjb_analysis.opj")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_analysis_service.AnalysisServiceConfigTests
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` because `originnsfitgjb.analysis_service` does not exist yet.

- [ ] **Step 3: Create service dataclasses and helper methods**

Create `src/originnsfitgjb/analysis_service.py` with this starter content:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal


DEFAULT_PATTERNS = ("*.csv", "*.tsv", "*.txt", "*.xlsx", "*.xls")
ProgressPhase = Literal[
    "discover",
    "fit",
    "write_outputs",
    "write_audit",
    "origin",
    "complete",
    "failed",
]


@dataclass(frozen=True)
class AnalysisConfig:
    input_dir: Path = Path("data")
    output_dir: Path = Path("output")
    patterns: tuple[str, ...] = DEFAULT_PATTERNS
    life_column: str | None = None
    response_column: str | None = None
    status_column: str | None = None
    level_column: str | None = None
    replicate_decimals: int = 8
    confidence: float = 0.95
    fit_points: int = 300
    symbol_kind: int = 2
    dry_run: bool = False
    audit: bool = False
    audit_workbook: bool = False
    audit_json: bool = False
    outlier_mode: str = "auto"
    audit_dir: Path | None = None
    hidden_origin: bool = False
    project_path: Path | None = None
    graph_template_path: Path | None = None
    no_graph_template: bool = False
    linearized_graph: bool = False
    no_runout_arrows: bool = False

    def resolved_audit_dir(self) -> Path:
        return self.audit_dir or (self.output_dir / "audit")

    def resolved_project_path(self) -> Path:
        return self.project_path or (self.output_dir / "gjb_analysis.opj")


@dataclass(frozen=True)
class AnalysisProgress:
    phase: ProgressPhase
    message: str
    current: int = 0
    total: int = 0


@dataclass(frozen=True)
class AnalysisTableFailure:
    label: str
    message: str


@dataclass(frozen=True)
class AnalysisRunResult:
    completed: bool
    output_paths: tuple[Path, ...] = ()
    table_failures: tuple[AnalysisTableFailure, ...] = ()
    messages: tuple[str, ...] = ()
    origin_error: str | None = None


ProgressCallback = Callable[[AnalysisProgress], None]
LogCallback = Callable[[str], None]


def emit_progress(
    callback: ProgressCallback | None,
    phase: ProgressPhase,
    message: str,
    *,
    current: int = 0,
    total: int = 0,
) -> None:
    if callback is not None:
        callback(AnalysisProgress(phase=phase, message=message, current=current, total=total))


def emit_log(callback: LogCallback | None, message: str, sink: list[str]) -> None:
    sink.append(message)
    if callback is not None:
        callback(message)
```

- [ ] **Step 4: Run config tests to verify they pass**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_analysis_service.AnalysisServiceConfigTests
```

Expected: PASS.

- [ ] **Step 5: Commit service type scaffold**

Run:

```powershell
git add src\originnsfitgjb\analysis_service.py tests\test_analysis_service.py
git commit -m "Add analysis service configuration types"
```

## Task 2: Move CLI Orchestration Into The Service

**Files:**
- Modify: `src/originnsfitgjb/analysis_service.py`
- Modify: `src/originnsfitgjb/cli.py`
- Modify: `tests/test_analysis_service.py`

- [ ] **Step 1: Add failing dry-run service test**

Append this test class to `tests/test_analysis_service.py`:

```python
import shutil

from originnsfitgjb.analysis_service import run_analysis


ROOT = Path(__file__).resolve().parents[1]


class AnalysisServiceRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="analysis_service_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_dry_run_writes_expected_csv_outputs_and_progress(self) -> None:
        output_dir = self.tmpdir / "out"
        progress: list[str] = []
        logs: list[str] = []

        result = run_analysis(
            AnalysisConfig(
                input_dir=ROOT / "examples",
                output_dir=output_dir,
                patterns=("gjb18a_strain_example.csv",),
                status_column="status",
                dry_run=True,
            ),
            progress_callback=lambda event: progress.append(event.phase),
            log_callback=logs.append,
        )

        self.assertTrue(result.completed)
        self.assertTrue((output_dir / "gjb_summary.csv").exists())
        self.assertTrue((output_dir / "gjb_fit_data.csv").exists())
        self.assertTrue((output_dir / "gjb_curve.csv").exists())
        self.assertIn("discover", progress)
        self.assertIn("fit", progress)
        self.assertIn("write_outputs", progress)
        self.assertIn("complete", progress)
        self.assertTrue(any("Wrote" in message for message in logs))
```

- [ ] **Step 2: Run the new test to verify it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_analysis_service.AnalysisServiceRunTests.test_dry_run_writes_expected_csv_outputs_and_progress
```

Expected: FAIL with `ImportError` for `run_analysis`.

- [ ] **Step 3: Move orchestration code from `cli.py` into `analysis_service.py`**

Copy the current helper functions from `src/originnsfitgjb/cli.py` into `analysis_service.py`:

```python
import re
import traceback

import pandas as pd

from .audit import AuditRecord, write_audit_outputs
from .data_loader import discover_files, read_table, strain_life_columns
from .gjb import GJBFit, fit_gjb18a
from .origin_client import OriginAutomationError, OriginClient, OriginGJBJob
```

Then add `run_analysis` above the helper functions:

```python
def run_analysis(
    config: AnalysisConfig,
    *,
    progress_callback: ProgressCallback | None = None,
    log_callback: LogCallback | None = None,
) -> AnalysisRunResult:
    output_dir = config.output_dir
    figures_dir = output_dir / "figures"
    messages: list[str] = []
    failures: list[AnalysisTableFailure] = []
    origin_error: str | None = None

    output_dir.mkdir(parents=True, exist_ok=True)
    emit_progress(progress_callback, "discover", f"正在发现输入文件：{config.input_dir}")
    files = discover_files(config.input_dir, list(config.patterns or DEFAULT_PATTERNS))
    if not files:
        message = f"No supported data files found in {config.input_dir}."
        emit_log(log_callback, message, messages)
        emit_progress(progress_callback, "failed", message)
        return AnalysisRunResult(completed=False, messages=tuple(messages))

    summaries: list[dict[str, object]] = []
    fit_frames: list[pd.DataFrame] = []
    runout_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    level_frames: list[pd.DataFrame] = []
    extra_table_frames: dict[str, list[pd.DataFrame]] = {}
    origin_jobs: list[OriginGJBJob] = []
    audit_records: list[AuditRecord] = []

    emit_progress(progress_callback, "fit", "正在执行 GJB/Z 18A 拟合", total=len(files))
    for file_index, path in enumerate(files, start=1):
        for table in read_table(path):
            try:
                life_column, response_column = strain_life_columns(
                    table.frame,
                    config.life_column,
                    config.response_column,
                )
                label = _safe_name(table.label)
                fit = fit_gjb18a(
                    table.frame,
                    life_column,
                    response_column,
                    confidence=config.confidence,
                    fit_points=config.fit_points,
                    status_column=config.status_column,
                    level_column=config.level_column,
                    replicate_decimals=config.replicate_decimals,
                    outlier_mode=config.outlier_mode,
                    source_file=str(path),
                    source_sheet=table.sheet or "",
                    source_group=table.group or "",
                    source_label=label,
                )
            except Exception as exc:
                label = _safe_name(table.label)
                message = f"GJB analysis failed for {table.label}: {exc}"
                failures.append(AnalysisTableFailure(label=label, message=str(exc)))
                emit_log(log_callback, message, messages)
                continue

            title = str(table.group or table.sheet or path.stem)
            summaries.append(
                _gjb_summary_record(
                    fit,
                    path,
                    table.sheet,
                    table.group,
                    label,
                    life_column,
                    response_column,
                )
            )
            fit_frames.append(_with_metadata(fit.data, path, table.sheet, table.group, label))
            if fit.runout_data is not None and not fit.runout_data.empty:
                runout_frames.append(_with_metadata(fit.runout_data, path, table.sheet, table.group, label))
            curve_frames.append(_with_metadata(fit.curve, path, table.sheet, table.group, label))
            level_frames.append(_with_metadata(fit.level_stats, path, table.sheet, table.group, label))
            if fit.extra_tables:
                for name, frame in fit.extra_tables.items():
                    extra_table_frames.setdefault(name, []).append(
                        _with_metadata(frame, path, table.sheet, table.group, label)
                    )
            origin_jobs.append(OriginGJBJob(fit=fit, label=label, title=title))
            audit_records.append(
                AuditRecord(
                    label=label,
                    file=str(path),
                    sheet=table.sheet or "",
                    group=table.group or "",
                    fit=fit,
                )
            )
        emit_progress(
            progress_callback,
            "fit",
            f"已处理 {path.name}",
            current=file_index,
            total=len(files),
        )

    if not summaries:
        message = "No GJB analyses were completed."
        emit_log(log_callback, message, messages)
        emit_progress(progress_callback, "failed", message)
        return AnalysisRunResult(
            completed=False,
            table_failures=tuple(failures),
            messages=tuple(messages),
        )

    emit_progress(progress_callback, "write_outputs", "正在写入 CSV 输出")
    summary_frame = pd.DataFrame(summaries)
    output_paths = _write_gjb_outputs(
        summary_frame,
        fit_frames,
        runout_frames,
        curve_frames,
        level_frames,
        extra_table_frames,
        output_dir,
    )

    audit_enabled = bool(config.audit or config.audit_workbook or config.audit_json)
    if audit_enabled:
        emit_progress(progress_callback, "write_audit", "正在写入审计输出")
        audit_paths = write_audit_outputs(
            audit_records,
            summary_frame,
            config.resolved_audit_dir(),
            write_json=bool(config.audit or config.audit_json),
            write_workbook=bool(config.audit or config.audit_workbook),
        )
        output_paths.extend(audit_paths)

    if not config.dry_run:
        emit_progress(progress_callback, "origin", "正在生成 Origin 项目")
        origin = None
        try:
            origin = OriginClient(visible=not config.hidden_origin).__enter__()
            saved_project, figure_records = origin.create_gjb_project(
                origin_jobs,
                summary_frame,
                config.resolved_project_path(),
                figures_dir=figures_dir,
                symbol_kind=config.symbol_kind,
                graph_template_path=config.graph_template_path,
                use_default_graph_template=not config.no_graph_template,
                include_linearized_graph=config.linearized_graph,
                show_runout_arrows=not config.no_runout_arrows,
            )
            _merge_origin_outputs(summaries, saved_project, figure_records)
            summary_frame = pd.DataFrame(summaries)
            summary_frame.to_csv(output_dir / "gjb_summary.csv", index=False, encoding="utf-8-sig")
            emit_log(log_callback, f"Wrote Origin project {saved_project}", messages)
        except OriginAutomationError as exc:
            origin_error = str(exc)
            emit_log(log_callback, f"Origin automation disabled: {exc}", messages)
            _write_origin_automation_log(output_dir, str(exc))
        except Exception:
            origin_error = traceback.format_exc()
            emit_log(log_callback, "Origin automation failed; see origin_automation.log for details.", messages)
            _write_origin_automation_log(output_dir, origin_error)
        finally:
            if origin is not None:
                origin.__exit__(None, None, None)

    for output_path in output_paths:
        if output_path.exists():
            emit_log(log_callback, f"Wrote {output_path}", messages)

    emit_progress(progress_callback, "complete", "分析完成")
    return AnalysisRunResult(
        completed=True,
        output_paths=tuple(path for path in output_paths if path.exists()),
        table_failures=tuple(failures),
        messages=tuple(messages),
        origin_error=origin_error,
    )
```

Move these helpers unchanged from `cli.py` to `analysis_service.py`: `_gjb_summary_record`, `_with_metadata`, `_write_gjb_outputs`, `_merge_origin_outputs`, `_write_origin_automation_log`, and `_safe_name`.

- [ ] **Step 4: Replace CLI orchestration with service delegation**

In `src/originnsfitgjb/cli.py`, keep `build_parser` and `main`, remove the helper implementations moved to the service, and change imports to:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from .analysis_service import AnalysisConfig, run_analysis
```

Replace `run_gjb_analysis` with:

```python
def run_gjb_analysis(args: argparse.Namespace) -> int:
    config = AnalysisConfig(
        input_dir=args.input,
        output_dir=args.output,
        patterns=tuple(args.pattern or ()),
        life_column=args.life,
        response_column=args.response,
        status_column=args.status,
        level_column=args.level,
        replicate_decimals=args.replicate_decimals,
        confidence=args.confidence,
        fit_points=args.fit_points,
        symbol_kind=args.symbol_kind,
        dry_run=args.dry_run,
        audit=args.audit,
        audit_workbook=args.audit_workbook,
        audit_json=args.audit_json,
        outlier_mode=args.outlier_mode,
        audit_dir=args.audit_dir,
        hidden_origin=args.hidden_origin,
        project_path=args.project,
        graph_template_path=args.graph_template,
        no_graph_template=args.no_graph_template,
        linearized_graph=args.linearized_graph,
        no_runout_arrows=args.no_runout_arrows,
    )
    result = run_analysis(config, log_callback=print)
    return 0 if result.completed else 1
```

Update `AnalysisConfig.patterns` inside `analysis_service.py` so empty tuples fall back to `DEFAULT_PATTERNS`:

```python
def resolved_patterns(self) -> tuple[str, ...]:
    return self.patterns or DEFAULT_PATTERNS
```

Then change the service discovery line to:

```python
files = discover_files(config.input_dir, list(config.resolved_patterns()))
```

- [ ] **Step 5: Run focused service and CLI audit tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_analysis_service tests.test_gjb_audit.GJBAuditCliTests.test_dry_run_audit_outputs_and_legacy_csvs
```

Expected: PASS.

- [ ] **Step 6: Run full existing tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Expected: PASS.

- [ ] **Step 7: Commit service extraction**

Run:

```powershell
git add src\originnsfitgjb\analysis_service.py src\originnsfitgjb\cli.py tests\test_analysis_service.py
git commit -m "Extract reusable analysis service"
```

## Task 3: Add GUI Settings Persistence

**Files:**
- Create: `src/originnsfitgjb/gui/__init__.py`
- Create: `src/originnsfitgjb/gui/settings.py`
- Create: `tests/test_gui_settings.py`

- [ ] **Step 1: Write failing settings tests**

Create `tests/test_gui_settings.py`:

```python
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from originnsfitgjb.gui.settings import GuiSettings, load_settings, save_settings


class GuiSettingsTests(unittest.TestCase):
    def test_settings_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            settings = GuiSettings(
                recent_input_dir="C:/data",
                recent_output_dir="C:/out",
                recent_patterns=("*.csv", "*.xlsx"),
                life_column="life",
                response_column="strain",
                status_column="status",
                audit=True,
                audit_workbook=True,
                hidden_origin=True,
                window_width=1200,
                window_height=760,
            )

            save_settings(settings, path)
            loaded = load_settings(path)

            self.assertEqual(loaded, settings)

    def test_missing_settings_file_returns_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_settings(Path(tmp) / "settings.json")

            self.assertEqual(loaded.recent_patterns, ("*.csv", "*.tsv", "*.txt", "*.xlsx", "*.xls"))
            self.assertEqual(loaded.window_width, 1120)
            self.assertEqual(loaded.window_height, 720)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gui_settings
```

Expected: FAIL because `originnsfitgjb.gui.settings` does not exist.

- [ ] **Step 3: Add settings implementation**

Create `src/originnsfitgjb/gui/__init__.py`:

```python
"""Qt GUI package for OriginNSFitGJB."""
```

Create `src/originnsfitgjb/gui/settings.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any

from ..analysis_service import DEFAULT_PATTERNS


@dataclass(frozen=True)
class GuiSettings:
    recent_input_dir: str = "data"
    recent_output_dir: str = "output"
    recent_patterns: tuple[str, ...] = DEFAULT_PATTERNS
    life_column: str = ""
    response_column: str = ""
    status_column: str = ""
    level_column: str = ""
    confidence: float = 0.95
    fit_points: int = 300
    outlier_mode: str = "auto"
    audit: bool = False
    audit_workbook: bool = False
    audit_json: bool = False
    hidden_origin: bool = False
    linearized_graph: bool = False
    no_runout_arrows: bool = False
    graph_template_path: str = ""
    project_path: str = ""
    window_width: int = 1120
    window_height: int = 720


def default_settings_path() -> Path:
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "OriginNSFitGJB" / "settings.json"
    return Path.home() / ".originnsfitgjb" / "settings.json"


def load_settings(path: Path | None = None) -> GuiSettings:
    settings_path = path or default_settings_path()
    if not settings_path.exists():
        return GuiSettings()
    payload = json.loads(settings_path.read_text(encoding="utf-8-sig"))
    return _settings_from_payload(payload)


def save_settings(settings: GuiSettings, path: Path | None = None) -> None:
    settings_path = path or default_settings_path()
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(settings)
    payload["recent_patterns"] = list(settings.recent_patterns)
    settings_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8-sig",
    )


def _settings_from_payload(payload: dict[str, Any]) -> GuiSettings:
    defaults = asdict(GuiSettings())
    merged = {**defaults, **payload}
    merged["recent_patterns"] = tuple(merged.get("recent_patterns") or DEFAULT_PATTERNS)
    return GuiSettings(**merged)
```

- [ ] **Step 4: Run settings tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gui_settings
```

Expected: PASS.

- [ ] **Step 5: Commit settings persistence**

Run:

```powershell
git add src\originnsfitgjb\gui\__init__.py src\originnsfitgjb\gui\settings.py tests\test_gui_settings.py
git commit -m "Add GUI settings persistence"
```

## Task 4: Add GUI Module Registry

**Files:**
- Create: `src/originnsfitgjb/gui/modules/__init__.py`
- Create: `src/originnsfitgjb/gui/modules/base.py`
- Create: `src/originnsfitgjb/gui/modules/registry.py`
- Create: `src/originnsfitgjb/gui/modules/gjb18a.py`
- Create: `tests/test_gui_module_registry.py`

- [ ] **Step 1: Write failing registry tests**

Create `tests/test_gui_module_registry.py`:

```python
from __future__ import annotations

import unittest

from originnsfitgjb.gui.modules.gjb18a import create_gjb18a_module
from originnsfitgjb.gui.modules.registry import ModuleRegistry, build_default_registry


class GuiModuleRegistryTests(unittest.TestCase):
    def test_default_registry_contains_gjb18a_module(self) -> None:
        registry = build_default_registry()

        module = registry.get("gjb18a")

        self.assertEqual(module.module_id, "gjb18a")
        self.assertEqual(module.title, "GJB/Z 18A 分析")

    def test_register_rejects_duplicate_module_id(self) -> None:
        registry = ModuleRegistry()
        module = create_gjb18a_module()
        registry.register(module)

        with self.assertRaisesRegex(ValueError, "Duplicate GUI module"):
            registry.register(module)

    def test_all_modules_preserve_registration_order(self) -> None:
        registry = ModuleRegistry()
        module = create_gjb18a_module()
        registry.register(module)

        self.assertEqual(registry.all_modules(), (module,))
```

- [ ] **Step 2: Run registry tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gui_module_registry
```

Expected: FAIL because the modules package does not exist.

- [ ] **Step 3: Add module metadata and registry implementation**

Create `src/originnsfitgjb/gui/modules/__init__.py`:

```python
"""Workbench module definitions."""
```

Create `src/originnsfitgjb/gui/modules/base.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol


class PageFactory(Protocol):
    def __call__(self) -> object:
        """Create and return a QWidget instance."""


@dataclass(frozen=True)
class GuiModule:
    module_id: str
    title: str
    description: str
    create_page: PageFactory
```

Create `src/originnsfitgjb/gui/modules/registry.py`:

```python
from __future__ import annotations

from .base import GuiModule
from .gjb18a import create_gjb18a_module


class ModuleRegistry:
    def __init__(self) -> None:
        self._modules: dict[str, GuiModule] = {}

    def register(self, module: GuiModule) -> None:
        if module.module_id in self._modules:
            raise ValueError(f"Duplicate GUI module: {module.module_id}")
        self._modules[module.module_id] = module

    def get(self, module_id: str) -> GuiModule:
        return self._modules[module_id]

    def all_modules(self) -> tuple[GuiModule, ...]:
        return tuple(self._modules.values())


def build_default_registry() -> ModuleRegistry:
    registry = ModuleRegistry()
    registry.register(create_gjb18a_module())
    return registry
```

Create `src/originnsfitgjb/gui/modules/gjb18a.py`:

```python
from __future__ import annotations

from .base import GuiModule


def create_gjb18a_module() -> GuiModule:
    return GuiModule(
        module_id="gjb18a",
        title="GJB/Z 18A 分析",
        description="批量执行 GJB/Z 18A 9.3.2 简化应变寿命拟合并生成 Origin 输出。",
        create_page=_create_page,
    )


def _create_page() -> object:
    from .gjb18a_page import Gjb18aPage

    return Gjb18aPage()
```

- [ ] **Step 4: Run registry tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gui_module_registry
```

Expected: PASS.

- [ ] **Step 5: Commit module registry**

Run:

```powershell
git add src\originnsfitgjb\gui\modules tests\test_gui_module_registry.py
git commit -m "Add GUI module registry"
```

## Task 5: Add PySide6 Dependencies And Entrypoints

**Files:**
- Create: `requirements-gui.txt`
- Modify: `requirements-build.txt`
- Modify: `pyproject.toml`
- Create: `src/originnsfitgjb/gui/app.py`
- Create: `src/originnsfitgjb/gui/__main__.py`
- Create: `scripts/run_originnsfitgjb_gui.py`

- [ ] **Step 1: Add GUI dependency files and script entrypoint metadata**

Create `requirements-gui.txt`:

```text
PySide6>=6.8,<7
```

Change `requirements-build.txt` to:

```text
-r requirements.txt
-r requirements-gui.txt
pyinstaller>=6.10
setuptools>=69
wheel>=0.43
```

Add this entry under `[project.scripts]` in `pyproject.toml`:

```toml
origin-ns-fit-gjb-gui = "originnsfitgjb.gui.app:main"
```

- [ ] **Step 2: Add minimal GUI app bootstrap**

Create `src/originnsfitgjb/gui/app.py`:

```python
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
```

Create `src/originnsfitgjb/gui/__main__.py`:

```python
from .app import main


raise SystemExit(main())
```

Create `scripts/run_originnsfitgjb_gui.py`:

```python
from originnsfitgjb.gui.app import main


raise SystemExit(main())
```

- [ ] **Step 3: Run existing non-Qt tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_analysis_service tests.test_gui_settings tests.test_gui_module_registry
```

Expected: PASS. This command must not require importing PySide6 page modules.

- [ ] **Step 4: Install GUI dependency in the local environment**

Run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gui.txt
```

Expected: pip installs PySide6 successfully or reports it is already installed.

- [ ] **Step 5: Commit dependency and entrypoint scaffold**

Run:

```powershell
git add requirements-gui.txt requirements-build.txt pyproject.toml src\originnsfitgjb\gui\app.py src\originnsfitgjb\gui\__main__.py scripts\run_originnsfitgjb_gui.py
git commit -m "Add GUI entrypoint scaffold"
```

## Task 6: Build Worker And Main Window Skeleton

**Files:**
- Create: `src/originnsfitgjb/gui/worker.py`
- Create: `src/originnsfitgjb/gui/main_window.py`
- Modify: `src/originnsfitgjb/gui/app.py`

- [ ] **Step 1: Add Qt worker object**

Create `src/originnsfitgjb/gui/worker.py`:

```python
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
        result = run_analysis(
            self._config,
            progress_callback=self._emit_progress,
            log_callback=self.log.emit,
        )
        self.finished.emit(result)

    def _emit_progress(self, event: AnalysisProgress) -> None:
        self.progress.emit(event)
```

- [ ] **Step 2: Add main window shell**

Create `src/originnsfitgjb/gui/main_window.py`:

```python
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
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
        row = QWidget()
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        layout.setContentsMargins(0, 0, 0, 0)

        body = QWidget()
        from PySide6.QtWidgets import QHBoxLayout

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
            self._pages.addWidget(module.create_page())

        self._navigation.currentRowChanged.connect(self._pages.setCurrentIndex)
        if self._navigation.count():
            self._navigation.setCurrentRow(0)
```

- [ ] **Step 3: Run app import check**

Run:

```powershell
.\.venv\Scripts\python.exe -c "from originnsfitgjb.gui.app import main; print(main.__name__)"
```

Expected: prints `main`.

- [ ] **Step 4: Commit worker and main window skeleton**

Run:

```powershell
git add src\originnsfitgjb\gui\worker.py src\originnsfitgjb\gui\main_window.py src\originnsfitgjb\gui\app.py
git commit -m "Add GUI workbench shell"
```

## Task 7: Build The GJB Analysis Page

**Files:**
- Create: `src/originnsfitgjb/gui/modules/gjb18a_page.py`
- Modify: `src/originnsfitgjb/gui/main_window.py`
- Modify: `src/originnsfitgjb/gui/settings.py`

- [ ] **Step 1: Create GJB page with form fields and run wiring**

Create `src/originnsfitgjb/gui/modules/gjb18a_page.py`:

```python
from __future__ import annotations

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

        self._run_button = QPushButton("开始全流程分析")
        self._run_button.clicked.connect(self._start_run)
        layout.addWidget(self._run_button)

        self._status_label = QLabel("等待运行")
        layout.addWidget(self._status_label)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(180)
        layout.addWidget(self._log, 1)

        self._outputs = QWidget()
        self._outputs_layout = QHBoxLayout(self._outputs)
        self._outputs_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._outputs)

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
        form.addRow("分组列", self._level_column)
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

        self._graph_template = QLineEdit()
        form.addRow("Origin 图模板", self._path_row(self._graph_template, self._choose_graph_template))
        return group

    def _path_row(self, line_edit: QLineEdit, slot) -> QWidget:
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
        path, _ = QFileDialog.getSaveFileName(self, "选择 Origin 项目路径", self._project_path.text(), "Origin Project (*.opj *.opju)")
        if path:
            self._project_path.setText(path)

    def _choose_graph_template(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 Origin 图模板", self._graph_template.text(), "Origin Template (*.otp *.otpu)")
        if path:
            self._graph_template.setText(path)

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

        self._thread = QThread(self)
        self._worker = AnalysisWorker(config)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(self._append_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _config_from_form(self) -> AnalysisConfig:
        input_dir = Path(self._input_dir.text().strip() or "data")
        output_dir = Path(self._output_dir.text().strip() or "output")
        if not input_dir.exists():
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
            graph_template_path=self._path_or_none(self._graph_template),
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
            QMessageBox.warning(self, "Origin 自动化异常", "Python 输出已完成，但 Origin 生成失败。请查看 origin_automation.log。")
        for path in result.output_paths:
            self._add_output_button(path)

    def _add_output_button(self, path: Path) -> None:
        button = QPushButton(path.name)
        button.clicked.connect(
            lambda checked=False, output_path=path: QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(output_path.resolve()))
            )
        )
        self._outputs_layout.addWidget(button)
        self._output_buttons.append(button)

    def _clear_outputs(self) -> None:
        for button in self._output_buttons:
            self._outputs_layout.removeWidget(button)
            button.deleteLater()
        self._output_buttons.clear()

    @staticmethod
    def _text_or_none(line_edit: QLineEdit) -> str | None:
        value = line_edit.text().strip()
        return value or None

    @staticmethod
    def _path_or_none(line_edit: QLineEdit) -> Path | None:
        value = line_edit.text().strip()
        return Path(value) if value else None
```

- [ ] **Step 2: Run GUI import check**

Run:

```powershell
.\.venv\Scripts\python.exe -c "from originnsfitgjb.gui.modules.gjb18a_page import Gjb18aPage; print(Gjb18aPage.__name__)"
```

Expected: prints `Gjb18aPage`.

- [ ] **Step 3: Launch the GUI locally**

Run:

```powershell
.\.venv\Scripts\python.exe -m originnsfitgjb.gui
```

Expected: a Windows desktop window opens with the `GJB/Z 18A 分析` page. Close the window after confirming the form renders.

- [ ] **Step 4: Run manual full-flow path through GUI**

Use these form values:

```text
输入目录: examples
文件模式: gjb18a_strain_example.csv
输出目录: output/gui-smoke
状态列: status
审计输出: checked
审计 workbook: checked
```

Expected: the log shows written CSV and audit files under `output/gui-smoke`.
If Origin is installed and available, the log also shows the Origin project path. If Origin is not available, the GUI shows the Origin warning and `output/gui-smoke/origin_automation.log` exists.

- [ ] **Step 5: Run automated regression tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Expected: PASS.

- [ ] **Step 6: Commit GJB page**

Run:

```powershell
git add src\originnsfitgjb\gui\modules\gjb18a_page.py src\originnsfitgjb\gui\main_window.py src\originnsfitgjb\gui\settings.py
git commit -m "Add GJB analysis GUI page"
```

## Task 8: Persist GUI Settings From The Page

**Files:**
- Modify: `src/originnsfitgjb/gui/modules/gjb18a_page.py`
- Modify: `src/originnsfitgjb/gui/main_window.py`
- Modify: `src/originnsfitgjb/gui/app.py`

- [ ] **Step 1: Pass settings into the page factory**

Adjust `GuiModule.create_page` in `src/originnsfitgjb/gui/modules/base.py` so page factories accept settings:

```python
from ..settings import GuiSettings


class PageFactory(Protocol):
    def __call__(self, settings: GuiSettings) -> object:
        """Create and return a QWidget instance."""
```

Change `src/originnsfitgjb/gui/modules/gjb18a.py`:

```python
from ..settings import GuiSettings


def _create_page(settings: GuiSettings) -> object:
    from .gjb18a_page import Gjb18aPage

    return Gjb18aPage(settings)
```

Change `MainWindow._build_layout`:

```python
self._pages.addWidget(module.create_page(self._settings))
```

- [ ] **Step 2: Make the page load settings values**

Change the page constructor:

```python
from ..settings import GuiSettings, save_settings


class Gjb18aPage(QWidget):
    def __init__(self, settings: GuiSettings) -> None:
        super().__init__()
        self._settings = settings
        self._thread: QThread | None = None
        self._worker: AnalysisWorker | None = None
        self._output_buttons: list[QPushButton] = []
        self._build_ui()
        self._apply_settings(settings)
```

Add this method:

```python
def _apply_settings(self, settings: GuiSettings) -> None:
    self._input_dir.setText(settings.recent_input_dir)
    self._output_dir.setText(settings.recent_output_dir)
    self._patterns.setText(";".join(settings.recent_patterns))
    self._life_column.setText(settings.life_column)
    self._response_column.setText(settings.response_column)
    self._status_column.setText(settings.status_column)
    self._level_column.setText(settings.level_column)
    self._confidence.setText(str(settings.confidence))
    self._fit_points.setValue(settings.fit_points)
    self._audit.setChecked(settings.audit)
    self._audit_workbook.setChecked(settings.audit_workbook)
    self._audit_json.setChecked(settings.audit_json)
    self._hidden_origin.setChecked(settings.hidden_origin)
    self._linearized_graph.setChecked(settings.linearized_graph)
    self._no_runout_arrows.setChecked(settings.no_runout_arrows)
    self._project_path.setText(settings.project_path)
    self._graph_template.setText(settings.graph_template_path)
    index = self._outlier_mode.findData(settings.outlier_mode)
    if index >= 0:
        self._outlier_mode.setCurrentIndex(index)
```

- [ ] **Step 3: Save settings when a run starts**

Add this method to `Gjb18aPage`:

```python
def _settings_from_form(self) -> GuiSettings:
    patterns = tuple(
        item.strip()
        for item in self._patterns.text().replace(",", ";").split(";")
        if item.strip()
    )
    return GuiSettings(
        recent_input_dir=self._input_dir.text().strip() or "data",
        recent_output_dir=self._output_dir.text().strip() or "output",
        recent_patterns=patterns,
        life_column=self._life_column.text().strip(),
        response_column=self._response_column.text().strip(),
        status_column=self._status_column.text().strip(),
        level_column=self._level_column.text().strip(),
        confidence=float(self._confidence.text().strip()),
        fit_points=self._fit_points.value(),
        outlier_mode=str(self._outlier_mode.currentData()),
        audit=self._audit.isChecked(),
        audit_workbook=self._audit_workbook.isChecked(),
        audit_json=self._audit_json.isChecked(),
        hidden_origin=self._hidden_origin.isChecked(),
        linearized_graph=self._linearized_graph.isChecked(),
        no_runout_arrows=self._no_runout_arrows.isChecked(),
        project_path=self._project_path.text().strip(),
        graph_template_path=self._graph_template.text().strip(),
        window_width=self.window().width(),
        window_height=self.window().height(),
    )
```

At the top of `_start_run`, after config is built, add:

```python
save_settings(self._settings_from_form())
```

Change `src/originnsfitgjb/gui/main_window.py` so closing the window preserves the latest size:

```python
from dataclasses import replace

from .settings import GuiSettings, load_settings, save_settings
```

Add this method to `MainWindow`:

```python
def closeEvent(self, event) -> None:
    settings = load_settings()
    save_settings(
        replace(
            settings,
            window_width=self.width(),
            window_height=self.height(),
        )
    )
    super().closeEvent(event)
```

- [ ] **Step 4: Run settings and registry tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gui_settings tests.test_gui_module_registry
```

Expected: PASS.

- [ ] **Step 5: Commit settings integration**

Run:

```powershell
git add src\originnsfitgjb\gui\modules src\originnsfitgjb\gui\main_window.py src\originnsfitgjb\gui\app.py
git commit -m "Persist GUI analysis settings"
```

## Task 9: Add GUI Packaging

**Files:**
- Create: `OriginNSFitGJB-GUI.spec`
- Modify: `README.md`
- Modify: `docs/user_manual.md`
- Modify: `offline/README.md`

- [ ] **Step 1: Add PyInstaller GUI spec**

Create `OriginNSFitGJB-GUI.spec`:

```python
# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules

hiddenimports = []
for package_name in ("originpro", "OriginExt", "originpy", "win32com", "PySide6"):
    try:
        hiddenimports += collect_submodules(package_name)
    except Exception:
        pass

a = Analysis(
    ["scripts/run_originnsfitgjb_gui.py"],
    pathex=["src"],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="OriginNSFitGJB-GUI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
```

- [ ] **Step 2: Add README GUI usage section**

Add this section to `README.md` after the dry-run command section:

```markdown
## Windows 图形界面

安装 GUI 依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gui.txt
```

启动 GUI：

```powershell
.\.venv\Scripts\python.exe -m originnsfitgjb.gui
```

图形界面首版提供 `GJB/Z 18A 分析` 模块。选择输入目录、输出目录、列名和 Origin 选项后，点击 `开始全流程分析` 会直接执行 CSV/审计输出和 Origin 项目生成。
```

- [ ] **Step 3: Add GUI build command to README**

Add this under the existing packaging section:

```markdown
GUI 打包：

```powershell
.\.venv\Scripts\pyinstaller.exe OriginNSFitGJB-GUI.spec
```

打包结果：

```text
dist\OriginNSFitGJB-GUI.exe
```
```

- [ ] **Step 4: Add user manual GUI notes**

Add this section to `docs/user_manual.md` after Common Commands:

```markdown
## 图形界面

运行：

```powershell
.\.venv\Scripts\python.exe -m originnsfitgjb.gui
```

界面字段与命令行参数一一对应。首版主按钮会直接执行完整流程，包括 Origin 自动化。若 Python 侧 CSV 和审计输出已完成但 Origin 失败，请打开输出目录中的 `origin_automation.log` 查看原因。
```

- [ ] **Step 5: Update offline documentation**

Add this paragraph to `offline/README.md`:

```markdown
如需构建或运行 GUI 版本，离线 wheelhouse 还需要包含 `requirements-gui.txt` 中的 PySide6 相关 wheel。更新 wheelhouse 后再运行离线安装脚本。
```

- [ ] **Step 6: Run packaging syntax check and docs check**

Run:

```powershell
.\.venv\Scripts\python.exe -m py_compile scripts\run_originnsfitgjb_gui.py src\originnsfitgjb\gui\app.py src\originnsfitgjb\gui\main_window.py src\originnsfitgjb\gui\worker.py src\originnsfitgjb\gui\modules\gjb18a_page.py
```

Expected: exit code 0.

- [ ] **Step 7: Commit packaging and docs**

Run:

```powershell
git add OriginNSFitGJB-GUI.spec README.md docs\user_manual.md offline\README.md requirements-gui.txt requirements-build.txt
git commit -m "Add GUI packaging documentation"
```

## Task 10: Final Verification

**Files:**
- Verify all changed files.

- [ ] **Step 1: Run all automated tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Expected: PASS.

- [ ] **Step 2: Run CLI smoke command**

Run:

```powershell
.\.venv\Scripts\python.exe -m originnsfitgjb --input examples --output output\plan-smoke --pattern gjb18a_strain_example.csv --status status --dry-run --audit --audit-workbook
```

Expected: exit code 0 and output files under `output\plan-smoke`, including `gjb_summary.csv` and `audit\gjb_audit_workbook.xlsx`.

- [ ] **Step 3: Run GUI import smoke command**

Run:

```powershell
.\.venv\Scripts\python.exe -c "from originnsfitgjb.gui.app import main; from originnsfitgjb.gui.modules.registry import build_default_registry; print(build_default_registry().get('gjb18a').title)"
```

Expected: prints `GJB/Z 18A 分析`.

- [ ] **Step 4: Build GUI executable**

Run:

```powershell
.\.venv\Scripts\pyinstaller.exe OriginNSFitGJB-GUI.spec
```

Expected: exit code 0 and `dist\OriginNSFitGJB-GUI.exe` exists.

- [ ] **Step 5: Launch packaged GUI**

Run:

```powershell
.\dist\OriginNSFitGJB-GUI.exe
```

Expected: the workbench window opens, shows the `GJB/Z 18A 分析` module, and can be closed cleanly.

- [ ] **Step 6: Check git status**

Run:

```powershell
git status --short
```

Expected: no uncommitted source changes. Generated `output\plan-smoke` and `dist` artifacts should be ignored or removed before final handoff.
