from __future__ import annotations

import csv
from contextlib import ExitStack
from dataclasses import dataclass
from importlib import resources
import math
import os
from pathlib import Path
import re
import subprocess
import time

import pandas as pd

from .gjb import GJBFit


class OriginAutomationError(RuntimeError):
    """Raised when Origin automation cannot be completed."""


@dataclass(frozen=True)
class OriginGJBJob:
    fit: GJBFit
    label: str
    title: str


@dataclass(frozen=True)
class ResponsePresentation:
    axis_label: str
    formula_variable: str


RUNOUT_TEXT_ARROW_LOG_OFFSET = 1.055
RUNOUT_TEXT_ARROW_LINEAR_OFFSET_FRACTION = 0.055
RUNOUT_TEXT_ARROW_LOG_Y_OFFSET = 1.106
RUNOUT_TEXT_ARROW_LINEAR_Y_OFFSET_FRACTION = 0.012
RUNOUT_TEXT_ARROW_FONT_SIZE = 18
ORIGIN_PROCESS_NAMES = {"origin.exe", "origin64.exe"}
ORIGIN_EXIT_TIMEOUT_SECONDS = 6.0


class OriginClient:
    def __init__(self, visible: bool = True) -> None:
        try:
            import originpro as op
        except ImportError as exc:
            raise OriginAutomationError(
                "originpro is not installed. Run `pip install -r requirements.txt` first."
            ) from exc

        self._op = op
        self._visible = visible
        self._resource_stack = ExitStack()
        self._origin_pids_before: set[int] = set()
        self._origin_pids_started: set[int] = set()

    def __enter__(self) -> "OriginClient":
        self._origin_pids_before = self._origin_process_ids()
        try:
            self._op.set_show(self._visible)
        except TypeError:
            if self._visible:
                self._op.set_show()
        self._remember_started_origin_processes()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            self._op.exit()
        except Exception:
            pass
        self._resource_stack.close()
        self._wait_for_started_origin_to_exit()
        self._terminate_lingering_started_origin_processes()

    def create_gjb_project(
        self,
        jobs: list[OriginGJBJob],
        summary: pd.DataFrame,
        output_path: Path,
        figures_dir: Path | None = None,
        symbol_kind: int = 3,
        graph_template_path: Path | None = None,
        use_default_graph_template: bool = True,
        include_linearized_graph: bool = False,
        show_runout_arrows: bool = True,
    ) -> tuple[Path, list[dict[str, str]]]:
        # Step 1 - Start from a clean Origin project before writing GJB results.
        self._op.new(False)
        self._remember_started_origin_processes()
        if not jobs:
            raise OriginAutomationError("No GJB analysis jobs to write to Origin.")

        # Step 2 - Create the project-level summary workbook.
        summary_book = self._op.new_book("w", lname="GJB Summary")
        summary_book.name = self._safe_origin_short_name("GJBSummary")
        summary_wks = summary_book[0]
        summary_wks.name = "Summary"
        self._write_frame_to_sheet(summary_wks, summary)
        self._remove_default_blank_workbook()

        figure_records: list[dict[str, str]] = []
        # Step 3 - Write one workbook and one or two graphs for each input table.
        for job in jobs:
            book = self._op.new_book("w", lname=job.title)
            book.name = self._safe_origin_short_name(job.label, "GJBData")
            data_wks = book[0]
            data_wks.name = "Data"
            self._write_frame_to_sheet(data_wks, job.fit.data)

            failure_frame = self._failure_plot_data(job.fit)
            failure_wks = book.add_sheet("Failures")
            self._write_frame_to_sheet(failure_wks, failure_frame)

            runout_frame = self._runout_plot_data(job.fit)
            runout_wks = None
            if not runout_frame.empty:
                runout_wks = book.add_sheet("RunOut")
                self._write_frame_to_sheet(runout_wks, runout_frame)

            curve_wks = book.add_sheet("CurveBand")
            self._write_frame_to_sheet(curve_wks, job.fit.curve)

            level_wks = book.add_sheet("Levels")
            self._write_frame_to_sheet(level_wks, job.fit.level_stats)

            job_summary = summary[summary["label"] == job.label] if "label" in summary else summary
            summary_wks = book.add_sheet("Summary")
            self._write_frame_to_sheet(summary_wks, job_summary.reset_index(drop=True))
            self._write_extra_tables_to_book(book, job.fit.extra_tables)

            record: dict[str, str] = {"label": job.label}
            if figures_dir is not None:
                engineering_path = figures_dir / f"{job.label}_gjb_engineering.png"
            else:
                engineering_path = None

            record["engineering_figure"] = str(
                self._plot_gjb_engineering(
                    job,
                    failure_wks,
                    failure_frame,
                    curve_wks,
                    runout_wks,
                    runout_frame,
                    engineering_path,
                    symbol_kind,
                    graph_template_path,
                    use_default_graph_template,
                    show_runout_arrows,
                )
                or ""
            )
            record["linearized_figure"] = ""
            if include_linearized_graph:
                linearized_path = (
                    figures_dir / f"{job.label}_gjb_linearized.png"
                    if figures_dir is not None
                    else None
                )
                record["linearized_figure"] = str(
                    self._plot_gjb_linearized(
                        job,
                        failure_wks,
                        failure_frame,
                        curve_wks,
                        runout_wks,
                        runout_frame,
                        linearized_path,
                        symbol_kind,
                        show_runout_arrows,
                    )
                    or ""
                )
            elif figures_dir is not None:
                self._remove_stale_figure(figures_dir / f"{job.label}_gjb_linearized.png")
            figure_records.append(record)

        # Step 4 - Save the Origin project after all workbooks and graphs are ready.
        saved_project = self._save_project(output_path)
        return saved_project, figure_records

    def _plot_gjb_engineering(
        self,
        job: OriginGJBJob,
        failure_wks,
        failure_frame: pd.DataFrame,
        curve_wks,
        runout_wks,
        runout_frame: pd.DataFrame,
        output_path: Path | None,
        symbol_kind: int,
        graph_template_path: Path | None,
        use_default_graph_template: bool,
        show_runout_arrows: bool,
    ) -> Path | None:
        # Step 1 - Create or load the engineering graph page.
        graph = self._new_gjb_graph(
            f"{job.title} GJB",
            graph_template_path,
            use_default_graph_template,
        )
        layer = graph[0]
        self._clear_template_text_labels(layer)
        self._clear_layer_plots(layer)
        graph_name = self._graph_name(graph)

        # Step 2 - Plot fitted curve first so data markers stay visible.
        self._plotxy_from_wks(
            curve_wks,
            self._column_index(job.fit.curve, "life_fit"),
            self._column_index(job.fit.curve, "response"),
            plot_code=200,
            target_graph=graph_name,
        )
        self._plotxy_from_wks(
            failure_wks,
            self._column_index(failure_frame, "gjb_life"),
            self._column_index(failure_frame, "gjb_response"),
            plot_code=201,
            target_graph=graph_name,
        )
        has_runout = runout_wks is not None and not runout_frame.empty
        if has_runout:
            self._plotxy_from_wks(
                runout_wks,
                self._column_index(runout_frame, "gjb_life"),
                self._column_index(runout_frame, "gjb_response"),
                plot_code=201,
                target_graph=graph_name,
            )

        graph = self._find_graph(graph_name) or graph
        layer = graph[0]
        plots = self._plot_list(layer)
        fit_plot = plots[0] if len(plots) > 0 else None
        data_plot = plots[1] if len(plots) > 1 else None
        runout_plot = plots[2] if has_runout and len(plots) > 2 else None
        expected_plots = 3 if has_runout else 2
        if len(plots) < expected_plots:
            raise OriginAutomationError(
                f"Origin created only {len(plots)} engineering plot(s) for {job.label}."
            )

        # Step 3 - Style plots and axes after Origin has created plot objects.
        if fit_plot is not None:
            self._safe_plot_cmd(fit_plot, "-c 2", "-w 1000")
        if data_plot is not None:
            self._style_data_plot(data_plot, symbol_kind)
        if runout_plot is not None:
            self._style_runout_plot(runout_plot, symbol_kind)

        self._set_layer_scale(
            layer,
            "log10",
            "log10" if job.fit.result.x_transform == "log" else "linear",
        )
        self._safe_rescale(layer)
        self._set_gjb_engineering_limits(layer, job.fit)
        self._style_grid(layer)
        self._delete_legend(layer)
        self._set_axis_label_text(
            layer,
            "疲劳寿命 N\\-(f) / cycles",
            self._response_presentation(job.fit.result.response_column).axis_label,
        )
        # Step 4 - Add the fitted equation and optional run-out arrows.
        self._add_gjb_engineering_label(layer, job)
        if has_runout and show_runout_arrows:
            self._add_runout_text_arrows(
                layer,
                runout_frame,
                x_column="gjb_life",
                y_column="gjb_response",
                log_x=True,
            )
        if output_path is None:
            return None
        return self._export_graph(graph, output_path)

    def _plot_gjb_linearized(
        self,
        job: OriginGJBJob,
        failure_wks,
        failure_frame: pd.DataFrame,
        curve_wks,
        runout_wks,
        runout_frame: pd.DataFrame,
        output_path: Path | None,
        symbol_kind: int,
        show_runout_arrows: bool,
    ) -> Path | None:
        # Step 1 - Create the diagnostic linearized graph page.
        graph = self._op.new_graph(lname=f"{job.title} GJB Linearized")
        if graph is None:
            raise OriginAutomationError("Origin did not create a linearized graph page.")
        layer = graph[0]
        self._clear_layer_plots(layer)
        graph_name = self._graph_name(graph)

        # Step 2 - Plot the linearized fit and observed points.
        self._plotxy_from_wks(
            curve_wks,
            self._column_index(job.fit.curve, "gjb_x"),
            self._column_index(job.fit.curve, "log10_life_fit"),
            plot_code=200,
            target_graph=graph_name,
        )
        self._plotxy_from_wks(
            failure_wks,
            self._column_index(failure_frame, "gjb_x"),
            self._column_index(failure_frame, "gjb_y_log10_life"),
            plot_code=201,
            target_graph=graph_name,
        )
        has_runout = runout_wks is not None and not runout_frame.empty
        if has_runout:
            self._plotxy_from_wks(
                runout_wks,
                self._column_index(runout_frame, "gjb_x"),
                self._column_index(runout_frame, "gjb_y_log10_life"),
                plot_code=201,
                target_graph=graph_name,
            )

        graph = self._find_graph(graph_name) or graph
        layer = graph[0]
        plots = self._plot_list(layer)
        fit_plot = plots[0] if len(plots) > 0 else None
        data_plot = plots[1] if len(plots) > 1 else None
        runout_plot = plots[2] if has_runout and len(plots) > 2 else None
        expected_plots = 3 if has_runout else 2
        if len(plots) < expected_plots:
            raise OriginAutomationError(
                f"Origin created only {len(plots)} linearized plot(s) for {job.label}."
            )

        if fit_plot is not None:
            self._safe_plot_cmd(fit_plot, "-c 2", "-w 1000")
        if data_plot is not None:
            self._style_data_plot(data_plot, symbol_kind)
        if runout_plot is not None:
            self._style_runout_plot(runout_plot, symbol_kind)

        self._set_layer_scale(layer, "linear", "linear")
        self._safe_rescale(layer)
        self._set_gjb_linearized_limits(layer, job.fit)
        self._style_grid(layer)
        self._delete_legend(layer)
        x_label = self._linearized_x_axis_label(job.fit)
        self._set_axis_label_text(layer, x_label, "log10(N)")
        self._add_gjb_linearized_label(layer, job)
        if has_runout and show_runout_arrows:
            self._add_runout_text_arrows(
                layer,
                runout_frame,
                x_column="gjb_x",
                y_column="gjb_y_log10_life",
                log_x=False,
            )
        if output_path is None:
            return None
        return self._export_graph(graph, output_path)

    def _set_axis_label_text(self, layer, x_text: str, y_text: str) -> None:
        x_label = layer.label("xb")
        if x_label is not None:
            x_label.text = x_text
            self._safe_set_int(x_label, "verbatim", 0)
        y_label = layer.label("yl")
        if y_label is not None:
            y_label.text = y_text
            self._safe_set_int(y_label, "verbatim", 0)

    def _style_grid(self, layer) -> None:
        commands = (
            "axis -ps X G 3;axis -ps Y G 3;",
            "layer.x.grid.show=3;layer.y.grid.show=3;",
            "layer.x.grid.majorcolor=18;layer.y.grid.majorcolor=18;",
            "layer.x.grid.minorcolor=19;layer.y.grid.minorcolor=19;",
            "layer.x.grid.majortype=2;layer.y.grid.majortype=2;",
            "layer.x.grid.minortype=3;layer.y.grid.minortype=3;",
            "layer.x.grid.majorwidth=0.5;layer.y.grid.majorwidth=0.5;",
            "layer.x.grid.minorwidth=0.25;layer.y.grid.minorwidth=0.25;",
        )
        for command in commands:
            self._safe_layer_lt_exec(layer, command)

    def _style_confidence_plot(self, plot) -> None:
        if plot is not None:
            self._safe_plot_cmd(plot, "-c 15", "-w 500")

    def _style_data_plot(self, plot, symbol_kind: int) -> None:
        try:
            plot.symbol_kind = symbol_kind
            plot.symbol_size = 15
            plot.symbol_interior = 1
        except Exception:
            pass
        self._safe_plot_cmd(plot, "-c 1", "-w 1500")

    def _style_runout_plot(self, plot, symbol_kind: int) -> None:
        try:
            plot.symbol_kind = symbol_kind
            plot.symbol_size = 15
            plot.symbol_interior = 1
        except Exception:
            pass
        self._safe_plot_cmd(plot, "-c 1", "-w 1500")

    def _safe_plot_cmd(self, plot, *commands: str) -> None:
        try:
            plot.set_cmd(*commands)
        except Exception:
            pass

    def _safe_layer_lt_exec(self, layer, command: str) -> None:
        self._try_layer_lt_exec(layer, command)

    def _try_layer_lt_exec(self, layer, command: str) -> bool:
        try:
            layer.lt_exec(command)
            return True
        except Exception:
            try:
                self._op.lt_exec(command)
                return True
            except Exception:
                return False

    def _plotxy_from_wks(
        self,
        wks,
        x_col: int,
        y_col: int,
        plot_code: int,
        target_graph: str | None,
    ) -> str:
        x = x_col + 1
        y = y_col + 1
        data_range = f"{self._worksheet_lt_ref(wks)}!({x},{y})"
        if target_graph:
            output_layer = f"[{target_graph}]1!"
        else:
            output_layer = "[<new>]"
        cmd = f"plotxy iy:={data_range} plot:={plot_code} ogl:={output_layer};"
        self._op.lt_exec(cmd)
        if target_graph:
            self._activate_graph(target_graph)
            return target_graph
        active = self._active_origin_window_name()
        if not active:
            raise OriginAutomationError("Origin did not report the new graph name after plotxy.")
        return active

    def _remove_default_blank_workbook(self) -> None:
        try:
            book = self._op.find_book("w", "Book1")
        except Exception:
            book = None
        if book is None:
            return
        try:
            if str(book.name) != "Book1":
                return
        except Exception:
            return
        if self._workbook_has_data(book):
            return
        try:
            book.destroy()
        except Exception:
            try:
                self._op.lt_exec("win -cd Book1;")
            except Exception:
                pass

    @staticmethod
    def _workbook_has_data(book) -> bool:
        try:
            sheets = list(book)
        except Exception:
            return True
        if not sheets:
            return False
        for sheet in sheets:
            try:
                if int(sheet.rows) > 0:
                    return True
            except Exception:
                return True
        return False

    @staticmethod
    def _remove_stale_figure(path: Path) -> None:
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass

    @staticmethod
    def _safe_origin_short_name(text: str, fallback: str = "OriginNSFitGJB") -> str:
        name = re.sub(r"[^A-Za-z0-9_]", "_", str(text)).strip("_")
        if not name:
            name = fallback
        if name[0].isdigit():
            name = f"W_{name}"
        return name[:24]

    @staticmethod
    def _failure_plot_data(fit: GJBFit) -> pd.DataFrame:
        if "gjb_is_failure" not in fit.data.columns:
            return fit.data.copy()
        return fit.data[fit.data["gjb_is_failure"].astype(bool)].copy()

    @staticmethod
    def _runout_plot_data(fit: GJBFit) -> pd.DataFrame:
        if fit.runout_data is not None and not fit.runout_data.empty:
            return fit.runout_data.copy()
        if "gjb_is_failure" not in fit.data.columns:
            return fit.data.iloc[0:0].copy()
        return fit.data[~fit.data["gjb_is_failure"].astype(bool)].copy()

    def _write_extra_tables_to_book(
        self,
        book,
        extra_tables: dict[str, pd.DataFrame] | None,
    ) -> None:
        if not extra_tables:
            return
        used_names: set[str] = set()
        for name, frame in extra_tables.items():
            if frame is None or frame.empty:
                continue
            sheet_name = self._safe_origin_sheet_name(name, used_names)
            try:
                wks = book.add_sheet(sheet_name)
            except Exception:
                continue
            self._write_frame_to_sheet(wks, frame)

    def _remember_started_origin_processes(self) -> None:
        """Track only Origin processes that appeared after this client started."""
        current = self._origin_process_ids()
        self._origin_pids_started.update(current - self._origin_pids_before)

    def _wait_for_started_origin_to_exit(self) -> None:
        """Give OriginExt a short window to close cleanly after op.exit()."""
        if not self._origin_pids_started:
            return
        deadline = time.monotonic() + ORIGIN_EXIT_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if not (self._origin_pids_started & self._origin_process_ids()):
                return
            time.sleep(0.25)

    def _terminate_lingering_started_origin_processes(self) -> None:
        """Force-close only Origin processes started by this automation run."""
        lingering = sorted(self._origin_pids_started & self._origin_process_ids())
        own_pid = os.getpid()
        for pid in lingering:
            if pid == own_pid:
                continue
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=8,
                )
            except Exception:
                continue
        if lingering:
            print(f"Cleaned up lingering Origin process id(s): {', '.join(map(str, lingering))}")

    @staticmethod
    def _origin_process_ids() -> set[int]:
        """Return running Origin application PIDs on Windows without psutil."""
        if os.name != "nt":
            return set()
        try:
            result = subprocess.run(
                ["tasklist", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                check=False,
                timeout=8,
            )
        except Exception:
            return set()
        if result.returncode != 0:
            return set()
        return OriginClient._parse_tasklist_origin_pids(result.stdout)

    @staticmethod
    def _parse_tasklist_origin_pids(output: str) -> set[int]:
        """Parse tasklist CSV output and keep real Origin app processes only."""
        pids: set[int] = set()
        for row in csv.reader(output.splitlines()):
            if len(row) < 2:
                continue
            process_name = row[0].strip().lower()
            if process_name not in ORIGIN_PROCESS_NAMES:
                continue
            try:
                pids.add(int(row[1]))
            except ValueError:
                continue
        return pids

    @staticmethod
    def _safe_origin_sheet_name(text: str, used_names: set[str]) -> str:
        base = re.sub(r"[^A-Za-z0-9_]", "_", str(text)).strip("_") or "Table"
        base = base[:28]
        candidate = base
        index = 1
        while candidate in used_names:
            suffix = f"_{index}"
            candidate = f"{base[:31 - len(suffix)]}{suffix}"
            index += 1
        used_names.add(candidate)
        return candidate

    def _worksheet_lt_ref(self, wks) -> str:
        try:
            book_name = wks.get_book().name
        except Exception as exc:
            raise OriginAutomationError("Could not resolve Origin workbook name.") from exc
        sheet_name = wks.name
        return f"[{book_name}]{sheet_name}"

    def _column_index(self, frame: pd.DataFrame, column: str) -> int:
        try:
            return int(frame.columns.get_loc(column))
        except KeyError as exc:
            raise OriginAutomationError(f"Column not found for Origin plotting: {column}") from exc

    def _graph_name(self, graph) -> str:
        try:
            name = str(graph.name)
            if name:
                return name
        except Exception:
            pass
        try:
            graph.activate()
        except Exception:
            pass
        name = self._active_origin_window_name()
        if not name:
            raise OriginAutomationError("Could not resolve Origin graph name.")
        return name

    def _active_origin_window_name(self) -> str:
        for getter in ("get_lt_str", "lt_str"):
            try:
                func = getattr(self._op, getter)
            except AttributeError:
                continue
            try:
                return str(func("%H"))
            except Exception:
                continue
        return ""

    def _find_graph(self, graph_name: str):
        try:
            return self._op.find_graph(graph_name)
        except Exception:
            return None

    def _activate_graph(self, graph_name: str) -> None:
        try:
            self._op.lt_exec(f'win -a "{graph_name}";')
        except Exception:
            pass

    def _plot_list(self, layer) -> list:
        try:
            return list(layer.plot_list())
        except Exception:
            return []

    def _new_gjb_graph(
        self,
        title: str,
        graph_template_path: Path | None,
        use_default_graph_template: bool,
    ):
        template_path = graph_template_path
        if template_path is None and use_default_graph_template:
            template_path = self._default_gjb_graph_template()
        if template_path is not None and template_path.exists():
            try:
                graph = self._op.new_graph(lname=title, template=str(template_path.resolve()))
                if graph is not None:
                    return graph
            except Exception as exc:
                print(f"Origin graph template skipped for compatibility: {exc}")
        graph = self._op.new_graph(lname=title)
        if graph is None:
            raise OriginAutomationError("Origin did not create a graph page.")
        return graph

    def _default_gjb_graph_template(self) -> Path | None:
        template = resources.files("originnsfitgjb").joinpath("templates/gjb_graph.otpu")
        if not template.is_file():
            return None
        stack = ExitStack()
        self._resource_stack.enter_context(stack)
        return stack.enter_context(resources.as_file(template))

    def _clear_template_text_labels(self, layer) -> None:
        for name in ("Text", "Text1", "Text2", "Text3", "Label", "Label1"):
            try:
                label = layer.label(name)
                if label is not None:
                    layer.remove_label(label)
            except Exception:
                pass

    def _clear_layer_plots(self, layer) -> None:
        try:
            plots = layer.plot_list()
        except Exception:
            return
        for index in range(len(plots) - 1, -1, -1):
            try:
                layer.remove_plot(index)
            except Exception:
                pass

    def _set_gjb_engineering_limits(self, layer, fit: GJBFit) -> None:
        runout_frame = self._runout_plot_data(fit)
        x_candidates = [
            float(fit.curve["life_fit"].min()),
            float(fit.curve["life_fit"].max()),
            fit.result.life_min,
            fit.result.life_max,
        ]
        y_candidates = [fit.result.response_min, fit.result.response_max]
        if not runout_frame.empty:
            runout_life = runout_frame["gjb_life"].astype(float)
            runout_response = runout_frame["gjb_response"].astype(float)
            x_candidates.extend([float(runout_life.min()), float((runout_life * 1.25).max())])
            y_candidates.extend([float(runout_response.min()), float(runout_response.max())])
        x_min = min(x_candidates)
        x_max = max(x_candidates)
        self._safe_set_xlim(layer, *self._expanded_log_limits(x_min, x_max, pad=0.06))
        if fit.result.x_transform == "log":
            self._safe_set_ylim(
                layer,
                *self._expanded_log_limits(
                    min(y_candidates),
                    max(y_candidates),
                    pad=0.08,
                )
            )
        else:
            self._safe_set_ylim(
                layer,
                *self._expanded_linear_limits(
                    min(y_candidates),
                    max(y_candidates),
                    pad=0.08,
                )
            )

    def _set_gjb_linearized_limits(self, layer, fit: GJBFit) -> None:
        runout_frame = self._runout_plot_data(fit)
        x_min = fit.result.x_min
        x_max = fit.result.x_max
        if not runout_frame.empty:
            runout_x = runout_frame["gjb_x"].astype(float)
            x_min = min(x_min, float(runout_x.min()))
            x_max = max(x_max, float(runout_x.max()))
        self._safe_set_xlim(
            layer,
            *self._expanded_linear_limits(
                x_min,
                x_max,
                pad=0.12 if not runout_frame.empty else 0.06,
            ),
        )
        y_min = min(
            float(fit.data["gjb_y_log10_life"].min()),
            float(fit.curve["log10_life_fit"].min()),
        )
        y_max = max(
            float(fit.data["gjb_y_log10_life"].max()),
            float(fit.curve["log10_life_fit"].max()),
        )
        if not runout_frame.empty:
            runout_y = runout_frame["gjb_y_log10_life"].astype(float)
            y_min = min(y_min, float(runout_y.min()))
            y_max = max(y_max, float(runout_y.max()))
        self._safe_set_ylim(layer, *self._expanded_linear_limits(y_min, y_max, pad=0.08))

    def _delete_legend(self, layer) -> None:
        for name in ("Legend", "legend"):
            try:
                label = layer.label(name)
                if label is not None:
                    layer.remove_label(label)
                    return
            except Exception:
                pass
        try:
            layer.lt_exec("legend -d")
        except Exception:
            pass

    def _add_gjb_linearized_label(self, layer, job: OriginGJBJob) -> None:
        fit = job.fit
        x_position = fit.result.x_min + 0.52 * (fit.result.x_max - fit.result.x_min)
        y_values = fit.data["gjb_y_log10_life"]
        y_min = float(y_values.min())
        y_max = float(y_values.max())
        y_position = y_min + 0.9 * (y_max - y_min)
        text = (
            f"{job.title}\n"
            f"log10(N) = {fit.result.coefficient_a:.6g} "
            f"{self._origin_signed(fit.result.coefficient_b)} X\n"
            f"R\\+(2) = {fit.result.r2:.5f}"
        )
        self._add_layer_label(layer, text, x_position, y_position)

    def _add_gjb_engineering_label(self, layer, job: OriginGJBJob) -> None:
        fit = job.fit
        x_position = 10 ** (
            0.72 * self._safe_log10(fit.result.life_min)
            + 0.28 * self._safe_log10(fit.result.life_max)
        )
        if fit.result.x_transform == "log":
            y_position = 10 ** (
                0.05 * self._safe_log10(fit.result.response_min)
                + 0.95 * self._safe_log10(fit.result.response_max)
            )
        else:
            y_position = fit.result.response_min + 0.82 * (
                fit.result.response_max - fit.result.response_min
            )
        text = self._origin_life_response_formula(fit)
        self._add_layer_label(layer, text, x_position, y_position)

    def _add_layer_label(self, layer, text: str, x_position: float, y_position: float) -> None:
        label = layer.add_label(text, x_position, y_position)
        if label is not None:
            self._safe_set_int(label, "verbatim", 0)
            self._safe_set_int(label, "attach", 2)
            self._safe_set_float(label, "x1", x_position)
            self._safe_set_float(label, "y1", y_position)

    def _add_runout_text_arrows(
        self,
        layer,
        runout_frame: pd.DataFrame,
        *,
        x_column: str,
        y_column: str,
        log_x: bool,
    ) -> None:
        offset = self._runout_text_arrow_offset(runout_frame[x_column], log_x)
        for _, row in runout_frame.iterrows():
            try:
                x_value = float(row[x_column])
                y_value = float(row[y_column])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(x_value) or not math.isfinite(y_value):
                continue
            x_position = x_value * offset if log_x else x_value + offset
            y_position = self._runout_text_arrow_y_position(
                runout_frame[y_column],
                y_value,
                log_y=self._axis_is_log(layer, "y"),
            )
            label = layer.add_label("\\x(2192)", x_position, y_position)
            if label is not None:
                self._safe_set_int(label, "verbatim", 0)
                self._safe_set_int(label, "attach", 2)
                self._safe_set_int(label, "fsize", RUNOUT_TEXT_ARROW_FONT_SIZE)
                self._safe_set_float(label, "x1", x_position)
                self._safe_set_float(label, "y1", y_position)

    @staticmethod
    def _runout_text_arrow_offset(values: pd.Series, log_x: bool) -> float:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if numeric.empty:
            return RUNOUT_TEXT_ARROW_LOG_OFFSET if log_x else RUNOUT_TEXT_ARROW_LINEAR_OFFSET_FRACTION
        if log_x:
            return RUNOUT_TEXT_ARROW_LOG_OFFSET
        span = float(numeric.max() - numeric.min())
        if span <= 0.0:
            span = abs(float(numeric.iloc[0])) or 1.0
        return RUNOUT_TEXT_ARROW_LINEAR_OFFSET_FRACTION * span

    @staticmethod
    def _runout_text_arrow_y_position(
        values: pd.Series,
        y_value: float,
        *,
        log_y: bool,
    ) -> float:
        if log_y and y_value > 0.0:
            return y_value * RUNOUT_TEXT_ARROW_LOG_Y_OFFSET
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if numeric.empty:
            return y_value
        span = float(numeric.max() - numeric.min())
        if span <= 0.0:
            span = abs(float(numeric.iloc[0])) or 1.0
        return y_value + RUNOUT_TEXT_ARROW_LINEAR_Y_OFFSET_FRACTION * span

    def _axis_is_log(self, layer, axis: str) -> bool:
        attr = "yscale" if axis == "y" else "xscale"
        try:
            return str(getattr(layer, attr)).lower() == "log10"
        except Exception:
            return False

    def _export_graph(self, graph, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        graph.activate()
        try:
            exported = graph.save_fig(str(output_path.resolve()), width=1600)
        except Exception:
            exported = graph.save_fig(str(output_path.resolve()))
        exported_path = Path(exported) if exported else output_path
        if not exported_path.exists():
            raise OriginAutomationError(f"Origin did not export figure: {output_path}")
        return exported_path

    def _save_project(self, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        candidates = self._project_save_candidates(output_path)
        for candidate in candidates:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            if self._try_save_project(candidate):
                return candidate
        tried = ", ".join(str(candidate) for candidate in candidates)
        raise OriginAutomationError(f"Origin did not save project. Tried: {tried}")

    def _project_save_candidates(self, output_path: Path) -> list[Path]:
        suffix = output_path.suffix.lower()
        if suffix == ".opj":
            return [output_path, output_path.with_suffix(".opju")]
        if suffix == ".opju":
            return [output_path, output_path.with_suffix(".opj")]
        if suffix:
            return [output_path, output_path.with_suffix(".opj"), output_path.with_suffix(".opju")]
        return [output_path.with_suffix(".opj"), output_path.with_suffix(".opju")]

    def _try_save_project(self, output_path: Path) -> bool:
        try:
            saved = self._op.save(str(output_path.resolve()))
        except Exception:
            return False
        return bool(saved and output_path.exists())

    def _origin_life_response_formula(self, fit: GJBFit) -> str:
        if fit.result.x_transform != "log":
            return fit.result.life_response_formula
        variable = self._response_presentation(fit.result.response_column).formula_variable
        shifted_variable = self._origin_shifted_variable(variable, fit.result.coefficient_c)
        return (
            f"N\\-(f) = {fit.result.life_response_coefficient_a:.6g} "
            f"* {shifted_variable}\\+({fit.result.life_response_coefficient_b:.6g})"
        )

    def _origin_shifted_variable(self, variable: str, coefficient_c: float | None) -> str:
        if coefficient_c is None:
            return f"({variable})"
        if coefficient_c < 0:
            return f"({variable} + {abs(coefficient_c):.6g})"
        return f"({variable} - {coefficient_c:.6g})"

    def _linearized_x_axis_label(self, fit: GJBFit) -> str:
        return "log10(strain - A4)"

    def _write_frame_to_sheet(self, sheet, frame: pd.DataFrame) -> None:
        try:
            sheet.from_df(frame)
            return
        except Exception as exc:
            print(f"Origin worksheet DataFrame import fallback: {exc}")

        for column_index, column in enumerate(frame.columns):
            values = [
                "" if pd.isna(value) else value
                for value in frame[column].tolist()
            ]
            sheet.from_list(column_index, values, str(column))

    def _response_presentation(self, response_column: str) -> ResponsePresentation:
        text = str(response_column).strip()
        lowered = text.lower().replace("_", " ").replace("-", " ")
        is_stress = "\u5e94\u529b" in text or "stress" in lowered or "sigma" in lowered
        is_max = any(token in text for token in ("\u6700\u5927", "\u5cf0\u503c")) or any(
            token in lowered for token in ("max", "maximum", "peak")
        )
        is_amplitude = "\u5e45" in text or "amplitude" in lowered or lowered.endswith(" amp")

        if is_stress:
            greek = "\\x(03C3)"
            if is_max:
                return ResponsePresentation(f"maximum stress {greek}\\-(max)", f"{greek}\\-(max)")
            if is_amplitude:
                return ResponsePresentation(f"{text} {greek}\\-(a)", f"{greek}\\-(a)")
            return ResponsePresentation(f"{text} {greek}", greek)

        greek = "\\x(03B5)"
        if is_max:
            return ResponsePresentation(f"maximum strain {greek}\\-(max)", f"{greek}\\-(max)")
        if is_amplitude:
            return ResponsePresentation(f"{text} {greek}\\-(a)", f"{greek}\\-(a)")
        return ResponsePresentation(f"{text} {greek}", greek)

    def _safe_set_xlim(self, layer, begin: float, end: float) -> None:
        try:
            layer.set_xlim(begin, end)
        except Exception:
            pass

    def _safe_set_ylim(self, layer, begin: float, end: float) -> None:
        try:
            layer.set_ylim(begin, end)
        except Exception:
            pass

    def _safe_set_int(self, obj, prop: str, value: int) -> None:
        try:
            obj.set_int(prop, value)
        except Exception:
            pass

    def _safe_set_float(self, obj, prop: str, value: float) -> None:
        try:
            obj.set_float(prop, value)
        except Exception:
            pass

    def _set_layer_scale(self, layer, xscale: str, yscale: str) -> None:
        try:
            layer.xscale = xscale
        except Exception:
            pass
        try:
            layer.yscale = yscale
        except Exception:
            pass

    def _safe_rescale(self, layer) -> None:
        try:
            layer.rescale()
        except Exception:
            try:
                layer.lt_exec("layer -r")
            except Exception:
                pass

    @staticmethod
    def _origin_signed(value: float) -> str:
        if value < 0:
            return f"- {abs(value):.6g}"
        return f"+ {value:.6g}"

    @staticmethod
    def _expanded_linear_limits(low: float, high: float, pad: float) -> tuple[float, float]:
        if high == low:
            margin = abs(high) * pad or pad
        else:
            margin = (high - low) * pad
        return low - margin, high + margin

    @staticmethod
    def _expanded_log_limits(low: float, high: float, pad: float) -> tuple[float, float]:
        if low <= 0 or high <= 0:
            return OriginClient._expanded_linear_limits(low, high, pad)
        log_low = math.log10(low)
        log_high = math.log10(high)
        if log_high == log_low:
            margin = pad
        else:
            margin = (log_high - log_low) * pad
        return 10 ** (log_low - margin), 10 ** (log_high + margin)

    @staticmethod
    def _safe_log10(value: float) -> float:
        return 0.0 if value <= 0 else math.log10(value)
