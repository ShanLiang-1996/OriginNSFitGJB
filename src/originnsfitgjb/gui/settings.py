from __future__ import annotations

from dataclasses import asdict, dataclass, fields
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
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8-sig"))
        return _settings_from_payload(payload)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return GuiSettings()


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
    if not isinstance(payload, dict):
        return GuiSettings()
    defaults = asdict(GuiSettings())
    setting_names = {field.name for field in fields(GuiSettings)}
    known_payload = {key: value for key, value in payload.items() if key in setting_names}
    merged = {**defaults, **known_payload}
    merged["recent_patterns"] = tuple(merged.get("recent_patterns") or DEFAULT_PATTERNS)
    return GuiSettings(**merged)
