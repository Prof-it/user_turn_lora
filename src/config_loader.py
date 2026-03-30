"""
Helpers for restoring saved pipeline configs without losing explicit values.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .config import PipelineConfig


def load_saved_pipeline_config(model_dir: Path) -> "PipelineConfig":
    """Load config.json and reapply saved values after PipelineConfig initialization."""
    from .config import PipelineConfig

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    with open(config_path) as f:
        raw = json.load(f)

    filtered = {key: value for key, value in raw.items() if key in PipelineConfig.__dataclass_fields__}
    config = PipelineConfig(**filtered)

    # Restore explicit persisted values after __post_init__ applies hardware defaults.
    for key, value in filtered.items():
        setattr(config, key, value)

    config.output_dir = str(model_dir)
    return config
