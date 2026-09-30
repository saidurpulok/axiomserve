import json
from pathlib import Path
from typing import Any


def save_report(report: dict[str, Any], output: str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
