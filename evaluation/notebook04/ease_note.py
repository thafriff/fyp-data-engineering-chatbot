"""
Adds an entry to evaluation/notebook04/ease_log.json, the evidence for Table 3.9's Ease of Implementation criterion.

The patch is a small JSON file, so each note is written once and can be kept with the run it describes:
- a key starting with "+" appends its value to that list in the model's entry ("+trial_reruns": {...});
- any other key sets the value directly;
- the string "now", anywhere in the value, becomes the current UTC time.
Scores are never assigned here: the author scores Ease of Implementation at the end of integration.

Usage (from the project root):
    venv\\Scripts\\python evaluation\\notebook04\\ease_note.py <model> <patch.json>
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402


def stamp(value, now: str):
    """Replace every "now" in a JSON value with the given UTC time."""
    if value == "now":
        return now
    if isinstance(value, dict):
        return {k: stamp(v, now) for k, v in value.items()}
    if isinstance(value, list):
        return [stamp(v, now) for v in value]
    return value


def apply_note(log: dict, model: str, patch: dict, now: str) -> list[str]:
    """Apply one patch to one model's entry and return the keys it touched. The model must already be in the log."""
    if model not in log["models"]:
        raise KeyError(f"{model} is not in ease_log.json; known: {sorted(log['models'])}")
    entry, touched = log["models"][model], []
    for key, value in patch.items():
        value = stamp(value, now)
        if key == "score" or key.lstrip("+") == "score":
            raise ValueError("Ease scores are assigned by the author at the end of integration, not by this helper.")
        if key.startswith("+"):
            entry.setdefault(key[1:], []).append(value)
        else:
            entry[key] = value
        touched.append(key)
    return touched


def main() -> None:
    """Read the patch, apply it, and write the log back."""
    if len(sys.argv) != 3:
        sys.exit("Usage: ease_note.py <model> <patch.json>")
    model, patch_path = sys.argv[1], Path(sys.argv[2])
    log = json.loads(settings.NB04_EASE_LOG_FILE.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    touched = apply_note(log, model, json.loads(patch_path.read_text(encoding="utf-8")), now)
    settings.NB04_EASE_LOG_FILE.write_text(json.dumps(log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{settings.NB04_EASE_LOG_FILE.name}: {model} updated at {now} ({', '.join(touched)})")


if __name__ == "__main__":
    main()
