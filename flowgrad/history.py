import json
import os
from datetime import datetime
from typing import Dict, Any, List

HISTORY_DIR = ".flowgrad"
HISTORY_FILE = "history.jsonl"


class HistoryTracker:
    """Tracks cross-experiment runs for AI context in a lightweight JSONL file."""

    @staticmethod
    def _get_path() -> str:
        """Ensure directory exists and return the file path."""
        if not os.path.exists(HISTORY_DIR):
            try:
                os.makedirs(HISTORY_DIR)
            except OSError:
                pass  # Might fail in some restricted environments
        return os.path.join(HISTORY_DIR, HISTORY_FILE)

    @classmethod
    def append_run(cls, run_data: Dict[str, Any]) -> None:
        """
        Append a single run's summary to the history file.
        Includes timestamp automatically.
        """
        path = cls._get_path()
        run_data["timestamp"] = datetime.now().isoformat()
        
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(run_data) + "\n")
        except Exception as e:
            # Silently fail if we can't write to history (e.g. read-only FS)
            pass

    @classmethod
    def get_recent_runs(cls, n: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the last N runs from the history file.
        """
        path = cls._get_path()
        if not os.path.exists(path):
            return []
            
        runs = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        runs.append(json.loads(line))
        except Exception:
            return []
            
        return runs[-n:]
