"""Structured logging configuration for mol-active-learn-lite."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    """Format log records as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add run/lineage IDs if available
        if hasattr(record, "run_id"):
            payload["run_id"] = record.run_id
        if hasattr(record, "lineage_id"):
            payload["lineage_id"] = record.lineage_id
        
        # Add extra fields
        if hasattr(record, "extra_fields"):
            payload.update(record.extra_fields)
        
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        
        return json.dumps(payload)


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    """Configure structured logging."""
    root = logging.getLogger()
    root.setLevel(level)
    
    # Remove existing handlers
    for handler in list(root.handlers):
        root.removeHandler(handler)
    
    # Create stderr handler
    handler = logging.StreamHandler(stream=sys.stderr)
    
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
        )
        handler.setFormatter(formatter)
    
    root.addHandler(handler)
    
    # Silence noisy libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("rdkit").setLevel(logging.WARNING)


# Create logger for this module
logger = logging.getLogger(__name__)