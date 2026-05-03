"""Structured logging and optional tracing utilities."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional


class JsonFormatter(logging.Formatter):
    """Simple JSON log formatter for machine-parseable logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        if hasattr(record, 'event'):
            payload['event'] = getattr(record, 'event')
        if hasattr(record, 'run_id'):
            payload['run_id'] = getattr(record, 'run_id')
        return json.dumps(payload, ensure_ascii=True)


def configure_logging(verbose: bool = False, json_logs: bool = False) -> None:
    """Configure root logger with plain or JSON formatting."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler()
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(handler)


def configure_tracing(service_name: str = 'qrc_thresher') -> Optional[Any]:
    """Configure OpenTelemetry tracing if dependency is available.

    Returns the tracer provider or None when OpenTelemetry is unavailable.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider

        # stdout exporter keeps tracing optional and self-contained.
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
    except Exception:
        return None

    resource = Resource.create({'service.name': service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    return provider
