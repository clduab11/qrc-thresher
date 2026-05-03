"""Plugin inventory CLI command implementation."""

from __future__ import annotations

import json

from qrc_thresher.plugins.registry import plugin_inventory


def plugins_handler() -> int:
    """Print registered plugin inventory as JSON."""
    print(json.dumps(plugin_inventory(), indent=2, sort_keys=True))
    return 0
