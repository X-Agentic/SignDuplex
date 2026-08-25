"""Select the repository's compatible Gradio/model Python stack."""

from __future__ import annotations

import sys
import os
from pathlib import Path


_COMPATIBLE_SITE_PACKAGES = os.environ.get("SIGNDUPLEX_SITE_PACKAGES")
COMPATIBLE_SITE_PACKAGES = (
    Path(_COMPATIBLE_SITE_PACKAGES).expanduser()
    if _COMPATIBLE_SITE_PACKAGES
    else None
)


def configure() -> None:
    """Prefer the known-compatible web/ML stack, with system packages as fallback."""
    if COMPATIBLE_SITE_PACKAGES is None:
        return
    site_path = str(COMPATIBLE_SITE_PACKAGES)
    if COMPATIBLE_SITE_PACKAGES.is_dir() and site_path not in sys.path:
        sys.path.insert(0, site_path)


def restore_system_packages() -> None:
    """Remove the compatibility path before importing the coherent system ML stack."""
    if COMPATIBLE_SITE_PACKAGES is None:
        return
    site_path = str(COMPATIBLE_SITE_PACKAGES)
    while site_path in sys.path:
        sys.path.remove(site_path)
