#!/usr/bin/env python
"""Run the CADET experiment suite from the repository root."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cadet.experiments import main as cadet_main  # noqa: E402


def main(argv: Sequence[str] | None = None) -> None:
    """Delegate to the package-level experiment CLI.

    Args:
        argv: Optional command-line arguments, excluding the program name.
    """

    cadet_main(None if argv is None else list(argv))


if __name__ == "__main__":
    main()
