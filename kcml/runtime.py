"""Small runtime setup for stable command-line execution."""

from __future__ import annotations

import os
from pathlib import Path


def configure_runtime(project_root: str | Path | None = None) -> None:
    # Some macOS shells export LC_CTYPE=UTF-8, which fontconfig interprets as an
    # invalid locale/region tag. Replace only that malformed value.
    for variable in ("LANG", "LC_ALL", "LC_CTYPE"):
        if os.environ.get(variable, "").strip() == "UTF-8":
            os.environ[variable] = "en_US.UTF-8"

    # Keep the Matplotlib font cache local and writable. The first invocation
    # may still build the cache once; subsequent invocations reuse it.
    root = Path(project_root) if project_root is not None else Path.cwd()
    cache = root / ".matplotlib_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))


def finish_cli(*, used_torch: bool = False) -> None:
    """Finish a command-line run after every output file has been written.

    This deliberately performs no PyTorch cleanup and no garbage collection.
    On macOS only, a neural run exits with os._exit(0) after flushing the
    terminal streams.  This bypasses the native-library interpreter-shutdown
    hang without disturbing training between lambda values.
    """
    import sys

    print(
        "KCML execution completed successfully; "
        "all requested output files were written."
    )
    sys.stdout.flush()
    sys.stderr.flush()

    force_exit_disabled = (
        os.environ.get("KCML_DISABLE_FORCE_EXIT", "").strip() == "1"
    )
    if used_torch and sys.platform == "darwin" and not force_exit_disabled:
        os._exit(0)
