"""Write a file whole, or not at all.

LOMA owns several files that a user would be upset to lose: ``MEMORY.md`` is the
only copy of everything it has learned, ``mcp.json`` is the whole tool
configuration, and the preferences decide which model runs. All of them used to
be written with ``path.write_text()``, which truncates *first* — so a crash, a
full disk, or a Ctrl-C between the truncate and the write leaves a zero-length
file. The user does not get a corrupted value they can spot; they get an empty
memory and no error.

The fix is the one every database uses: write to a sibling temporary file on the
same volume, flush it to disk, then rename over the target. Rename within a
filesystem is atomic, so a concurrent reader sees either the old file or the new
one and never a half-written one.

Permissions are preserved from the file being replaced, and default to ``0600``:
these files hold personal facts, and a fresh one should not be world-readable.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

DEFAULT_MODE = 0o600


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Replace ``path`` with ``text`` so a crash cannot leave a partial file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    mode = _existing_mode(target)
    handle, temporary = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding=encoding) as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, target)
    except BaseException:
        # Never leave the scratch file behind for the next run to find.
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise
    return target


def _existing_mode(target: Path) -> int:
    """Keep whatever mode the file already had; new files are private."""
    try:
        return os.stat(target).st_mode & 0o777
    except OSError:
        return DEFAULT_MODE
