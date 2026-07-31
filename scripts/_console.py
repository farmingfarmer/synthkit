"""Make stdout survive a Windows console.

The `synthkit` command already does this, but standalone scripts
bypassed that entry point entirely — and four of them died with
UnicodeEncodeError on a cp437 console, which is exactly the class
of failure the earlier Windows audit was supposed to have closed.
Fixing it once here rather than policing punctuation forever:
characters the console cannot render are replaced instead of
raising.
"""
from __future__ import annotations

import sys


def console_safe() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
