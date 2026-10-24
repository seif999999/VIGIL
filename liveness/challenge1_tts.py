"""Offline spoken Challenge 1 instructions (no cloud, no ASR).

Preferred: Windows SAPI via PowerShell System.Speech (no extra pip deps).
Optional: pyttsx3 if installed. Always fail soft — HUD text remains authoritative.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from typing import Callable


def _speak_powershell(text: str) -> bool:
    ps = shutil.which("powershell") or shutil.which("pwsh")
    if not ps:
        return False
    # Escape single quotes for PowerShell single-quoted string
    safe = text.replace("'", "''")
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.Speak('{safe}')"
    )
    try:
        subprocess.run(
            [ps, "-NoProfile", "-NonInteractive", "-Command", script],
            check=False,
            capture_output=True,
            timeout=15,
        )
        return True
    except (OSError, subprocess.TimeoutExpired):
        return False


def _speak_pyttsx3(text: str) -> bool:
    try:
        import pyttsx3  # type: ignore
    except ImportError:
        return False
    try:
        eng = pyttsx3.init()
        eng.say(text)
        eng.runAndWait()
        return True
    except Exception:  # noqa: BLE001
        return False


def speak_instruction(text: str, *, blocking: bool = False) -> None:
    """Speak text offline. Non-blocking by default so the UI stays responsive."""

    def _run() -> None:
        if _speak_powershell(text):
            return
        _speak_pyttsx3(text)

    if blocking:
        _run()
    else:
        threading.Thread(target=_run, name="challenge1-tts", daemon=True).start()


def make_async_speaker() -> Callable[[str], None]:
    return lambda text: speak_instruction(text, blocking=False)
