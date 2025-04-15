"""Some silly idea to make the PS filewatcher invokable from python"""
raise NotImplementedError("chmengine.stream.scripts has no Python support at this time.")
# pylint: disable=unreachable
from subprocess import Popen
from pathlib import Path


def run_stream_alerts():
    """Launches the PowerShell training stream watcher."""
    script_path = Path(__file__).parent / "stream_alerts.ps1"

    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    # Launch in new terminal window (for interactive visual feedback)
    with Popen([
        "powershell.exe",
        "-NoExit",  # Keeps the window open
        "-ExecutionPolicy", "Bypass",  # Avoid PS policy issues
        "-File", str(script_path)
    ]) as open_proc:
        open_proc.wait()
