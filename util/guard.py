from pathlib import Path
import json, os, platform, shutil, subprocess, threading, time
from util.io import save


class ProtectedStop(RuntimeError):
    pass


def linux_memory():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        values[k] = int(v.strip().split()[0]) * 1024
    result = dict(available=values["MemAvailable"], total=values["MemTotal"])
    # Respect the current process cgroup v2 path, not only the mount root.
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            group = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
            base = Path("/sys/fs/cgroup")
            while group.is_relative_to(base):
                try:
                    cap = (group / "memory.max").read_text().strip()
                    used = int((group / "memory.current").read_text())
                    if cap != "max":
                        result["available"] = min(result["available"], int(cap) - used)
                        result["total"] = min(result["total"], int(cap))
                except FileNotFoundError:
                    pass
                if group == base:
                    break
                group = group.parent
    return result


def windows_commit():
    executable = (
        shutil.which("powershell.exe")
        or "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    )
    command = "$m=Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory; @{commit=[double]$m.CommittedBytes;limit=[double]$m.CommitLimit}|ConvertTo-Json -Compress"
    run = subprocess.run(
        [executable, "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        timeout=12,
    )
    if run.returncode:
        raise ProtectedStop("Windows commit telemetry failed")
    h = json.loads(run.stdout.decode("utf-8-sig"))
    if not 0 < h["commit"] < h["limit"]:
        raise ProtectedStop("Windows commit telemetry invalid")
    return h


class ResourceGuard:
    def __init__(self, output, min_available_gib=4.0):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.wsl = "microsoft" in platform.release().lower()
        self.min_available = min_available_gib * 1024**3
        self.latest = None
        self.error = None
        self.closed = threading.Event()
        self.thread = None
        if self.wsl:
            self.latest = (time.monotonic(), windows_commit())
            self.thread = threading.Thread(target=self._monitor, daemon=True)
            self.thread.start()
        self.check()

    def _monitor(self):
        while not self.closed.wait(4):
            try:
                self.latest = (time.monotonic(), windows_commit())
                self.error = None
            except Exception as e:
                self.error = type(e).__name__

    def check(self):
        h = linux_memory()
        if h["available"] < self.min_available:
            raise ProtectedStop(
                "Linux/cgroup available memory below configured reserve"
            )
        if shutil.disk_usage(self.output).free < 10 * 1024**3:
            raise ProtectedStop("Output disk has less than 10 GiB free")
        if self.wsl:
            if self.error or not self.latest or time.monotonic() - self.latest[0] > 20:
                raise ProtectedStop("Windows commit monitor missing or stale")
            host = self.latest[1]
            h.update(host)
            if host["commit"] / host["limit"] >= 0.85:
                raise ProtectedStop(
                    "Windows commit emergency threshold 85%; avoid allocating checkpoint"
                )
        return h

    def boundary(self):
        h = self.check()
        requested = (self.output / "STOP_REQUEST").exists()
        return (
            "user_stop_request"
            if requested
            else (
                "windows_commit_75_percent"
                if self.wsl and h["commit"] / h["limit"] >= 0.75
                else None
            )
        )

    def log(self, event, **fields):
        with (self.output / "resource_events.jsonl").open("a") as f:
            f.write(json.dumps(dict(event=event, time=time.time(), **fields)) + "\n")

    def close(self):
        self.closed.set()
