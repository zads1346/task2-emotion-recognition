"""Run the task LoRA experiment under an independent hidden Windows process tree."""
import ctypes
from ctypes import wintypes
from datetime import datetime
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    (ROOT / "logs").mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = ROOT / "logs" / f"flip_consistency_v6_worker_{stamp}.log"
    stream = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = stream
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    result = wintypes.BOOL()
    if not kernel.IsProcessInJob(kernel.GetCurrentProcess(), None, ctypes.byref(result)):
        raise ctypes.WinError(ctypes.get_last_error())
    job_limits = None
    if result.value:
        buffer = ctypes.create_string_buffer(144)
        size = wintypes.DWORD()
        if not kernel.QueryInformationJobObject(None, 9, buffer, len(buffer), ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        job_limits = int.from_bytes(buffer.raw[16:20], "little")
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    parent_handle = kernel.OpenProcess(0x1000, False, os.getppid())
    if not parent_handle:
        raise ctypes.WinError(ctypes.get_last_error())
    parent_result = wintypes.BOOL()
    try:
        if not kernel.IsProcessInJob(parent_handle, None, ctypes.byref(parent_result)):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(parent_handle)
    receipt = {
        "pid": os.getpid(), "parent_pid": os.getppid(),
        "started_at": datetime.now().astimezone().isoformat(),
        "in_windows_job": bool(result.value), "job_limit_flags": job_limits,
        "venv_launcher_in_job": bool(parent_result.value),
        "kill_on_job_close": bool((job_limits or 0) & 0x2000), "log": str(log_path),
    }
    (ROOT / "logs" / "flip_consistency_v6_runtime.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    # Python's venv launcher normally creates its own job. It must itself live
    # outside the invoking app's job so closing that app does not kill training.
    if receipt["venv_launcher_in_job"]:
        raise RuntimeError("The venv launcher remains in a Windows job; inspect its lifetime")
    sys.path.insert(0, str(ROOT / "src"))
    sys.argv = [str(ROOT / "src" / "run_flip_consistency_v6.py")]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
