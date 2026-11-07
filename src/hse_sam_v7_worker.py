"""Run HSE AdamW/SAM V7 under an independent hidden Windows process tree."""
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
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs / f"hse_sam_v7_worker_{stamp}_{os.getpid()}.log"
    stream = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = stream
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    kernel.IsProcessInJob.restype = wintypes.BOOL
    kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                                wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    kernel.QueryInformationJobObject.restype = wintypes.BOOL
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
    kernel.CloseHandle.restype = wintypes.BOOL
    parent_pid = os.getppid()
    parent_handle = kernel.OpenProcess(0x1000, False, parent_pid)
    if not parent_handle:
        raise ctypes.WinError(ctypes.get_last_error())
    parent_result = wintypes.BOOL()
    try:
        if not kernel.IsProcessInJob(parent_handle, None, ctypes.byref(parent_result)):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(parent_handle)
    receipt = {
        "pid": os.getpid(), "parent_pid": parent_pid,
        "started_at": datetime.now().astimezone().isoformat(),
        "in_windows_job": bool(result.value), "job_limit_flags": job_limits,
        "venv_launcher_in_job": bool(parent_result.value),
        "kill_on_job_close": bool((job_limits or 0) & 0x2000), "log": str(log_path),
    }
    receipt_path = logs / "hse_sam_v7_runtime.json"
    temporary = receipt_path.with_name(receipt_path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(receipt_path)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    # The venv launcher may own its own job, but must itself be outside the
    # invoking app's job so closing that app cannot kill the experiment.
    if receipt["venv_launcher_in_job"]:
        raise RuntimeError("The venv launcher remains in a Windows job; inspect its lifetime")
    sys.path.insert(0, str(ROOT / "src"))
    sys.argv = [str(ROOT / "src" / "run_hse_sam_v7.py")]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
