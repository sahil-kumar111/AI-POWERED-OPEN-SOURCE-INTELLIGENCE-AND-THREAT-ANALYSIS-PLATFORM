"""Portability of `evaluation/measure.py` (native Windows has no `resource` module and no /proc).

All tests run on any OS: Windows behaviour is simulated with monkeypatching (a missing `resource`
module, `sys.platform == "win32"`, a fake `ctypes.WinDLL`). They verify OUR fallbacks and that nothing
is fabricated (unavailable -> None). They do not run the real Win32 API; that needs a Windows host."""

import builtins
import ctypes
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from representation_learning.evaluation import inputs as inp
from representation_learning.evaluation import measure as M
from representation_learning.testing.mocks import mock_encoder

ROOT = Path(__file__).resolve().parents[1]
MIB = 2 ** 20


def fake_resource(maxrss):
    return types.SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda who: types.SimpleNamespace(ru_maxrss=maxrss))


@pytest.fixture
def no_proc(monkeypatch):
    """Every /proc read fails like on Windows."""
    real = builtins.open

    def fake_open(file, *a, **k):
        if str(file).replace("\\", "/").startswith("/proc/"):
            raise FileNotFoundError(file)
        return real(file, *a, **k)
    monkeypatch.setattr(builtins, "open", fake_open)


def proc_with(monkeypatch, files):
    """Serve the given /proc files with the given text; others fail."""
    import io
    real = builtins.open

    def fake_open(file, *a, **k):
        key = str(file)
        if key.startswith("/proc/"):
            if key in files:
                return io.StringIO(files[key])
            raise FileNotFoundError(file)
        return real(file, *a, **k)
    monkeypatch.setattr(builtins, "open", fake_open)


# ------------------------------------------------------------------ import without `resource`
def test_module_imports_without_the_resource_module():
    code = (f"import sys, importlib.abc; sys.path.insert(0, {str(ROOT.parent)!r})\n"
            "class B(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name == 'resource': raise ModuleNotFoundError(\"No module named 'resource'\", name='resource')\n"
            "sys.modules.pop('resource', None); sys.meta_path.insert(0, B())\n"
            "from representation_learning.evaluation import measure as M\n"
            "assert M.resource is None\n"
            "assert M._peak_rss_mb() is None or sys.platform == 'win32'\n"
            "import representation_learning.evaluation.run_eval\n"
            "print('import-ok')")
    r = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "import-ok" in r.stdout, r.stderr


def test_no_unconditional_posix_only_import_remains():
    src = (ROOT / "evaluation" / "measure.py").read_text().splitlines()
    top = [l for l in src if l.startswith("import resource") or l.startswith("from resource")]
    assert top == []                                   # the only import is indented, inside try/except


# ------------------------------------------------------------------ existing Linux behaviour
def test_peak_rss_with_resource_is_unchanged_ru_maxrss_kib_over_1024(monkeypatch):
    monkeypatch.setattr(M, "resource", fake_resource(2048))
    assert M._peak_rss_mb() == 2.0
    monkeypatch.setattr(M, "resource", fake_resource(123456))
    assert M._peak_rss_mb() == 123456 / 1024


def test_peak_rss_with_resource_never_falls_back_to_the_windows_api(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not be consulted when `resource` exists")
    monkeypatch.setattr(M, "_windows_working_set_mb", boom)
    monkeypatch.setattr(M, "resource", fake_resource(1024))
    assert M._peak_rss_mb() == 1.0

    def failing(who):
        raise OSError("nope")
    monkeypatch.setattr(M, "resource", types.SimpleNamespace(RUSAGE_SELF=0, getrusage=failing))
    assert M._peak_rss_mb() is None                    # unchanged: failure -> None


@pytest.mark.skipif(M.resource is None, reason="POSIX `resource` module not available")
def test_real_peak_rss_on_posix_is_a_positive_number():
    v = M._peak_rss_mb()
    assert isinstance(v, float) and v > 0


def test_rss_reads_proc_self_status_when_available(monkeypatch):
    proc_with(monkeypatch, {"/proc/self/status": "Name:\tx\nVmRSS:\t   2048 kB\nThreads:\t1\n"})
    monkeypatch.setattr(M, "_windows_working_set_mb", lambda: (_ for _ in ()).throw(AssertionError("unused")))
    assert M._rss_mb() == 2.0


def test_hardware_context_reads_proc_when_available(monkeypatch):
    proc_with(monkeypatch, {"/proc/cpuinfo": "processor\t: 0\nmodel name\t: Test CPU @ 1GHz\n",
                            "/proc/meminfo": "MemTotal:       8192000 kB\n"})
    ctx = M.hardware_context()
    assert ctx["cpu_model"] == "Test CPU @ 1GHz" and ctx["ram_total_mb"] == 8192000 // 1024


# ------------------------------------------------------------------ no resource, not Windows
def test_without_resource_on_a_non_windows_platform_peak_is_none(monkeypatch):
    monkeypatch.setattr(M, "resource", None)
    monkeypatch.setattr(sys, "platform", "linux")
    assert M._peak_rss_mb() is None


def test_windows_api_is_not_even_touched_off_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **k: (_ for _ in ()).throw(AssertionError("touched")),
                        raising=False)
    assert M._windows_working_set_mb() is None


# ------------------------------------------------------------------ simulated Windows
def install_fake_win32(monkeypatch, *, ok=True, working=100 * MIB, peak=250 * MIB):
    """Fake kernel32/psapi. GetProcessMemoryInfo fills the real struct through byref(), so the field
    names, the `cb` size and the conversion to MiB are exercised."""
    calls = []

    class Fn:
        def __init__(self, impl):
            self.impl, self.argtypes, self.restype = impl, None, None

        def __call__(self, *a):
            return self.impl(*a)

    def get_info(handle, ref, size):
        c = ref._obj
        calls.append((handle, size, c.cb))
        if not ok:
            return 0
        c.WorkingSetSize, c.PeakWorkingSetSize = working, peak
        return 1

    def winDLL(name, use_last_error=False):
        if name == "kernel32":
            return types.SimpleNamespace(GetCurrentProcess=Fn(lambda: 4242))
        if name == "psapi":
            return types.SimpleNamespace(GetProcessMemoryInfo=Fn(get_info))
        raise OSError(name)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "WinDLL", winDLL, raising=False)
    return calls


def test_windows_working_set_is_read_through_psapi_and_converted_to_mib(monkeypatch):
    calls = install_fake_win32(monkeypatch)
    assert M._windows_working_set_mb() == (100.0, 250.0)
    handle, size, cb = calls[0]
    assert handle == 4242 and size == cb == ctypes.sizeof(M._process_memory_counters_type())


def test_process_memory_counters_layout_matches_psapi_h():
    t = M._process_memory_counters_type()
    names = [f[0] for f in t._fields_]
    assert names == ["cb", "PageFaultCount", "PeakWorkingSetSize", "WorkingSetSize",
                     "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                     "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage"]
    from ctypes import wintypes
    # two DWORDs + eight SIZE_T fields. (wintypes.DWORD is 4 bytes on Windows but 8 on LP64 Linux, so the
    # absolute size is only asserted where it is meaningful: 72 bytes on 64-bit Windows.)
    assert ctypes.sizeof(t) == 2 * ctypes.sizeof(wintypes.DWORD) + 8 * ctypes.sizeof(ctypes.c_size_t)
    if sys.platform == "win32" and ctypes.sizeof(ctypes.c_size_t) == 8:
        assert ctypes.sizeof(t) == 72


def test_windows_failures_return_none_not_a_number(monkeypatch):
    install_fake_win32(monkeypatch, ok=False)
    assert M._windows_working_set_mb() is None         # API reports failure
    monkeypatch.undo()
    monkeypatch.setattr(sys, "platform", "win32")      # win32 but no ctypes.WinDLL (e.g. odd build / Linux host)
    monkeypatch.delattr(ctypes, "WinDLL", raising=False)
    assert M._windows_working_set_mb() is None


def test_peak_and_current_rss_on_simulated_windows(monkeypatch, no_proc):
    monkeypatch.setattr(M, "resource", None)
    install_fake_win32(monkeypatch, working=300 * MIB, peak=512 * MIB)
    assert M._peak_rss_mb() == 512.0
    assert M._rss_mb() == 300.0


def test_simulated_windows_with_failing_api_gives_none_everywhere(monkeypatch, no_proc):
    monkeypatch.setattr(M, "resource", None)
    install_fake_win32(monkeypatch, ok=False)
    assert M._peak_rss_mb() is None and M._rss_mb() is None


# ------------------------------------------------------------------ /proc is optional
def test_rss_is_none_without_proc_off_windows(monkeypatch, no_proc):
    monkeypatch.setattr(sys, "platform", "linux")
    assert M._rss_mb() is None


@pytest.mark.parametrize("status", ["", "Name:\tx\n", "VmRSS:\n", "VmRSS:\tabc kB\n"])
def test_malformed_proc_status_gives_none(monkeypatch, status):
    monkeypatch.setattr(sys, "platform", "linux")
    proc_with(monkeypatch, {"/proc/self/status": status})
    assert M._rss_mb() is None


def test_hardware_context_without_proc_keeps_none_and_everything_else(monkeypatch, no_proc):
    ctx = M.hardware_context()
    assert ctx["cpu_model"] is None and ctx["ram_total_mb"] is None
    assert ctx["platform"] and ctx["machine"] is not None and ctx["logical_cpus"] and "versions" in ctx
    json.dumps(ctx)


@pytest.mark.parametrize("meminfo", ["", "\n", "MemTotal:\n", "MemTotal: lots kB\n"])
def test_malformed_meminfo_gives_none_ram(monkeypatch, meminfo):
    proc_with(monkeypatch, {"/proc/meminfo": meminfo})
    assert M.hardware_context()["ram_total_mb"] is None


# ------------------------------------------------------------------ end to end through measure_encoder
def test_measure_encoder_on_simulated_windows_without_proc_or_resource(monkeypatch, no_proc):
    monkeypatch.setattr(M, "resource", None)
    install_fake_win32(monkeypatch, working=300 * MIB, peak=512 * MIB)
    res = M.measure_encoder(mock_encoder("dinov2"), inp.synthetic_inputs(4), repeats=1, batch_sizes=(1,))
    assert res["memory"]["peak_rss_mb_process_so_far"] == 512.0 and res["memory"]["rss_now_mb"] == 300.0
    assert res["load"]["rss_before_load_mb"] == 300.0
    json.dumps(res)


def test_measure_encoder_reports_none_memory_when_nothing_is_measurable(monkeypatch, no_proc):
    monkeypatch.setattr(M, "resource", None)
    monkeypatch.setattr(sys, "platform", "linux")
    res = M.measure_encoder(mock_encoder("clip"), inp.synthetic_inputs(4), repeats=1, batch_sizes=(1,))
    assert res["memory"]["peak_rss_mb_process_so_far"] is None and res["memory"]["rss_now_mb"] is None
    assert res["load"]["rss_before_load_mb"] is None and res["load"]["rss_after_load_mb"] is None
    assert res["extraction"]["n_ok"] == 4              # the measurement itself is unaffected
    json.dumps(res)
