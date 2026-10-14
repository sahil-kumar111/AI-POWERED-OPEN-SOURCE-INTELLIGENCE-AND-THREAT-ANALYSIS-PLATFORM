"""Label-free measurements for one encoder. Nothing here measures semantic quality."""

from __future__ import annotations

import os
import platform
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..encoders.base import VisualEncoder, runtime_versions
from ..encoders.datatypes import EncodeFailure
from ..encoders.errors import EncoderError
from ..encoders.preprocessing import preprocess
from .inputs import EvalInput

try:                                  # POSIX only; absent on native Windows
    import resource
except ImportError:                   # pragma: no cover - exercised by tests via monkeypatching
    resource = None                   # type: ignore[assignment]


def hardware_context() -> Dict[str, Any]:
    cpu = None
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    mem_mb = None
    try:
        with open("/proc/meminfo") as f:
            mem_mb = int(f.readline().split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    ctx = {"platform": platform.platform(), "machine": platform.machine(), "cpu_model": cpu,
           "logical_cpus": os.cpu_count(), "ram_total_mb": mem_mb,
           "versions": runtime_versions()}
    try:
        import torch
        ctx["torch_num_threads"] = torch.get_num_threads()
        ctx["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            ctx["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return ctx


def _process_memory_counters_type():
    """ctypes layout of the Win32 PROCESS_MEMORY_COUNTERS structure (psapi.h)."""
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    return PROCESS_MEMORY_COUNTERS


def _windows_working_set_mb() -> Optional[Tuple[float, float]]:
    """(current, peak) working set of this process in MiB via psapi `GetProcessMemoryInfo`
    (standard-library ctypes only). Windows only; any failure returns None, never a guess."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        counters = _process_memory_counters_type()()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(type(counters)),
                                               wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters),
                                          counters.cb):
            return None
        return counters.WorkingSetSize / 2**20, counters.PeakWorkingSetSize / 2**20
    except Exception:
        return None


def _rss_mb() -> Optional[float]:
    """Current resident memory in MiB: /proc on Linux, working set on Windows, else None."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError, IndexError):
        pass
    ws = _windows_working_set_mb()
    return None if ws is None else ws[0]


def _peak_rss_mb() -> Optional[float]:
    """Peak resident memory in MiB: `resource` where available (unchanged Linux behaviour,
    ru_maxrss in KiB), peak working set on Windows, else None."""
    if resource is not None:
        try:
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KiB
        except Exception:
            return None
    ws = _windows_working_set_mb()
    return None if ws is None else ws[1]


def _stats(ms: Sequence[float]) -> Dict[str, Any]:
    if not ms:
        return {"n": 0}
    s = sorted(ms)
    p95 = s[min(len(s) - 1, int(np.ceil(0.95 * len(s))) - 1)]
    return {"n": len(s), "mean_ms": round(statistics.fmean(s), 3),
            "median_ms": round(statistics.median(s), 3), "p95_ms": round(p95, 3),
            "min_ms": round(s[0], 3), "max_ms": round(s[-1], 3)}


def _enc(enc: VisualEncoder, it: EvalInput):
    if it.image is None:
        raise EncoderError(it.extra.get("load_error", "input could not be loaded"))
    return enc.encode(it.image, color_order=it.color_order, evidence=it.evidence)


def measure_encoder(enc: VisualEncoder, inputs: Sequence[EvalInput], *, repeats: int = 3,
                    warmup: int = 1, batch_sizes: Sequence[int] = (1, 4),
                    repeat_check_items: int = 3) -> Dict[str, Any]:
    res: Dict[str, Any] = {"encoder": enc.spec.name, "family": enc.spec.family,
                           "feature": enc.spec.feature_name, "expected_dim": enc.spec.expected_dim,
                           "preprocessing": enc.spec.preprocess.to_dict()}
    rss0 = _rss_mb()
    t0 = time.perf_counter()
    try:
        info = enc.load()
    except EncoderError as e:
        res["load"] = {"ok": False, "error_type": type(e).__name__, "error": str(e)}
        return res
    res["load"] = {"ok": True, "load_time_s": round(time.perf_counter() - t0, 3),
                   "device": info.device, "precision": info.precision,
                   "weights_sha256": info.weights_sha256, "warnings": list(info.warnings),
                   "framework_versions": info.framework_versions,
                   "rss_before_load_mb": None if rss0 is None else round(rss0, 1),
                   "rss_after_load_mb": None if _rss_mb() is None else round(_rss_mb(), 1),
                   "mock": bool(info.extra.get("mock"))}
    res["space_id"] = enc.space_id

    # ---- extraction success / validity (single-item path)
    ok_vecs: List[np.ndarray] = []
    ok_items: List[EvalInput] = []
    failures: List[Dict[str, Any]] = []
    for it in inputs:
        try:
            r = _enc(enc, it)
            ok_vecs.append(r.vector)
            ok_items.append(it)
        except EncoderError as e:
            failures.append({"input": it.name, "error_type": type(e).__name__, "error": str(e)})
    res["extraction"] = {"n_inputs": len(inputs), "n_ok": len(ok_vecs), "n_failed": len(failures),
                         "failures": failures}
    if ok_vecs:
        m = np.stack(ok_vecs).astype(np.float64)
        norms = np.linalg.norm(m, axis=1)
        res["validity"] = {
            "dim": int(m.shape[1]), "dtype": str(ok_vecs[0].dtype),
            "all_finite": bool(np.isfinite(m).all()),
            "norm_min": float(norms.min()), "norm_max": float(norms.max()),
            "max_abs_norm_deviation": float(np.abs(norms - 1).max()),
            "dim_matches_spec": int(m.shape[1]) == enc.spec.expected_dim}

    # ---- latency (single item, includes validation + preprocessing + forward + record)
    lat: List[float] = []
    for _ in range(max(0, warmup)):
        for it in ok_items[:1]:
            _enc(enc, it)
    for _ in range(max(1, repeats)):
        for it in ok_items:
            t = time.perf_counter()
            _enc(enc, it)
            lat.append((time.perf_counter() - t) * 1000)
    res["latency_single_item_end_to_end"] = dict(_stats(lat), repeats=repeats, warmup=warmup,
                                                  batch_size=1)

    # ---- preprocessing vs forward split
    if ok_items:
        from ..encoders.image_io import to_rgb_uint8, crop_bbox
        pre, fwd = [], []
        for it in ok_items[: min(len(ok_items), 8)]:
            rgb = to_rgb_uint8(it.image, it.color_order)
            if it.evidence and it.evidence.get("source_kind") == "crop":
                rgb, _ = crop_bbox(rgb, it.evidence["bbox_xyxy"])
            t = time.perf_counter()
            x = preprocess(rgb, enc.spec.preprocess)[None]
            pre.append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            enc._embed(x)
            fwd.append((time.perf_counter() - t) * 1000)
        res["latency_split_first_items"] = {"preprocess": _stats(pre), "forward_plus_normalise": _stats(fwd)}

    # ---- batch behaviour (per-image time when encoding in chunks of bs)
    bt: Dict[str, Any] = {}
    for bs in batch_sizes:
        if bs < 1 or not ok_items:
            continue
        items = (ok_items * ((bs // len(ok_items)) + 1))[:bs]
        per_run = []
        for _ in range(max(1, repeats)):
            t = time.perf_counter()
            _batch(enc, items, bs)
            per_run.append((time.perf_counter() - t) * 1000 / bs)
        bt[str(bs)] = dict(_stats(per_run), note="end-to-end ms per image")
    res["latency_per_image_by_batch_size"] = bt

    # ---- repeatability and batch-vs-single consistency
    rep: Dict[str, Any] = {"n_items": 0}
    k = min(repeat_check_items, len(ok_items))
    if k:
        maxdiff, mincos = 0.0, 1.0
        for it in ok_items[:k]:
            a = _enc(enc, it).vector.astype(np.float64)
            for _ in range(3):
                b = _enc(enc, it).vector.astype(np.float64)
                maxdiff = max(maxdiff, float(np.abs(a - b).max()))
                mincos = min(mincos, float(a @ b))
        rep = {"n_items": k, "n_repeats_each": 3, "max_abs_diff": maxdiff, "min_cosine": mincos,
               "bitwise_identical": maxdiff == 0.0}
        sub = ok_items[: min(len(ok_items), 6)]
        singles = np.stack([_enc(enc, i).vector for i in sub]).astype(np.float64)
        bres = _batch(enc, sub)
        if all(not isinstance(x, EncodeFailure) for x in bres):
            bm = np.stack([x.vector for x in bres]).astype(np.float64)
            rep["batch_vs_single_max_abs_diff"] = float(np.abs(singles - bm).max())
    res["repeatability"] = rep

    # ---- descriptive spread (NOT a quality metric)
    if len(ok_vecs) >= 3:
        m = np.stack(ok_vecs).astype(np.float64)
        g = m @ m.T
        off = g[~np.eye(len(m), dtype=bool)]
        res["descriptive_spread_NOT_A_QUALITY_METRIC"] = {
            "n": len(m), "mean_pairwise_cosine": round(float(off.mean()), 4),
            "min_pairwise_cosine": round(float(off.min()), 4),
            "max_pairwise_cosine": round(float(off.max()), 4),
            "caveat": "Within-encoder only. Depends on input mix; says nothing about "
                      "semantic quality and must not be compared across encoders."}

    res["memory"] = {"peak_rss_mb_process_so_far": None if _peak_rss_mb() is None
                     else round(_peak_rss_mb(), 1),
                     "rss_now_mb": None if _rss_mb() is None else round(_rss_mb(), 1),
                     "note": "Process-wide peak (includes anything run earlier in the process); "
                             "run one encoder per process for clean numbers."}
    be = getattr(enc, "_backend", None)
    if be is not None and hasattr(be, "peak_cuda_mb"):
        res["memory"]["peak_cuda_mb"] = be.peak_cuda_mb()
    return res


def _batch(enc: VisualEncoder, items: Sequence[EvalInput], batch_size: Optional[int] = None):
    """Batch path for mixed inputs: group by color_order so encode_batch's single
    color_order argument stays correct; evidences/crops are passed through."""
    out: List[Any] = [None] * len(items)
    groups: Dict[Optional[str], List[int]] = {}
    for i, it in enumerate(items):
        groups.setdefault(it.color_order, []).append(i)
    for co, idxs in groups.items():
        res = enc.encode_batch([items[i].image for i in idxs], color_order=co,
                               evidences=[items[i].evidence for i in idxs],
                               batch_size=batch_size)
        for i, r in zip(idxs, res):
            out[i] = r
    return out


def measure_diagnostic_frames(enc: VisualEncoder, frames: Sequence[EvalInput]) -> Dict[str, Any]:
    """Whole-frame PIPELINE DIAGNOSTICS only (never target-object evaluation inputs).
    Reports extraction success, validity and latency; excluded from every other statistic."""
    ok, fails, lat = [], [], []
    for it in frames:
        t = time.perf_counter()
        try:
            r = enc.encode(it.image, color_order=it.color_order, evidence=it.evidence,
                           diagnostic_only=True)
            lat.append((time.perf_counter() - t) * 1000)
            ok.append(r.vector)
        except EncoderError as e:
            fails.append({"input": it.name, "error_type": type(e).__name__, "error": str(e)})
    out: Dict[str, Any] = {
        "purpose": "pipeline_diagnostic_only; NOT target-object representation, not used in any "
                   "other statistic or in any quality statement",
        "n_inputs": len(frames), "n_ok": len(ok), "n_failed": len(fails), "failures": fails,
        "latency_end_to_end": _stats(lat)}
    if ok:
        m = np.stack(ok).astype(np.float64)
        out["all_finite"] = bool(np.isfinite(m).all())
        out["max_abs_norm_deviation"] = float(np.abs(np.linalg.norm(m, axis=1) - 1).max())
    return out
