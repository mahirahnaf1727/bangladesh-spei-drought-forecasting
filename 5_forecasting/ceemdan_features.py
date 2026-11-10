"""
================================================================================
ceemdan_features.py  --  leakage-free CEEMDAN front-end for CEEMDAN-BiLSTM
================================================================================
PORTED UNCHANGED from Models/CEEMDAN_LSTM/ceemdan_lstm_forecast.py
(decompose_window, decompose_block). Two practical additions only:

  1. The cache file name includes a fingerprint of the input windows, so a
     stale cache can never be reused for different data (e.g. strict pipeline).
  2. Optional parallel decomposition (n_jobs). Each window is decomposed on its
     own with the same fixed noise seed, so the result is identical whether it
     runs on 1 core or many. This module does NOT import TensorFlow, so the
     worker processes start quickly (important on Windows).

Leakage-free by construction: CEEMDAN is applied to the SPEI-history channel of
each input window on its own, using only timesteps that were already the
model's input. No whole-series decomposition.
================================================================================
"""

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

try:
    from PyEMD import CEEMDAN
except Exception as e:                                    # fail loud and clear
    raise ImportError(
        "PyEMD not found. Install it in the same Python you train with:\n"
        "    python -m pip install EMD-signal"
    ) from e


def decompose_window(spei_seq, n_imfs=4, trials=50, seed=42):
    """Leakage-free CEEMDAN on ONE window's SPEI history (unchanged).
    Input  (T,)  ->  output (T, n_imfs), zero-padded / surplus folded into the
    last channel so the channels always sum back to the input."""
    T = spei_seq.shape[0]
    out = np.zeros((n_imfs, T), dtype=float)
    if T < 2 or np.allclose(spei_seq, spei_seq[0]):
        return out.T
    c = CEEMDAN(trials=trials, parallel=False)
    c.noise_seed(seed)
    try:
        imfs = c.ceemdan(spei_seq.astype(float))           # (k, T)
    except Exception:
        return out.T
    k = imfs.shape[0]
    take = min(k, n_imfs)
    out[:take] = imfs[:take]
    if k > n_imfs:
        out[n_imfs - 1] += imfs[n_imfs:].sum(axis=0)
    return out.T


def _decompose_chunk(args):
    block, n_imfs, trials, seed = args
    return np.stack([decompose_window(w, n_imfs, trials, seed) for w in block])


def fingerprint(block):
    """Short hash of the exact window values (float32), for cache naming."""
    return hashlib.md5(np.ascontiguousarray(block, dtype=np.float32).tobytes()).hexdigest()[:12]


def decompose_block(spei_block, tag, cache_dir, n_imfs=4, trials=50, seed=42,
                    n_jobs=None, log=print):
    """Decompose every window of an (n, T) SPEI block -> (n, T, n_imfs).
    Cached to disk; the cache name carries a fingerprint of the input."""
    cache_dir = os.fspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    fp = fingerprint(spei_block)
    cache_f = os.path.join(cache_dir, f"imfs_{tag}_{fp}.npy")
    if os.path.exists(cache_f):
        arr = np.load(cache_f)
        if arr.shape[:2] == spei_block.shape:
            log(f"    [ceemdan cache] loaded {arr.shape} <- {os.path.basename(cache_f)}")
            return arr

    n, T = spei_block.shape
    if n_jobs is None:
        n_jobs = int(os.environ.get("P1_CEEMDAN_JOBS", "0") or 0) or \
            max(1, (os.cpu_count() or 2) - 1)
    log(f"    [ceemdan] decomposing {n} windows (T={T}) on {n_jobs} process(es) "
        f"-- one-time step, cached afterwards")

    import time
    t0 = time.time()
    chunks = [c for c in np.array_split(spei_block, min(n, max(n_jobs * 8, 20))) if len(c)]
    parts, done = [], 0
    def _progress(k):
        el = time.time() - t0
        left = el / done * (n - done) if done else 0
        log(f"      {done}/{n} windows ({100 * done / n:.0f}%)  "
            f"elapsed {el / 60:.1f} min, ~{left / 60:.1f} min left")
    if n_jobs == 1:
        for k, c in enumerate(chunks, 1):
            parts.append(_decompose_chunk((c, n_imfs, trials, seed)))
            done += len(c)
            if k % 2 == 0 or k == len(chunks):
                _progress(k)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for k, part in enumerate(ex.map(_decompose_chunk,
                                            [(c, n_imfs, trials, seed) for c in chunks]), 1):
                parts.append(part)
                done += len(part)
                if k % max(1, len(chunks) // 10) == 0 or k == len(chunks):
                    _progress(k)
    out = np.concatenate(parts, axis=0)

    out = out.astype(np.float32)
    np.save(cache_f, out)
    log(f"    [ceemdan cache] saved -> {os.path.basename(cache_f)}")
    return out
