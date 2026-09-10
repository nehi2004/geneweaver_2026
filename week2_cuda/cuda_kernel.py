"""
cuda_kernel.py
---------------
WHY THIS FILE EXISTS:
This is the heart of the whole project — the GPU version of Week 1's naive
loop. Instead of checking one genome position at a time, we launch one GPU
thread PER POSITION, and thousands of them run the comparison simultaneously.

KEY CONCEPTS (read before editing this file):
- @cuda.jit: tells Numba to JIT-compile this Python function into real CUDA
  machine code. Inside a @cuda.jit function you can't use normal Python
  features (no lists, no string methods) — only numbers, numpy-style arrays,
  and a restricted subset of Python. That's the tradeoff for GPU speed.
- cuda.grid(1): returns this thread's unique global index. Every one of the
  thousands of threads runs this exact same function, but each gets a
  different index, so each checks a different genome position.
- Host vs Device: `cuda.to_device(array)` copies a numpy array from CPU RAM
  (Host) into GPU VRAM (Device). The kernel can only read/write Device
  memory. `.copy_to_host()` copies results back.
- Chunking: we never send the whole genome to the GPU at once — we send it
  in pieces (chunk_size), so a 3-billion-base genome doesn't need 3GB+ of
  contiguous VRAM in one shot. This is what avoids CUDA_OUT_OF_MEMORY.

If no GPU is available, pass --simulate to run the exact same logic on CPU
via a plain Python loop (slow, but lets you test correctness without hardware).
"""
import argparse
import time
import sys
import os

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "week1_cpu_baseline"))
from genomic_parser import load_genome_sequence, clean_sequence, encode_sequence, chunk_genome, BASE_TO_INT

try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False


def _cuda_kernel_factory():
    """Defines the actual GPU kernel. Wrapped in a function so this file can
    still be imported even in environments where `numba.cuda` isn't usable."""
    from numba import cuda

    @cuda.jit
    def hamming_scan_kernel(genome_dev, guide_dev, max_mismatches, mismatch_counts_dev):
        i = cuda.grid(1)
        guide_len = guide_dev.shape[0]
        n = genome_dev.shape[0]

        if i > n - guide_len:
            return  # this thread's window would run off the end of the chunk

        mismatches = 0
        for j in range(guide_len):
            if genome_dev[i + j] != guide_dev[j]:
                mismatches += 1
        mismatch_counts_dev[i] = mismatches

    return hamming_scan_kernel


def gpu_scan_chunk(encoded_chunk: np.ndarray, encoded_guide: np.ndarray, max_mismatches: int):
    """Runs one chunk through the GPU kernel and returns mismatch counts per position."""
    kernel = _cuda_kernel_factory()

    n = len(encoded_chunk)
    genome_dev = cuda.to_device(encoded_chunk)
    guide_dev = cuda.to_device(encoded_guide)
    # 255 = "not enough bases left to check" sentinel value
    mismatch_counts = np.full(n, 255, dtype=np.uint8)
    mismatch_counts_dev = cuda.to_device(mismatch_counts)

    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

    kernel[blocks_per_grid, threads_per_block](genome_dev, guide_dev, max_mismatches, mismatch_counts_dev)
    cuda.synchronize()

    return mismatch_counts_dev.copy_to_host()


def cpu_simulated_scan_chunk(encoded_chunk: np.ndarray, encoded_guide: np.ndarray, max_mismatches: int):
    """Same math as the GPU kernel, run serially on CPU. Used only with
    --simulate when no CUDA device is present, so you can verify logic."""
    n = len(encoded_chunk)
    guide_len = len(encoded_guide)
    mismatch_counts = np.full(n, 255, dtype=np.uint8)
    for i in range(n - guide_len + 1):
        mismatches = 0
        for j in range(guide_len):
            if encoded_chunk[i + j] != encoded_guide[j]:
                mismatches += 1
        mismatch_counts[i] = mismatches
    return mismatch_counts


def scan_genome(genome_path: str, guide: str, max_mismatches: int, chunk_size: int, simulate: bool):
    raw = clean_sequence(load_genome_sequence(genome_path))
    encoded_genome = encode_sequence(raw)
    encoded_guide = encode_sequence(guide.upper())

    overlap = len(guide) - 1
    chunks = chunk_genome(encoded_genome, chunk_size, overlap)

    all_hits = []
    scan_fn = cpu_simulated_scan_chunk if simulate else gpu_scan_chunk

    for chunk_start, chunk_data in chunks:
        counts = scan_fn(chunk_data, encoded_guide, max_mismatches)
        hit_positions = np.where(counts <= max_mismatches)[0]
        for local_pos in hit_positions:
            global_pos = chunk_start + int(local_pos)
            all_hits.append((global_pos, int(counts[local_pos])))

    # Dedup positions found in overlapping regions of adjacent chunks
    seen = {}
    for pos, mm in all_hits:
        if pos not in seen or mm < seen[pos]:
            seen[pos] = mm
    return sorted(seen.items())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU-accelerated off-target scan via Numba CUDA.")
    parser.add_argument("--genome", required=True)
    parser.add_argument("--guide", required=True)
    parser.add_argument("--max-mismatches", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=1_000_000,
                         help="Bases per GPU chunk. Lower this if you hit CUDA_OUT_OF_MEMORY.")
    parser.add_argument("--simulate", action="store_true",
                         help="Run on CPU instead of GPU (for testing without hardware).")
    args = parser.parse_args()

    if not args.simulate and not CUDA_AVAILABLE:
        print("No CUDA-capable GPU detected. Re-run with --simulate to test the logic on CPU,")
        print("or fix your CUDA/driver install (see README troubleshooting table).")
        sys.exit(1)

    print(f"Mode: {'CPU SIMULATION' if args.simulate else 'GPU (CUDA)'}")
    start = time.perf_counter()
    hits = scan_genome(args.genome, args.guide, args.max_mismatches, args.chunk_size, args.simulate)
    elapsed = time.perf_counter() - start

    print(f"\n--- GPU KERNEL RESULT ---")
    print(f"Found {len(hits)} candidate off-target sites (<= {args.max_mismatches} mismatches)")
    print(f"Time taken: {elapsed:.4f} seconds")
    print("Compare this against week1_cpu_baseline's printed time.")

    for pos, mm in hits[:5]:
        print(f"  position {pos:>10,}  mismatches={mm}")
