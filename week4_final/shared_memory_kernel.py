"""
shared_memory_kernel.py
-------------------------
WHY THIS FILE EXISTS:
Week 2's kernel works, but every thread reads its own window directly from
GLOBAL memory (VRAM) — the slowest memory on the GPU. Nearby threads reread
almost the same overlapping bytes over and over. CUDA "shared memory" is a
small, extremely fast memory pool shared by all threads in the same block.
This kernel has each block cooperatively load its needed genome tile into
shared memory ONCE, then every thread in that block reads from shared memory
instead of hammering global memory repeatedly. This is the standard
"tiling" optimization pattern used in real high-performance CUDA kernels.

If you don't have a GPU yet, this file still imports safely — the kernel
definition just isn't built until you actually call scan_genome_optimized().
"""
import argparse
import sys
import os
import time

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "week1_cpu_baseline"))
from genomic_parser import load_genome_sequence, clean_sequence, encode_sequence, chunk_genome

try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False

THREADS_PER_BLOCK = 256
MAX_GUIDE_LEN = 64  # shared memory arrays need a compile-time-ish fixed size
# NOTE: cuda.shared.array's shape must be a compile-time LITERAL constant.
# An arithmetic expression like (THREADS_PER_BLOCK + MAX_GUIDE_LEN) computed
# inside the kernel is NOT accepted by Numba's typing pass (it sees a plain
# int64, not a literal) and raises TypingError. Pre-computing it as its own
# named literal here, and referencing that name directly in the kernel, is
# what makes it fold correctly.
TILE_SIZE = 320  # must equal THREADS_PER_BLOCK + MAX_GUIDE_LEN


def _build_shared_memory_kernel():
    from numba import cuda

    @cuda.jit
    def hamming_scan_shared_kernel(genome_dev, guide_dev, max_mismatches, mismatch_counts_dev):
        guide_len = guide_dev.shape[0]
        n = genome_dev.shape[0]

        # Shared memory tile: holds this block's slice of the genome PLUS the
        # extra (guide_len - 1) bases needed so windows near the tile's right
        # edge don't run off the end. Declared with a fixed max size.
        tile = cuda.shared.array(shape=(TILE_SIZE,), dtype=np.uint8)
        shared_guide = cuda.shared.array(shape=(MAX_GUIDE_LEN,), dtype=np.uint8)

        tx = cuda.threadIdx.x
        block_start = cuda.blockIdx.x * cuda.blockDim.x
        global_i = block_start + tx

        # Cooperative load: each thread loads one (or two) bytes into the
        # shared tile instead of every thread independently hitting global
        # memory for the whole window it needs.
        if block_start + tx < n:
            tile[tx] = genome_dev[block_start + tx]
        if tx < guide_len - 1 and block_start + THREADS_PER_BLOCK + tx < n:
            tile[THREADS_PER_BLOCK + tx] = genome_dev[block_start + THREADS_PER_BLOCK + tx]
        if tx < guide_len:
            shared_guide[tx] = guide_dev[tx]

        cuda.syncthreads()  # wait until every thread in the block has finished loading

        if global_i > n - guide_len:
            return

        mismatches = 0
        for j in range(guide_len):
            if tile[tx + j] != shared_guide[j]:
                mismatches += 1
        mismatch_counts_dev[global_i] = mismatches

    return hamming_scan_shared_kernel


def gpu_scan_chunk_optimized(encoded_chunk: np.ndarray, encoded_guide: np.ndarray, max_mismatches: int):
    if len(encoded_guide) > MAX_GUIDE_LEN:
        raise ValueError(f"Guide length {len(encoded_guide)} exceeds MAX_GUIDE_LEN={MAX_GUIDE_LEN}. "
                          f"Raise the constant if you need longer guides.")

    kernel = _build_shared_memory_kernel()
    n = len(encoded_chunk)

    genome_dev = cuda.to_device(encoded_chunk)
    guide_dev = cuda.to_device(encoded_guide)
    mismatch_counts = np.full(n, 255, dtype=np.uint8)
    mismatch_counts_dev = cuda.to_device(mismatch_counts)

    blocks_per_grid = (n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    kernel[blocks_per_grid, THREADS_PER_BLOCK](genome_dev, guide_dev, max_mismatches, mismatch_counts_dev)
    cuda.synchronize()

    return mismatch_counts_dev.copy_to_host()


def scan_genome_optimized(genome_path: str, guide: str, max_mismatches: int, chunk_size: int):
    raw = clean_sequence(load_genome_sequence(genome_path))
    encoded_genome = encode_sequence(raw)
    encoded_guide = encode_sequence(guide.upper())
    overlap = len(guide) - 1
    chunks = chunk_genome(encoded_genome, chunk_size, overlap)

    all_hits = []
    for chunk_start, chunk_data in chunks:
        counts = gpu_scan_chunk_optimized(chunk_data, encoded_guide, max_mismatches)
        hit_positions = np.where(counts <= max_mismatches)[0]
        for p in hit_positions:
            all_hits.append((chunk_start + int(p), int(counts[p])))

    seen = {}
    for pos, mm in all_hits:
        if pos not in seen or mm < seen[pos]:
            seen[pos] = mm
    return sorted(seen.items())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shared-memory-optimized CUDA off-target scan.")
    parser.add_argument("--genome", required=True)
    parser.add_argument("--guide", required=True)
    parser.add_argument("--max-mismatches", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    args = parser.parse_args()

    if not CUDA_AVAILABLE:
        print("No CUDA-capable GPU detected. This optimized kernel requires real GPU hardware")
        print("since shared memory tiling has no meaningful CPU equivalent to simulate.")
        sys.exit(1)

    start = time.perf_counter()
    hits = scan_genome_optimized(args.genome, args.guide, args.max_mismatches, args.chunk_size)
    elapsed = time.perf_counter() - start

    print(f"\n--- SHARED MEMORY KERNEL RESULT ---")
    print(f"Found {len(hits)} candidate off-target sites")
    print(f"Time taken: {elapsed:.4f} seconds")
    print("Compare against week2_cuda/cuda_kernel.py's global-memory-only version.")