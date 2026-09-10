"""
dask_scheduler.py
-------------------
WHY THIS FILE EXISTS:
Week 2's kernel handles one GPU. Real genomic pipelines often have access to
multiple GPUs (or multiple machines in a small cluster). Dask lets us
describe "here are N independent chunks of work" and have it schedule them
across whatever workers are available — 1 GPU, 4 GPUs, or a laptop with none
(falls back to CPU processes).

KEY CONCEPTS:
- dask.delayed: wraps a normal Python function call so it doesn't execute
  immediately — instead it builds a task graph. Nothing runs until you call
  .compute().
- distributed.Client: spins up a local "cluster" of worker processes (or
  connects to a real multi-machine cluster if you point it at one). Each
  worker can be pinned to a specific GPU via CUDA_VISIBLE_DEVICES.
- LocalCUDACluster (from dask_cuda, optional): auto-detects all GPUs on the
  machine and creates one Dask worker per GPU. We fall back to a plain
  LocalCluster if dask_cuda isn't installed, since not everyone will have
  multiple GPUs for this project.
"""
import argparse
import os
import sys
import time

import numpy as np
from dask.distributed import Client, LocalCluster, as_completed

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "week1_cpu_baseline"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "week2_cuda"))
from genomic_parser import load_genome_sequence, clean_sequence, encode_sequence, chunk_genome
from cuda_kernel import gpu_scan_chunk, cpu_simulated_scan_chunk, CUDA_AVAILABLE


def process_one_chunk(chunk_start, chunk_data, encoded_guide, max_mismatches, gpu_id, simulate):
    """This function is what actually runs on each Dask worker. If a specific
    gpu_id is assigned, we pin this worker's process to that device before
    touching CUDA at all."""
    if gpu_id is not None and not simulate:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    scan_fn = cpu_simulated_scan_chunk if simulate else gpu_scan_chunk
    counts = scan_fn(chunk_data, encoded_guide, max_mismatches)
    hit_positions = np.where(counts <= max_mismatches)[0]
    return [(chunk_start + int(p), int(counts[p])) for p in hit_positions]


def detect_gpu_count(simulate: bool) -> int:
    if simulate:
        return 0
    try:
        from numba import cuda
        return len(cuda.gpus)
    except Exception:
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dask-distributed multi-GPU off-target scan.")
    parser.add_argument("--genome", required=True)
    parser.add_argument("--guide", required=True)
    parser.add_argument("--max-mismatches", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=None,
                         help="Force a specific worker count. Defaults to GPU count (or CPU cores if --simulate).")
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()

    gpu_count = detect_gpu_count(args.simulate)
    n_workers = args.workers or max(gpu_count, 1)
    print(f"Detected {gpu_count} GPU(s). Starting Dask cluster with {n_workers} worker(s)...")

    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1, processes=True)
    client = Client(cluster)
    print(client)

    raw = clean_sequence(load_genome_sequence(args.genome))
    encoded_genome = encode_sequence(raw)
    encoded_guide = encode_sequence(args.guide.upper())
    overlap = len(args.guide) - 1
    chunks = chunk_genome(encoded_genome, args.chunk_size, overlap)

    print(f"Submitting {len(chunks)} chunks across {n_workers} worker(s)...")
    start = time.perf_counter()

    futures = []
    for idx, (chunk_start, chunk_data) in enumerate(chunks):
        gpu_id = idx % gpu_count if gpu_count > 0 else None
        fut = client.submit(process_one_chunk, chunk_start, chunk_data, encoded_guide,
                             args.max_mismatches, gpu_id, args.simulate)
        futures.append(fut)

    all_hits = []
    for fut in as_completed(futures):
        all_hits.extend(fut.result())

    elapsed = time.perf_counter() - start

    seen = {}
    for pos, mm in all_hits:
        if pos not in seen or mm < seen[pos]:
            seen[pos] = mm

    print(f"\n--- DASK DISTRIBUTED RESULT ---")
    print(f"Workers used: {n_workers}  |  Chunks: {len(chunks)}")
    print(f"Found {len(seen)} candidate off-target sites")
    print(f"Time taken: {elapsed:.4f} seconds")

    client.close()
    cluster.close()
