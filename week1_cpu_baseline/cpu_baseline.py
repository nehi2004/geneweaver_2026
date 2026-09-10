"""
cpu_baseline.py
----------------
WHY THIS FILE EXISTS:
This is the "before" picture — a plain, single-threaded Python implementation
of guide-RNA off-target search. It's deliberately naive. Its only job is to
give you an honest timing number to compare the GPU version against in
Week 2. Do not optimize this file. The whole point of the project is showing
how much faster the GPU version is BY COMPARISON.

ALGORITHM (Hamming-distance sliding window):
For every position i in the genome, compare the `guide_length` bases starting
at i against the guide RNA, counting mismatches. If mismatches <= threshold,
record it as a candidate off-target site.
"""
import argparse
import time

import sys
import os
sys.path.append(os.path.dirname(__file__))
from genomic_parser import load_genome_sequence, clean_sequence


def find_offtargets_cpu(genome: str, guide: str, max_mismatches: int = 3):
    """Naive O(n * m) sliding window scan. n = genome length, m = guide length.
    This is exactly the kind of loop that takes days on a real 3.2-billion-base
    genome — it never touches a GPU or another thread."""
    hits = []
    guide_len = len(guide)
    n = len(genome)

    for i in range(n - guide_len + 1):
        window = genome[i:i + guide_len]
        mismatches = 0
        for a, b in zip(window, guide):
            if a != b:
                mismatches += 1
                if mismatches > max_mismatches:
                    break
        if mismatches <= max_mismatches:
            hits.append((i, mismatches, window))

    return hits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU baseline off-target scan (slow, on purpose).")
    parser.add_argument("--genome", required=True, help="Path to FASTA genome file.")
    parser.add_argument("--guide", required=True, help="Guide RNA sequence, e.g. GACCTTGATCGATGCA")
    parser.add_argument("--max-mismatches", type=int, default=3)
    args = parser.parse_args()

    print("Loading genome...")
    genome = clean_sequence(load_genome_sequence(args.genome))
    print(f"Genome length: {len(genome):,} bases. Guide: {args.guide}")

    print("Scanning (pure Python, single-threaded)...")
    start = time.perf_counter()
    hits = find_offtargets_cpu(genome, args.guide, args.max_mismatches)
    elapsed = time.perf_counter() - start

    print(f"\n--- CPU BASELINE RESULT ---")
    print(f"Found {len(hits)} candidate off-target sites (<= {args.max_mismatches} mismatches)")
    print(f"Time taken: {elapsed:.4f} seconds")
    print(f"Throughput: {len(genome) / elapsed:,.0f} bases/sec")
    print("Write this number down — Week 2's GPU kernel should crush it.")

    for pos, mm, window in hits[:5]:
        print(f"  position {pos:>10,}  mismatches={mm}  window={window}")
