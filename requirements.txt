"""
main.py
--------
The finished GeneWeaver pipeline, end to end:

    parse FASTA -> chunk -> GPU (or Dask multi-GPU) scan -> PAM/severity
    scoring -> live TUI dashboard showing ranked, color-highlighted results.

USAGE:
    python main.py --genome data/mock_genome.fasta --guide GACCTTGATCGATGCA --pam NGG
    python main.py --genome data/mock_genome.fasta --guide GACCTTGATCGATGCA --simulate   # no GPU needed
    python main.py --genome data/mock_genome.fasta --guide GACCTTGATCGATGCA --dask       # multi-GPU
"""
import argparse
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "week1_cpu_baseline"))
sys.path.append(os.path.join(os.path.dirname(__file__), "week2_cuda"))
sys.path.append(os.path.join(os.path.dirname(__file__), "week3_dask"))
sys.path.append(os.path.join(os.path.dirname(__file__), "week4_final"))

from genomic_parser import load_genome_sequence, clean_sequence
from cuda_kernel import scan_genome, CUDA_AVAILABLE
from scoring import rank_hits
from tui_dashboard import GeneWeaverDashboard


def run_pipeline(args):
    print(f"[1/4] Loading genome from {args.genome} ...")
    genome_text = clean_sequence(load_genome_sequence(args.genome))
    print(f"      {len(genome_text):,} bases loaded.")

    print(f"[2/4] Scanning for off-target sites (guide={args.guide}, "
          f"max_mismatches={args.max_mismatches}) ...")
    mode = "CPU SIMULATION" if (args.simulate or not CUDA_AVAILABLE) else "GPU"
    print(f"      Mode: {mode}")

    start = time.perf_counter()
    if args.dask:
        # Import here to avoid requiring dask.distributed for users who
        # just want the single-GPU / simulate path.
        from dask_scheduler import detect_gpu_count
        os.system(
            f"python {os.path.join(os.path.dirname(__file__), 'week3_dask', 'dask_scheduler.py')} "
            f"--genome {args.genome} --guide {args.guide} --max-mismatches {args.max_mismatches} "
            f"{'--simulate' if args.simulate else ''}"
        )
        return
    else:
        raw_hits = scan_genome(args.genome, args.guide, args.max_mismatches,
                                chunk_size=1_000_000, simulate=(args.simulate or not CUDA_AVAILABLE))
    elapsed = time.perf_counter() - start
    print(f"      Found {len(raw_hits)} raw candidate sites in {elapsed:.2f}s")

    print(f"[3/4] Scoring candidates by PAM proximity + seed-region weighting (PAM={args.pam}) ...")
    positions_only = [pos for pos, _ in raw_hits]
    scored = rank_hits(genome_text, args.guide, positions_only, pam_pattern=args.pam)
    valid = [s for s in scored if s["pam_present"]]
    print(f"      {len(valid)}/{len(scored)} sites have a valid adjacent PAM (realistic cut sites).")

    print(f"[4/4] Launching live dashboard ...")
    app = GeneWeaverDashboard(results=valid, guide=args.guide, total_chunks=10)
    app.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeneWeaver — full off-target alignment pipeline.")
    parser.add_argument("--genome", required=True, help="Path to FASTA genome file.")
    parser.add_argument("--guide", required=True, help="Guide RNA sequence.")
    parser.add_argument("--pam", default="NGG", help="PAM pattern, N = wildcard. Default NGG (Cas9).")
    parser.add_argument("--max-mismatches", type=int, default=3)
    parser.add_argument("--simulate", action="store_true", help="Force CPU simulation instead of GPU.")
    parser.add_argument("--dask", action="store_true", help="Use the Dask multi-GPU scheduler instead of single-GPU.")
    args = parser.parse_args()

    run_pipeline(args)
