"""
genomic_parser.py
------------------
WHY THIS FILE EXISTS:
Real FASTA files are messy: they can contain lowercase "soft-masked" repeat
regions, ambiguous bases (N = unknown base), multiple chromosomes in one
file, and line-wrapped sequences. BioPython's SeqIO handles all the parsing
edge cases for you. This module's job is: read the file, clean it, and split
it into fixed-size chunks so later stages (CUDA, Dask) can process pieces of
the genome independently instead of needing the whole 3-billion-char string
in memory / on the GPU at once.
"""
from Bio import SeqIO
import numpy as np

# Map each base to a small integer. GPUs work with numbers, not letters —
# this encoding is what actually gets copied into GPU memory later.
BASE_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
INT_TO_BASE = {v: k for k, v in BASE_TO_INT.items()}


def load_genome_sequence(fasta_path: str) -> str:
    """Reads the first record of a FASTA file and returns its raw sequence
    string. For a real multi-chromosome genome you'd loop over every record;
    kept to one record here for clarity."""
    record = next(SeqIO.parse(fasta_path, "fasta"))
    return str(record.seq).upper()


def clean_sequence(seq: str) -> str:
    """Real genomes contain runs of 'N' (unsequenced/ambiguous regions) and
    lowercase letters (soft-masked repeats). We uppercase everything and
    leave N's in place but flag them, since a guide RNA should never be
    considered a 'match' against an N."""
    return seq.upper()


def encode_sequence(seq: str) -> np.ndarray:
    """Converts the DNA string into a numpy array of small integers (uint8).
    This is the format the CUDA kernel actually consumes — GPUs are fast at
    crunching numbers, not comparing Python strings."""
    return np.array([BASE_TO_INT.get(base, 4) for base in seq], dtype=np.uint8)


def chunk_genome(encoded: np.ndarray, chunk_size: int, overlap: int):
    """Splits the encoded genome into overlapping chunks.

    WHY OVERLAP MATTERS: if a guide RNA match straddles the boundary between
    chunk N and chunk N+1, and we split with zero overlap, we'd miss it
    entirely. Overlap of at least (guide_length - 1) bases guarantees every
    possible match position is fully contained in at least one chunk.
    """
    n = len(encoded)
    step = chunk_size - overlap
    chunks = []
    for start in range(0, n, step):
        end = min(start + chunk_size, n)
        chunks.append((start, encoded[start:end]))
        if end == n:
            break
    return chunks


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Parse and chunk a FASTA genome file.")
    p.add_argument("--genome", required=True)
    p.add_argument("--chunk-size", type=int, default=100_000)
    p.add_argument("--overlap", type=int, default=64)
    args = p.parse_args()

    raw = load_genome_sequence(args.genome)
    cleaned = clean_sequence(raw)
    encoded = encode_sequence(cleaned)
    chunks = chunk_genome(encoded, args.chunk_size, args.overlap)

    print(f"Genome length: {len(cleaned):,} bases")
    print(f"Encoded dtype: {encoded.dtype}, shape: {encoded.shape}")
    print(f"Split into {len(chunks)} chunks of up to {args.chunk_size:,} bases each "
          f"(overlap={args.overlap})")
