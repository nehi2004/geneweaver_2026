"""
generate_mock_genome.py
------------------------
WHY THIS FILE EXISTS:
You don't have a real 3.2-billion-base human genome file sitting around (and
if you did, it'd be ~3GB+ just for one copy). This script creates a synthetic
.fasta file of any size you want, filled with random A/C/G/T bases, and
deliberately plants a handful of "off-target-like" sequences near PAM sites
so you have something known to search for later.

USAGE:
    python generate_mock_genome.py --bases 2000000 --out ../data/mock_genome.fasta
"""
import argparse
import random

BASES = "ACGT"


def build_genome(num_bases: int, guide: str, num_planted_sites: int, seed: int = 42) -> str:
    """Builds a random DNA string and plants near-matches to `guide` at random
    spots, each immediately followed by an NGG-style PAM, so Week 3's scoring
    logic has real targets to find."""
    rng = random.Random(seed)
    genome = [rng.choice(BASES) for _ in range(num_bases)]

    for _ in range(num_planted_sites):
        pos = rng.randint(0, num_bases - len(guide) - 5)
        # Plant a near-match: copy the guide but mutate 1-3 random bases
        planted = list(guide)
        n_mutations = rng.randint(1, 3)
        for _ in range(n_mutations):
            i = rng.randint(0, len(planted) - 1)
            planted[i] = rng.choice(BASES.replace(planted[i], ""))
        genome[pos:pos + len(guide)] = planted
        # Plant a PAM (NGG) directly after it
        pam_start = pos + len(guide)
        genome[pam_start:pam_start + 3] = [rng.choice(BASES), "G", "G"]

    return "".join(genome)


def write_fasta(path: str, header: str, sequence: str, line_width: int = 70):
    with open(path, "w") as f:
        f.write(f">{header}\n")
        for i in range(0, len(sequence), line_width):
            f.write(sequence[i:i + line_width] + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic FASTA genome for testing.")
    parser.add_argument("--bases", type=int, default=2_000_000, help="Number of base pairs to generate.")
    parser.add_argument("--guide", type=str, default="GACCTTGATCGATGCA", help="Guide RNA sequence to plant near-matches for.")
    parser.add_argument("--sites", type=int, default=25, help="Number of off-target-like sites to plant.")
    parser.add_argument("--out", type=str, default="data/mock_genome.fasta", help="Output FASTA path.")
    args = parser.parse_args()

    print(f"Generating {args.bases:,} synthetic bases with {args.sites} planted near-match sites...")
    genome = build_genome(args.bases, args.guide, args.sites)
    write_fasta(args.out, "mock_chromosome_1 synthetic test genome", genome)
    print(f"Done. Wrote {args.out}")
