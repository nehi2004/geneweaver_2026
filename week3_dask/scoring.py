"""
scoring.py
-----------
WHY THIS FILE EXISTS:
Not every off-target hit is equally dangerous. A real CRISPR analysis ranks
candidates by biological severity so a researcher can triage a handful of
high-risk sites instead of manually reviewing thousands of raw hits. This
module implements a simplified but scientifically-grounded scoring scheme:

1. PAM PROXIMITY: Cas9 only cuts DNA if a valid PAM sequence (typically NGG)
   sits immediately after the target site. A near-perfect guide match with NO
   adjacent PAM is very unlikely to actually be cut — so it scores low. A
   near-perfect match WITH a PAM right next to it is high-risk.
2. MISMATCH POSITION: mismatches in the "seed region" (the ~10-12 bases
   closest to the PAM) are far more disruptive to binding than mismatches
   further away (the literature-supported basis of tools like MIT/CFD
   scoring). We approximate this with a positional weight curve.
3. MISMATCH COUNT: fewer mismatches = higher risk, obviously.

This is a simplified educational model, not a clinical-grade CFD/MIT scorer —
but it demonstrates the correct biological reasoning and is a solid
foundation to extend with a real published scoring matrix later.
"""
from typing import List, Tuple


def has_valid_pam(genome: str, position: int, guide_len: int, pam_pattern: str = "NGG") -> bool:
    """Checks whether a valid PAM sequence follows the candidate site.
    'N' in the pattern matches any base."""
    pam_start = position + guide_len
    pam_end = pam_start + len(pam_pattern)
    if pam_end > len(genome):
        return False
    candidate = genome[pam_start:pam_end]
    return all(p == "N" or p == c for p, c in zip(pam_pattern, candidate))


def seed_region_weight(mismatch_index: int, guide_len: int, seed_len: int = 12) -> float:
    """Mismatches closer to the PAM-adjacent end (the 'seed region') matter
    more. We treat the LAST `seed_len` positions of the guide as the seed
    region (closest to the PAM) and weight mismatches there higher."""
    distance_from_pam_end = guide_len - mismatch_index
    if distance_from_pam_end <= seed_len:
        return 1.0  # full weight: seed region mismatch
    return 0.4      # reduced weight: distal region mismatch, more tolerated


def score_site(genome: str, guide: str, position: int, pam_pattern: str = "NGG") -> dict:
    """Combines PAM presence + weighted mismatch penalty into one severity
    score. Higher score = higher off-target risk. Returns 0 for sites with
    no valid PAM, since Cas9 realistically won't cut there."""
    guide_len = len(guide)
    window = genome[position:position + guide_len]

    if not has_valid_pam(genome, position, guide_len, pam_pattern):
        return {"position": position, "score": 0.0, "pam_present": False,
                 "mismatches": None, "window": window}

    mismatch_indices = [i for i, (a, b) in enumerate(zip(window, guide)) if a != b]
    if not mismatch_indices:
        penalty = 0.0  # perfect match = maximum risk, zero penalty
    else:
        penalty = sum(seed_region_weight(i, guide_len) for i in mismatch_indices)

    # Score decays as penalty grows; perfect match (penalty 0) = score 100
    score = round(100.0 / (1.0 + penalty), 2)

    return {
        "position": position,
        "score": score,
        "pam_present": True,
        "mismatches": len(mismatch_indices),
        "window": window,
    }


def rank_hits(genome: str, guide: str, hit_positions: List[int], pam_pattern: str = "NGG") -> List[dict]:
    """Scores every raw hit and returns them sorted highest-risk first."""
    scored = [score_site(genome, guide, pos, pam_pattern) for pos in hit_positions]
    return sorted(scored, key=lambda s: s["score"], reverse=True)


if __name__ == "__main__":
    # Small self-contained demo so you can see the scoring logic work without
    # needing a real genome file or GPU.
    demo_genome = "TTTTGACCTTGATCGATGCAAGGTTTTTTGACCTTGATTGATGCAAGGTTTTTTGACCTTGATCGATGCATTTTTTTT"
    guide = "GACCTTGATCGATGCA"
    # Position 4: perfect match + valid AGG PAM -> highest risk.
    # Position 29: 1 mismatch + valid AGG PAM -> medium risk.
    # Position 54: perfect match but PAM is TTT (invalid) -> scores 0, Cas9 won't cut here.
    hits = [4, 29, 54]
    for result in rank_hits(demo_genome, guide, hits):
        print(result)
