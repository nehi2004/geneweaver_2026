# 🧬 GeneWeaver

**GPU-Accelerated CRISPR Off-Target Alignment Engine**

Scanning the human genome for unintended CRISPR mutations shouldn't take days on a CPU cluster. GeneWeaver offloads the search across thousands of GPU cores using Numba-compiled CUDA kernels, distributes work with Dask, and scores every candidate against real Cas9 biology — all wrapped in a live terminal dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900)
![Numba](https://img.shields.io/badge/Numba-0.67-00A3E0)
![Dask](https://img.shields.io/badge/Dask-distributed-FDA061)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## 📌 The Problem

CRISPR gene editing uses a short guide RNA sequence to target one specific location in the genome. But that guide can also bind *almost*-matching sequences elsewhere — these are **off-target sites**, and unwanted edits there can cause serious side effects. Finding them means comparing a ~20-base guide against **every single position** in a genome of up to **3.2 billion bases**.

A naive single-threaded Python scan does this one comparison at a time. At full genome scale, that's tens of billions of character comparisons — a job that takes hours to days on a single CPU core, and traditionally requires expensive proprietary HPC software to speed up.

**GeneWeaver proves this can be done on a single consumer GPU**, using only open-source Python tooling.

---

## ⚡ Results at a Glance

| Genome Size | CPU Baseline | GPU (global memory) | GPU (shared memory, optimized) | Speedup |
|---|---|---|---|---|
| 2,000,000 bases | 0.88 s | 1.09 s | 1.14 s | 0.8× *(launch overhead dominates at this size)* |
| 200,000,000 bases | 159.14 s | 30.08 s | **19.09 s** | **8.3×** |
| 1,000,000,000 bases | 537.26 s | 100.04 s | 110.51 s | 5.4× |

Tested on a single **NVIDIA GeForce RTX 3050 (6GB Laptop GPU)**. Every implementation tier — CPU, GPU, and optimized GPU — returned **identical candidate positions and mismatch counts** at every genome size, confirming correctness before speed was ever measured.

> 📎 See [`Why the speedup isn't a straight line`](#-honest-engineering-notes) below for why the optimized kernel isn't the fastest at every scale — that result is reported as measured, not smoothed over.

---

## 🖥️ Live Dashboard

An interactive web dashboard (`GeneWeaver_Dashboard.html`) visualizes the full pipeline, architecture, build log, and live scan results with red-highlighted mismatches:

```
Open GeneWeaver_Dashboard.html directly in any browser — no server, no install.
```

*(Add your own screenshots here — drag and drop into this section on GitHub, or reference `/docs/screenshots/`.)*

---

## 🏗️ Architecture

| Stage | Module | Technology |
|---|---|---|
| 01 · Parse | Genomic Parser | BioPython — FASTA ingestion, cleaning, overlap-aware chunking |
| 02 · Compute | CUDA Kernel Engine | Numba `@cuda.jit` — JIT-compiled GPU kernels, bypasses the GIL |
| 03 · Scale | Distributed Scheduler | Dask distributed — spreads chunks across every available GPU |
| 04 · Score | Scoring Engine | Custom PAM-proximity + seed-region weighted severity scoring |
| 05 · Display | TUI / Web Dashboard | Textual + Rich (terminal) and a standalone HTML dashboard |

```
FASTA file → Parser → Chunker → CUDA Kernel (per-chunk) → Dask Scheduler
           → PAM/Severity Scorer → Ranked results → Dashboard
```

---

## 📅 Four-Week Build Log

<details>
<summary><strong>Week 1 — CPU Baseline & Data Pipeline</strong></summary>

A deliberately unoptimized, single-threaded Hamming-distance scanner (`cpu_baseline.py`) sets the performance floor everything else is measured against. `genomic_parser.py` uses BioPython to ingest FASTA files, clean ambiguous bases, and chunk the genome with overlap so no candidate site is missed at chunk boundaries.

**Result:** 0.88 s on 2,000,000 bases · 34 candidate sites found (this became the correctness baseline for every later stage).
</details>

<details>
<summary><strong>Week 2 — CUDA Kernel Compilation</strong></summary>

The scan is rewritten as a Numba `@cuda.jit` kernel (`cuda_kernel.py`). Each GPU thread is assigned one genome position via `cuda.grid(1)` and independently computes its Hamming distance to the guide RNA — thousands of comparisons run simultaneously instead of sequentially. Explicit `cuda.to_device` / `copy_to_host` calls manage Host↔Device memory transfer, and the genome is processed in fixed-size chunks specifically to avoid `CUDA_OUT_OF_MEMORY` on large inputs. A Textual-based TUI scaffold (`tui_scaffold.py`) ships alongside it.

**Verified:** identical hit positions and mismatch counts to the Week 1 baseline, at every genome size tested up to 1 billion bases.
</details>

<details>
<summary><strong>Week 3 — Distributed Scaling & Biological Scoring</strong></summary>

`dask_scheduler.py` integrates Dask's distributed scheduler to split genome chunks across every detected GPU (or CPU workers in simulation mode), so the pipeline scales from one GPU to a multi-GPU machine without code changes. `scoring.py` implements PAM-proximity scoring: a raw sequence match is only biologically meaningful if a valid PAM sequence (`NGG` for Cas9) sits immediately after it, and mismatches closer to the PAM-adjacent seed region are weighted more heavily than distal ones — mirroring real Cas9 cutting behavior.

**Verified:** a perfect guide match next to an invalid PAM scores **0** (Cas9 physically cannot cut there), while a single mismatch next to a valid PAM scores 50 — confirming the ranking reflects biology, not just string similarity.
</details>

<details>
<summary><strong>Week 4 — Shared Memory Optimization & Final Dashboard</strong></summary>

`shared_memory_kernel.py` rewrites the kernel to use CUDA shared memory: each thread block cooperatively loads its required genome tile into a small, fast on-chip memory pool once, and every thread in the block reads from that shared tile instead of repeatedly hitting slower global VRAM. `tui_dashboard.py` and `main.py` complete the pipeline with live GPU telemetry (via `pynvml`), a progress bar, and a results table rendering every mismatched base in bold red.

**Result:** at 200M bases, shared memory reduced GPU time from 30.08 s to 19.09 s — a ~37% additional improvement over the Week 2 kernel.
</details>

---

## 📁 Project Structure

```
geneweaver/
├── week1_cpu_baseline/
│   ├── generate_mock_genome.py   # synthetic FASTA test-genome generator
│   ├── genomic_parser.py         # BioPython ingestion + chunking
│   └── cpu_baseline.py           # naive single-threaded scanner (baseline)
├── week2_cuda/
│   ├── cuda_kernel.py            # @cuda.jit alignment kernel
│   └── tui_scaffold.py           # Textual progress-bar UI
├── week3_dask/
│   ├── dask_scheduler.py         # multi-GPU/multi-worker distribution
│   └── scoring.py                # PAM-proximity severity scoring
├── week4_final/
│   ├── shared_memory_kernel.py   # optimized CUDA kernel (shared memory tiling)
│   └── tui_dashboard.py          # final terminal dashboard
├── main.py                       # full pipeline, single entry point
├── GeneWeaver_Dashboard.html     # standalone web dashboard (open in any browser)
├── requirements.txt
├── data/                         # generated genome files (gitignored)
└── results/                      # scan outputs (gitignored)
```

---

## 🚀 Getting Started

### Prerequisites
- An NVIDIA GPU with CUDA support (or run with `--simulate` on CPU)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Setup
```bash
conda create -n geneweaver python=3.11 -y
conda activate geneweaver
conda install -c conda-forge cudatoolkit numba -y
pip install -r requirements.txt
```

Verify Numba can see your GPU:
```bash
python -c "from numba import cuda; print(cuda.gpus)"
```

### Run the full pipeline
```bash
python week1_cpu_baseline/generate_mock_genome.py --bases 200000000 --out data/genome.fasta
python main.py --genome data/genome.fasta --guide GACCTTGATCGATGCA --pam NGG
```

Add `--simulate` to any script to run its logic on CPU if no GPU is available.

---

## 🔬 Honest Engineering Notes

At 1 billion bases, the shared-memory kernel (110.5 s) ran slightly *slower* than the plain global-memory kernel (100.0 s) from Week 2. This is reported as measured rather than adjusted, for three real reasons:

1. **Light compute-per-byte** — Hamming-distance comparison is simple enough that GPU L1/L2 cache already services much of the global-memory traffic efficiently; shared-memory tiling helps most when the compute-per-byte ratio is higher than this workload's.
2. **Synchronization overhead scales with block count** — the shared-memory kernel issues a `cuda.syncthreads()` barrier per block; at 1B bases, the much larger block count makes this cumulative cost outweigh the memory-access savings.
3. **Consumer GPU thermal throttling** — a laptop GPU under sustained multi-minute kernel execution can throttle clock speed, adding measurement noise at the largest scale.

GPU optimization is workload-dependent, not automatic — this result demonstrates that rather than hiding it.

---

## 🛠️ Tech Stack

- **Python 3.11**
- **Numba** — JIT-compiled CUDA kernels
- **CUDA Toolkit 11.8**
- **Dask (distributed)** — multi-GPU task scheduling
- **BioPython** — genomic file parsing
- **Textual + Rich** — terminal dashboard
- **pynvml** — live GPU telemetry

---

## 📄 License

MIT — free to use, modify, and build on for research or coursework.

---

*Built as a hands-on exploration of GPU computing, bioinformatics, and distributed systems — from a pure-Python baseline to a working CUDA pipeline, verified for correctness at every step.*
