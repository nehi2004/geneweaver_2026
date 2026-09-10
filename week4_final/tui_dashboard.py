"""
tui_dashboard.py
------------------
WHY THIS FILE EXISTS:
This is the finished, polished terminal dashboard the project description
promises: live GPU memory/utilization stats, alignment progress, and a
results table that highlights mutated (mismatched) base pairs in red so a
researcher can visually scan results at a glance.

GPU telemetry uses `pynvml` (NVIDIA's management library bindings) when
available; falls back to a "no GPU stats available" placeholder otherwise so
the dashboard still runs on a machine without a GPU for UI development.
"""
import asyncio
from typing import List, Tuple

from textual.app import App, ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Header, Footer, ProgressBar, Label, DataTable, Static
from rich.text import Text

try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except Exception:
    PYNVML_AVAILABLE = False


def get_gpu_stats() -> str:
    """Builds a GPU stats string defensively. Some environments (virtualized /
    'Managed Device' GPUs, certain cloud instances, driver quirks) support
    SOME nvml queries but not others -- e.g. memory info works but utilization
    queries return NVMLError_Unknown. Each stat is fetched independently so
    one unsupported call doesn't crash the whole dashboard; unsupported
    stats are simply omitted from the line instead."""
    if not PYNVML_AVAILABLE:
        return "GPU stats unavailable (pynvml not initialized / no NVIDIA GPU found)"

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        return "GPU stats unavailable (could not get device handle)"

    parts = []

    try:
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        parts.append(name)
    except Exception:
        pass

    try:
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_gb = mem.used / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)
        parts.append(f"VRAM {used_gb:.2f}/{total_gb:.2f} GB")
    except Exception:
        pass

    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        parts.append(f"Core util {util.gpu}%")
    except Exception:
        parts.append("Core util n/a (not supported on this device)")

    return "  |  ".join(parts) if parts else "GPU stats unavailable on this device"


def render_mismatch_window(window: str, guide: str) -> Text:
    """Builds a Rich Text object where mismatched bases are shown in red bold
    and matching bases are plain -- this is what makes mutations visually
    pop in the results table."""
    text = Text()
    for genome_base, guide_base in zip(window, guide):
        if genome_base == guide_base:
            text.append(genome_base)
        else:
            text.append(genome_base, style="bold red")
    return text


class GeneWeaverDashboard(App):
    CSS = """
    #top { height: auto; }
    #gpu_stats { color: cyan; margin: 1 0; }
    DataTable { height: 1fr; }
    """
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, results: List[dict] = None, guide: str = "", total_chunks: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.results = results or []
        self.guide = guide
        self.total_chunks = total_chunks

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="top"):
            yield Label("GeneWeaver -- Live Alignment Dashboard")
            yield Static(get_gpu_stats(), id="gpu_stats")
            yield ProgressBar(total=self.total_chunks, id="progress")
        yield DataTable(id="results_table")
        yield Footer()

    async def on_mount(self) -> None:
        table = self.query_one("#results_table", DataTable)
        table.add_columns("Position", "Mismatches", "Sequence (mutations in red)", "Severity Score")
        self.set_interval(2.0, self.refresh_gpu_stats)
        self.run_worker(self.simulate_and_populate())

    def refresh_gpu_stats(self) -> None:
        self.query_one("#gpu_stats", Static).update(get_gpu_stats())

    async def simulate_and_populate(self) -> None:
        bar = self.query_one("#progress", ProgressBar)
        table = self.query_one("#results_table", DataTable)

        for i in range(1, self.total_chunks + 1):
            bar.advance(1)
            await asyncio.sleep(0.1)

        for r in sorted(self.results, key=lambda x: x.get("score", 0), reverse=True):
            table.add_row(
                f"{r['position']:,}",
                str(r.get("mismatches", "-")),
                render_mismatch_window(r["window"], self.guide),
                f"{r.get('score', 0):.1f}",
            )


if __name__ == "__main__":
    demo_guide = "GACCTTGATCGATGCA"
    demo_results = [
        {"position": 104822, "mismatches": 1, "window": "GACCTTGATCGATGCT", "score": 82.3},
        {"position": 552011, "mismatches": 2, "window": "GACGTTGATCGATGCA", "score": 61.7},
        {"position": 998120, "mismatches": 0, "window": "GACCTTGATCGATGCA", "score": 100.0},
    ]
    app = GeneWeaverDashboard(results=demo_results, guide=demo_guide, total_chunks=15)
    app.run()