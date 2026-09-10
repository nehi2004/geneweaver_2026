"""
tui_scaffold.py
----------------
WHY THIS FILE EXISTS:
A researcher running a multi-minute GPU job needs live feedback, not a blank
terminal. Textual lets us build a proper terminal dashboard in pure Python.
This is the Week 2 scaffold: just a progress bar tracking simulated chunk
processing, so the UI skeleton exists before Week 4 wires in real GPU stats.

RUN IT DIRECTLY:
    python tui_scaffold.py
Press 'q' to quit.
"""
import asyncio

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Header, Footer, ProgressBar, Label


class GeneWeaverScaffold(App):
    CSS = """
    Vertical {
        align: center middle;
        height: 100%;
    }
    #status {
        margin-bottom: 1;
    }
    """
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, total_chunks: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.total_chunks = total_chunks

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield Label("GeneWeaver — Genomic Chunk Processing", id="status")
            yield ProgressBar(total=self.total_chunks, id="progress")
        yield Footer()

    async def on_mount(self) -> None:
        self.run_worker(self.simulate_progress())

    async def simulate_progress(self) -> None:
        bar = self.query_one("#progress", ProgressBar)
        label = self.query_one("#status", Label)
        for chunk_num in range(1, self.total_chunks + 1):
            label.update(f"Processing genomic chunk {chunk_num}/{self.total_chunks}...")
            bar.advance(1)
            await asyncio.sleep(0.15)  # placeholder for real chunk processing time
        label.update("All chunks processed. Press 'q' to quit.")


if __name__ == "__main__":
    app = GeneWeaverScaffold(total_chunks=20)
    app.run()
