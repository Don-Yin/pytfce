"""Rich-based TUI for parallel experiment progress tracking.

Displays one progress row per experiment, updating concurrently from
separate ``multiprocessing.Process`` workers via a shared ``Queue``.
Uses fork start method for macOS compatibility (avoids pickle issues).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import platform
import signal
import sys
import time
import traceback
from typing import Any, Callable

if platform.system() == "Darwin":
    mp.set_start_method("fork", force=True)

from rich.console import Console
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def _worker(func: Callable, name: str, queue: mp.Queue,
            result_pipe, args: tuple) -> None:
    """Process target — runs *func*, sends lightweight status via pipe."""
    try:
        result = func(queue, *args)
        result_pipe.send(("ok", name))
    except Exception as exc:
        traceback.print_exc()
        result_pipe.send(("error", str(exc)))
    finally:
        queue.put({"name": name, "phase": "done", "current": 100, "total": 100})
        try:
            result_pipe.close()
        except OSError:
            pass


class ParallelRunner:
    """Launch experiment functions in parallel with a live multi-row TUI.

    Each experiment function signature: ``func(queue, *args) -> result``
    Progress dicts on the queue: ``{"name": ..., "current": ..., "total": ..., "phase": ...}``
    """

    def __init__(self, experiments: list[tuple[str, Callable, tuple]]):
        self.experiments = experiments
        self.console = Console(force_terminal=True)
        self._processes: list[tuple[str, mp.Process]] = []

    def _cleanup(self, signum=None, frame=None):
        """Kill all child processes on interrupt."""
        for name, p in self._processes:
            if p.is_alive():
                p.kill()
        for _, p in self._processes:
            p.join(timeout=3)
        if signum is not None:
            sys.exit(1)

    def run(self) -> dict[str, Any]:
        queue: mp.Queue = mp.Queue()
        pipes = {}

        for name, func, args in self.experiments:
            parent_conn, child_conn = mp.Pipe(duplex=False)
            p = mp.Process(
                target=_worker,
                args=(func, name, queue, child_conn, args),
                daemon=True,
            )
            self._processes.append((name, p))
            pipes[name] = parent_conn

        prev_sigint = signal.signal(signal.SIGINT, self._cleanup)
        prev_sigterm = signal.signal(signal.SIGTERM, self._cleanup)

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.fields[exp_name]:<18}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("[cyan]{task.fields[phase]:<20}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )

        task_ids = {}
        for name, _, _ in self.experiments:
            tid = progress.add_task(name, total=100, exp_name=name, phase="starting")
            task_ids[name] = tid

        for _, p in self._processes:
            p.start()

        alive = set(range(len(self._processes)))

        try:
            with Live(progress, console=self.console, refresh_per_second=8):
                while alive:
                    try:
                        msg = queue.get(timeout=0.25)
                    except Exception:
                        msg = None

                    if msg and isinstance(msg, dict):
                        mname = msg.get("name", "")
                        tid = self._match_task(mname, task_ids)
                        if tid is not None:
                            progress.update(
                                tid,
                                completed=msg.get("current", 0),
                                total=msg.get("total", 100),
                                phase=msg.get("phase", ""),
                            )
                            if msg.get("phase") == "done":
                                progress.update(tid, completed=msg.get("total", 100),
                                                phase="[green]done")

                    for i in list(alive):
                        if not self._processes[i][1].is_alive():
                            alive.discard(i)
                            name = self._processes[i][0]
                            if name in task_ids:
                                progress.update(task_ids[name], completed=100,
                                                phase="[green]done")
        except KeyboardInterrupt:
            self.console.print("\n[red bold]Interrupted — killing all workers...[/]")
            self._cleanup()
            return {}
        finally:
            signal.signal(signal.SIGINT, prev_sigint)
            signal.signal(signal.SIGTERM, prev_sigterm)

        results = {}
        for name, p in self._processes:
            p.join(timeout=30)
            try:
                if pipes[name].poll(timeout=5):
                    status, val = pipes[name].recv()
                    results[name] = val if status == "ok" else {"error": val}
            except (OSError, EOFError, BrokenPipeError):
                results[name] = {"error": "pipe broken (worker may have been killed)"}
        return results

    @staticmethod
    def _match_task(name: str, task_ids: dict) -> Any | None:
        if name in task_ids:
            return task_ids[name]
        nl = name.lower()
        for tn, tid in task_ids.items():
            tl = tn.lower()
            if nl in tl or tl in nl:
                return tid
            if nl.split("-")[0] == tl.split("-")[0]:
                return tid
        return None
