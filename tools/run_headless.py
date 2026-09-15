"""Execute pacman_dqn.ipynb top to bottom without a browser and save the outputs.

This is how the submitted notebook was produced on an Intel Mac (no Colab, no VS Code).
It changes nothing inside the notebook except, optionally, EPISODES for a smoke test.

    python tools/run_headless.py                       # real run, writes outputs back into pacman_dqn.ipynb
    python tools/run_headless.py --episodes 5 --output smoke.ipynb   # setup check on a copy
"""
import argparse
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", default=str(ROOT / "pacman_dqn.ipynb"))
    parser.add_argument("--output", default=None, help="Defaults to overwriting --notebook with the executed copy.")
    parser.add_argument("--kernel", default="pacman-dqn")
    parser.add_argument("--episodes", type=int, default=None, help="Override EPISODES (smoke tests only).")
    parser.add_argument("--deadline", default=None,
                        help="Local wall-clock HH:MM. If training is still running then, send ONE interrupt to the kernel: "
                             "the notebook catches it, saves the agent as an interrupted run, and evaluation continues.")
    args = parser.parse_args()

    nb_path = Path(args.notebook)
    out_path = Path(args.output) if args.output else nb_path
    nb = nbformat.read(nb_path, as_version=4)
    if args.episodes is not None:
        for cell in nb.cells:
            if cell.cell_type == "code" and re.search(r"^EPISODES = \d+$", cell.source, flags=re.M):
                cell.source = re.sub(r"^EPISODES = \d+$", f"EPISODES = {args.episodes}", cell.source, flags=re.M)
    nb.metadata["kernelspec"] = {"name": args.kernel, "display_name": f"Python ({args.kernel})", "language": "python"}

    client = NotebookClient(nb, timeout=None, kernel_name=args.kernel,
                            resources={"metadata": {"path": str(ROOT)}},
                            record_timing=True)
    started = time.monotonic()
    print(f"Executing {nb_path.name} with kernel '{args.kernel}' ...", flush=True)
    if args.deadline:
        hh, mm = map(int, args.deadline.split(":"))
        fire_at = datetime.now().replace(hour=hh, minute=mm, second=0, microsecond=0)
        wait = (fire_at - datetime.now()).total_seconds()
        if wait > 0:
            def interrupt_once():
                print(f"Deadline {args.deadline} reached: interrupting the kernel once so training saves and evaluation runs.", flush=True)
                try:
                    client.km.interrupt_kernel()
                except Exception as error:  # noqa: BLE001
                    print(f"Interrupt failed: {error}", flush=True)
            timer = threading.Timer(wait, interrupt_once)
            timer.daemon = True
            timer.start()
            print(f"Safety deadline armed for {fire_at:%Y-%m-%d %H:%M} ({wait / 3600:.1f} h from now).", flush=True)
    try:
        client.execute()
    finally:
        nbformat.write(nb, out_path)
        print(f"Wrote {out_path} after {(time.monotonic() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    sys.exit(main())
