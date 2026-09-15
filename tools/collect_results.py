"""Copy the evidence from one pacman_runs/<run> folder into results/ and print README-ready tables.

    python tools/collect_results.py pacman_runs/20260915_130000_000000

Checkpoints (*.pt) are deliberately not copied; they stay in the local run folder and its ZIP.
"""
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["config.json", "baseline.json", "comparison.json", "training.csv", "training_summary.json",
         "training_dashboard.png", "demo_scores.json"]


def main(run_dir):
    run = Path(run_dir)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "demos").mkdir(exist_ok=True)
    for name in FILES:
        if (run / name).exists():
            shutil.copy2(run / name, out / name)
    for gif in sorted((run / "demos").glob("*.gif")):
        shutil.copy2(gif, out / "demos" / gif.name)

    config = json.loads((run / "config.json").read_text())
    summary = json.loads((run / "training_summary.json").read_text())
    comparison = json.loads((run / "comparison.json").read_text())
    rows = list(csv.DictReader((run / "training.csv").open()))
    before, after = comparison["before"], comparison["after"]

    print(f"\nRun folder: {run}")
    print(f"Choices: exploration={config['exploration']}, episodes_requested={config['episodes_requested']}, "
          f"learning_rate={config['learning_rate']}, device={config['device']}, python={config['python']}")
    print(f"Packages: {config['packages']}")
    print(f"Summary: {summary}")
    elapsed = summary["elapsed_seconds_including_periodic_demos"]
    print(f"Elapsed: {elapsed / 60:.1f} min ({elapsed / 3600:.2f} h)")
    print("\n| Game | Seed | Untrained score | Trained score | Change |")
    print("|---|---|---|---|---|")
    for i, (seed, b, a) in enumerate(zip(before["seeds"], before["scores"], after["scores"]), 1):
        print(f"| {i} | {seed} | {b:.0f} | {a:.0f} | {a - b:+.0f} |")
    print(f"| **Mean** | | **{before['mean']:.1f}** | **{after['mean']:.1f}** | **{after['mean'] - before['mean']:+.1f}** |")
    print(f"\nTime-limited games before/after: {sum(before['time_limited'])}/{sum(after['time_limited'])}")
    print(f"Game lengths (decisions) before: {before['steps']} | after: {after['steps']}")
    scores = [float(r["score"]) for r in rows]
    if rows:
        n = len(scores)
        first = scores[:max(1, n // 4)]
        last = scores[-max(1, n // 4):]
        print(f"\nTraining games: {n} | first-quarter mean {sum(first)/len(first):.1f} | last-quarter mean {sum(last)/len(last):.1f} "
              f"| best {max(scores):.0f} (episode {scores.index(max(scores)) + 1}) | truncated at time limit: {sum(r['truncated'] == 'True' for r in rows)}")
        losses = [float(r["mean_loss"]) for r in rows if r["mean_loss"] not in ("nan", "")]
        if losses:
            print(f"Mean loss: first {losses[0]:.4f} -> last {losses[-1]:.4f} | min {min(losses):.4f} | max {max(losses):.4f}")
    demos = json.loads((run / "demo_scores.json").read_text()) if (run / "demo_scores.json").exists() else []
    if demos:
        print("\n| Checkpoint episode | Demo score (seed 101) | GIF |")
        print("|---|---|---|")
        for d in demos:
            print(f"| {d['episode']} | {d['scores'][0]:.0f} | results/demos/episode_{d['episode']:04d}.gif |")
    print(f"\nCopied into {out}: {sorted(p.name for p in out.iterdir())} + {len(list((out / 'demos').glob('*.gif')))} GIFs")


if __name__ == "__main__":
    main(sys.argv[1])
