"""
Compression check via phase-split (Yonglin Limit companion, v2)
================================================================
Problem with v1 compression: two "hint" orbits start at the same no-evidence
belief (world parameters dominate), so d0 ~ 0 and k = dT/d0 blows up.

Phase-split design: same world, same d, same seed. Two orbits differ only in
the FIRST detector report (Red vs Blue), then share the identical report tail
[2..T]. The initial divergence d1 = ||p1^R - p1^B|| > 0 is purely a phase
offset; if F (the reasoning operator with the same evidence stream) is
contractive, d_t decays to 0. This measures the memory half-life of a unit
phase shift -- the empirical form of the effective-reasoning-window.

Outputs: compress_phase.jsonl, fig_compress_phase.png
"""
import json, math, os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import httpx
from tqdm import tqdm

from yonglin_llm_exp import (call_raw, read_belief, probe, gen_reports,
                             WORLDS, D_VALS, T, BASE)

N_SEEDS = 4


def orbit(client, world, d, seq):
    """belief at t=0..T for a given report sequence (reports[:t] in prompt)."""
    return [probe(client, world, seq[: t], d) for t in range(T + 1)]


def one_cell(client, world, d, seed):
    base = gen_reports(seed, world, d, T)
    seqR = [0] + base[1:]
    seqB = [1] + base[1:]
    orbR = orbit(client, world, d, seqR)
    orbB = orbit(client, world, d, seqB)
    ok = [t for t in range(1, T + 1)
          if orbR[t] is not None and orbB[t] is not None]
    dist = {}
    prev = None
    for t in ok:
        d_t = float(np.linalg.norm(orbR[t] - orbB[t]))
        dist[t] = d_t
        if prev is None:
            prev = d_t
    # per-step contraction ratios and half-life
    steps = sorted(dist.keys())
    ratios = {}
    for a, b in zip(steps, steps[1:]):
        if dist[a] > 0:
            ratios[b] = dist[b] / dist[a]
    half = None
    for t in steps:
        if dist[t] <= 0.5 * max(dist.values()):
            half = t
            break
    return {"world": list(world), "d": d, "seed": seed,
            "dist": dist, "ratios": ratios, "half_life": half}


def main():
    os.makedirs(BASE, exist_ok=True)
    client = httpx.Client(limits=httpx.Limits(max_connections=120))
    cells = [(w, d, s) for w in WORLDS for d in D_VALS for s in range(N_SEEDS)]
    recs = []

    def run(c):
        return one_cell(client, *c)

    with ThreadPoolExecutor(max_workers=100) as ex:
        for f in tqdm(as_completed([ex.submit(run, c) for c in cells]),
                      total=len(cells), desc="phase cells", unit="cell", ncols=100):
            recs.append(f.result())

    with open(f"{BASE}/compress_phase.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    # ---------------- summary ----------------
    print("\nphase-split contraction (d_t = ||p_t^R - p_t^B||_2):")
    for d in sorted(D_VALS):
        g = [r for r in recs if r["d"] == d]
        ds = [r["dist"] for r in g]
        tmax = max(max(r["dist"]) for r in g if r["dist"])
        print(f"\n  d={d}:")
        # median trajectory at t=1..T
        for t in range(1, T + 1):
            vals = [r["dist"].get(t) for r in g if r["dist"].get(t) is not None]
            if vals:
                print(f"    t={t}: median d = {np.median(vals):.4f} "
                      f"(min {np.min(vals):.4f} / max {np.max(vals):.4f})")
        # per-step ratio
        rats = []
        for r in g:
            rats += [v for v in r["ratios"].values() if v is not None]
        if rats:
            rats = np.array(rats)
            print(f"    per-step ratio: median {np.median(rats):.4f}  "
                  f"(mean {np.mean(rats):.4f} +- {np.std(rats):.4f})")
        halves = [r["half_life"] for r in g if r["half_life"] is not None]
        if halves:
            print(f"    half-life steps: median {np.median(halves):.1f} (n={len(halves)})")

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    colors = {0.55: "#D55E00", 0.7: "#0072B2", 0.9: "#009E73"}
    for d in sorted(D_VALS):
        g = [r for r in recs if r["d"] == d]
        ts = list(range(1, T + 1))
        med = []
        for t in ts:
            vals = [r["dist"].get(t) for r in g if r["dist"].get(t) is not None]
            med.append(np.median(vals) if vals else np.nan)
        ax.plot(ts, med, "o-", color=colors[d], label=f"d={d}")
    ax.set_yscale("log")
    ax.set_xlabel("step t after split"); ax.set_ylabel("||p_t^R - p_t^B||_2 (log)")
    ax.set_title("phase-split memory decay")
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1]
    ds = sorted(D_VALS)
    halves = [np.median([r["half_life"] for r in recs if r["d"] == d and r["half_life"] is not None]) for d in ds]
    ax.bar(range(len(ds)), halves, color=[colors[d] for d in ds], alpha=0.85)
    ax.set_xticks(range(len(ds))); ax.set_xticklabels([f"d={d}" for d in ds])
    ax.set_ylabel("median half-life (steps)")
    ax.set_title("memory half-life of a phase shift")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{BASE}/fig_compress_phase.png", dpi=140)
    print(f"\nsaved: {BASE}/fig_compress_phase.png")


if __name__ == "__main__":
    main()
