"""
Yonglin-Limit LLM experiment: belief-anchor -> evidence dynamics on Delta^2
============================================================================
Direct empirical companion to the Yonglin Limit (永霖极限) derivations:
  - ch05 (eta_max = 2 min^2 / max, tangent-space spectral bound)
  - ch12 (compression + Banach, anchor A != A*, meta-layer break)

World: a bag of balls, 3 colors (Red/Blue/Black). Detector with known
reliability d reports a color. The reasoning operator F is realized by
reading the model's belief (token-prob aggregate over Red/Blue/Black)
after each incremental evidence step.

Measurements
------------
1) anchor A: zero-information 3-choice belief (pure prior simplex point).
2) evidence orbits: p_t under a report sequence; compare against the exact
   Bayesian posterior of the same sequence (closed-form, since P(r|c) is
   known), and against the book's simulated gradient-flow orbit.
3) compression ratio: two orbits from different initial hints (red-ish vs
   blue-ish priming) on the same world; k = ||p_T - p'_T||_2 / ||p_0 - p'_0||_2.
   (Banach compresses => k < 1; k > 1 would be a real counterexample.)

TRANSPORT NOTE: deepseek-v4-flash is a reasoning model -- every response
spends completion tokens on a chain-of-thought first (logprobs under
"reasoning_content", token text masked as ***), then emits the answer. If
max_tokens is too small the answer never gets generated and logprobs.content
is None (this also crashes the official openai SDK). We therefore talk to
the API over raw httpx, parse the raw JSON ourselves, and read only
logprobs.content (the final answer).

Run:
  uv run --with httpx --with numpy --with matplotlib --with tqdm python3 yonglin_llm_exp.py --mode smoke
  uv run --with httpx --with numpy --with matplotlib --with tqdm python3 yonglin_llm_exp.py --mode full
"""
import argparse, json, os, time, math
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import httpx
from tqdm import tqdm

BASE = os.path.expanduser("~/yonglin-llm-exp")
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-flash"
MAX_TOKENS = 1200          # reasoning model: chain-of-thought eats tokens first
COLS = ["red", "blue", "black"]

WORLDS = [
    (0.70, 0.20, 0.10),   # book figure setup: kappa = 7.00
    (0.60, 0.30, 0.10),
    (0.50, 0.30, 0.20),
]
D_VALS = [0.9, 0.7, 0.55]   # logLR ~ 2.2 / 0.85 / 0.2
T = 6                       # evidence steps in full mode


# --------------------------- world / posterior ---------------------------
def world_numbers(p):
    n = 10
    counts = [max(1, int(round(x * n))) for x in p]
    s = sum(counts)
    if s != n:
        counts[0] += n - s
    return tuple(counts)


def bag_prompt(p):
    c = world_numbers(p)
    return (f"A bag contains {sum(c)} balls: {c[0]} Red, {c[1]} Blue, {c[2]} Black. "
            f"The balls are shuffled. A detector examines one ball drawn at "
            f"random each time and reports its color. Known detector accuracy: "
            f"when the ball is Red it reports Red with probability {0.70:.2f} "
            f"(Blue/Black each {0.15:.2f}); likewise for Blue and Black with the "
            f"same accuracy. The detector's reports are independent.")


def report_line(r, i):
    return f" Test {i+1}: the detector says {CAPS[r]}."


CAPS = ["Red", "Blue", "Black"]


def gen_reports(seed, p_true, d, T, direction="cntr"):
    rng = np.random.default_rng(seed)
    draws = rng.choice(3, size=T, p=p_true)
    reports = []
    for c in draws:
        if rng.random() < d:
            reports.append(int(c))
        else:
            wrong = rng.choice([x for x in range(3) if x != c])
            reports.append(int(wrong))
    if direction == "supp":
        reports[0] = 0
    elif direction == "rev":
        reports[0] = 1
    return reports


def bayes_traj(p_start, reports, d):
    p = np.array(p_start, float)
    out = [p.copy()]
    for r in reports:
        lik = np.array([math.log(d) if r == c else math.log((1 - d) / 2)
                        for c in range(3)])
        p = np.exp(np.log(p) + lik)
        p = p / p.sum()
        out.append(p.copy())
    return out


# --------------------------- raw httpx transport -------------------------
def call_raw(client: httpx.Client, prompt: str, retries=3):
    """POST to DeepSeek; return raw parsed dict or None. Never swallows the
    reason silently -- prints error lines for diagnosis."""
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS, "logprobs": True,
            "top_logprobs": 20, "temperature": 0.0}
    for k in range(retries):
        try:
            r = client.post(API_URL, json=body, timeout=120,
                            headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"})
            if r.status_code == 200:
                d = r.json()
                return d
            print(f"  [api {r.status_code}] {r.text[:200]}", flush=True)
        except Exception as e:
            print(f"  [transport] {type(e).__name__}: {str(e)[:150]}", flush=True)
        if k < retries - 1:
            time.sleep(2 ** k)
    return None


def read_belief(raw: dict, labels=COLS):
    """Aggregate top-k logprobs of the FINAL answer (logprobs.content) into a
    belief vector on Delta^2. reasoning_content logprobs are masked (***) and
    irrelevant anyway."""
    ch = (raw.get("choices") or [{}])[0]
    lp = ch.get("logprobs") or {}
    cnt = lp.get("content") or None
    if not cnt:
        return None
    best = None
    for it in cnt:
        s = np.zeros(len(labels))
        for t in it.get("top_logprobs", []):
            tok = (t.get("token") or "").strip().lower()
            for i, lab in enumerate(labels):
                if tok == lab:
                    s[i] += math.exp(t.get("logprob", 0.0))
        if s.sum() > 0 and (best is None or s.sum() > best[1]):
            best = (s, s.sum())
    if best is None:
        return None
    v = np.maximum(best[0], 1e-12)
    return v / v.sum()


def probe(client, world_p, reports, d, hint=None):
    parts = [bag_prompt(world_p)]
    if hint:
        parts.append(hint)
    for i, r in enumerate(reports):
        parts.append(report_line(r, i))
    parts.append("Which color is most likely the drawn ball? "
                 "You MUST name exactly one color. Answer immediately with "
                 "exactly one word: Red, Blue or Black. Do not explain.")
    raw = call_raw(client, " ".join(parts))
    if raw is None:
        return None
    return read_belief(raw)


# --------------------------- anchors ------------------------------------
# Forced-choice formulation (the model otherwise answers "None" on zero-info
# questions, dragging the measured anchor to an artifact). Permutations of
# the option order let us average out position bias in the anchor estimate.
ANCHOR_ORDERS = [
    ("Red", "Blue", "Black"),
    ("Black", "Red", "Blue"),
    ("Blue", "Black", "Red"),
]


def make_anchor(order):
    a, b, c = order
    return (f"A completely unknown random process picks one of three colors: "
            f"{a}, {b}, {c}. No information at all about which one. "
            f"You MUST name exactly one color as your single best guess. "
            f"Answer immediately with exactly one word, the color itself. "
            f"Do not explain.")


def anchor_probe(client, reps=8):
    per_perm = {}
    all_v = []
    for order in ANCHOR_ORDERS:
        vs = []
        for _ in tqdm(range(reps), desc=f"anchor {order[0]}-{order[1]}-{order[2]}",
                      unit="call", ncols=90):
            raw = call_raw(client, make_anchor(order))
            if raw is None:
                continue
            v = read_belief(raw)
            if v is not None:
                vs.append(v)
        if vs:
            per_perm["/".join(order)] = np.mean(vs, axis=0).tolist()
            all_v.extend(vs)
    return {"per_perm": per_perm,
            "A": np.mean(all_v, axis=0).tolist() if all_v else None}


# --------------------------- compression --------------------------------
HINT_RED = ("A number of participants previously guessed Red to be most likely "
            "before seeing any test.")
HINT_BLUE = ("A number of participants previously guessed Blue to be most likely "
             "before seeing any test.")


# --------------------------- run ----------------------------------------
def run_orbit(client, world_p, d, reports, hint=None):
    p = []
    for t in range(len(reports) + 1):
        pr = probe(client, world_p, reports[:t], d, hint=hint)
        p.append(pr)
    return p   # p_0 .. p_T


def one_item(client, world, d, seed, direction="cntr", with_compression=False):
    reports = gen_reports(seed, world, d, T, direction)
    traj_true = bayes_traj(world, reports, d)
    orbit = run_orbit(client, world, d, reports, hint=None)
    rec = {"world": list(world), "d": d, "seed": seed, "dir": direction,
           "reports": list(reports), "bayes": [list(x) for x in traj_true],
           "orbit": [None if x is None else list(x) for x in orbit]}
    if with_compression:
        orb_r = run_orbit(client, world, d, reports, hint=HINT_RED)
        orb_b = run_orbit(client, world, d, reports, hint=HINT_BLUE)
        rec["orbit_red_hint"] = [None if x is None else list(x) for x in orb_r]
        rec["orbit_blue_hint"] = [None if x is None else list(x) for x in orb_b]
    return rec


def kl(p, q):
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--per-cell", type=int, default=4)
    args = ap.parse_args()

    assert os.environ.get("DEEPSEEK_API_KEY"), "export DEEPSEEK_API_KEY first"
    os.makedirs(BASE, exist_ok=True)
    client = httpx.Client(limits=httpx.Limits(max_connections=120))

    # anchor measurement
    print("measuring anchor A (zero-info K=3 belief) ...", flush=True)
    anch = anchor_probe(client, reps=8)
    A = np.array(anch["A"]) if anch["A"] is not None else None
    print(f"  anchor A = {None if A is None else np.round(A, 4)}", flush=True)
    if A is not None:
        for perm, v in anch["per_perm"].items():
            print(f"    {perm}: {np.round(np.array(v), 4)}", flush=True)
    with open(f"{BASE}/anchor.json", "w") as f:
        json.dump({"A": anch["A"], "per_perm": anch["per_perm"],
                   "kappa_A": None if A is None else float(A.max() / A.min())}, f)

    if args.mode == "smoke":
        cells = [(WORLDS[0], D_VALS[1], 1)]
        with_compression = True
    else:
        cells = [(w, d, s) for w in WORLDS for d in D_VALS for s in range(args.per_cell)]
        with_compression = True

    recs = []
    def run(cell):
        return one_item(client, *cell, with_compression=with_compression)

    with ThreadPoolExecutor(max_workers=100) as ex:
        futures = [ex.submit(run, c) for c in cells]
        for f in tqdm(as_completed(futures), total=len(cells),
                      desc="cells", unit="cell", ncols=100):
            recs.append(f.result())

    with open(f"{BASE}/orbits.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    # ---------------- summary + figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for r in recs:
        orbit = [np.array(x) for x in r["orbit"] if x is not None]
        bayes = [np.array(x) for x in r["bayes"]]
        if not orbit:
            continue
        E = [kl(bayes[t], orbit[t]) for t in range(min(len(orbit), len(bayes)))]
        rows.append({**{k: v for k, v in r.items() if k != "orbit"},
                     "E": E, "kl_to_A":
                     None if A is None else [kl(A, x) for x in orbit]})
    comp = []
    for r in recs:
        orb_r = [np.array(x) for x in r.get("orbit_red_hint", []) if x is not None]
        orb_b = [np.array(x) for x in r.get("orbit_blue_hint", []) if x is not None]
        if len(orb_r) >= 2 and len(orb_b) >= 2:
            d0 = np.linalg.norm(orb_r[0] - orb_b[0])
            dT = np.linalg.norm(orb_r[-1] - orb_b[-1])
            comp.append({"d": r["d"], "k": float(dT / d0) if d0 > 0 else 0.0,
                         "d0": float(d0), "dT": float(dT)})

    comps = {}
    for c in comp:
        comps.setdefault(c["d"], []).append(c["k"])
    print("\ncompression ratio k (||pT-p'T|| / ||p0-p'0||, L2):")
    for d, ks in sorted(comps.items()):
        print(f"  d={d}: k = {np.mean(ks):.3f} +- {np.std(ks):.3f}  (n={len(ks)})")

    def xy(p):
        return np.array([p[1] * 1.0 + p[2] * 0.5, p[2] * np.sqrt(3) / 2])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    r0 = recs[0]
    ax = axes[0]
    if A is not None:
        ax.scatter(*xy(A), marker="*", s=260, color="#555555", zorder=5, label="anchor A")
    orb = [np.array(x) for x in r0["orbit"] if x is not None]
    bay = [np.array(x) for x in r0["bayes"]]
    o = np.array([xy(x) for x in orb])
    b = np.array([xy(x) for x in bay])
    ax.plot(*b.T, "--o", color="#0072B2", label="exact Bayes", ms=4)
    ax.plot(*o.T, "-o", color="#D55E00", label="LLM orbit", ms=5)
    for t in range(len(o)):
        ax.annotate(str(t), xy=o[t], fontsize=8, color="#D55E00")
    ax.set_title(f"orbit vs Bayes (world {r0['world']}, d={r0['d']})")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.axis("off")

    ax = axes[1]
    for r in rows[:6]:
        ax.plot(range(len(r["E"])), r["E"], "-" if r["d"] == 0.7 else "--",
                label=f"d={r['d']}", alpha=0.7)
    ax.set_xlabel("evidence step t")
    ax.set_ylabel("KL(bayes_t || p_t)")
    ax.set_title("energy gap to exact posterior")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ds = sorted(comps.keys())
    ax.bar(range(len(ds)), [np.mean(comps[d]) for d in ds],
           yerr=[np.std(comps[d]) for d in ds], capsize=4,
           color="#0072B2", alpha=0.85)
    ax.axhline(1.0, color="#D55E00", ls="--", label="k=1 (no compression)")
    ax.set_xticks(range(len(ds)))
    ax.set_xticklabels([f"d={d}" for d in ds])
    ax.set_ylabel("k")
    ax.set_title("empirical compression ratio")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{BASE}/fig_yonglin_llm.png", dpi=140)
    print(f"\nsaved: {BASE}/fig_yonglin_llm.png")


if __name__ == "__main__":
    main()
