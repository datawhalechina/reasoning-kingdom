"""
ADS Optimizer vs Adam vs SGD — Tiny GPT Ablation (Fair Comparison)
==================================================================
Key changes for fairness:
  - Larger dataset: 512 samples, train/val split (384/128)
  - Re-sample mini-batches each step (no memorization)
  - Track VALIDATION loss (generalization, not memorization)
  - Longer training: 500 steps
  - Light theme figure

Output: docs/public/figures/ch12_ads_optimizer_ablation.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import copy

np.random.seed(42)

# ── Tiny GPT config ──────────────────────────────
T, D, H_dim, C = 16, 32, 16, 8
BATCH = 32
STEPS = 500
ETA_TARGET = 0.02
N_TOTAL = 512
N_TRAIN = 384
N_VAL = N_TOTAL - N_TRAIN


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis, keepdims=True))
    return e / e.sum(axis, keepdims=True)


def cross_entropy(logits, y):
    p = softmax(logits)
    return -np.log(p[np.arange(len(y)), y] + 1e-10).mean(), p


def entropy_of_probs(probs, n_classes):
    H_max = np.log(n_classes)
    H = -(probs * np.log(probs + 1e-10)).sum(-1).mean()
    B = min(float(H / H_max), 1 - 1e-6)
    alpha = -np.log(1 - B)
    return B, alpha


def init_params():
    s = 0.02
    return {
        "Wq": np.random.randn(D, H_dim)*s, "Wk": np.random.randn(D, H_dim)*s,
        "Wv": np.random.randn(D, H_dim)*s, "Wo": np.random.randn(H_dim, D)*s,
        "W1": np.random.randn(D, D*2)*s,   "b1": np.zeros(D*2),
        "W2": np.random.randn(D*2, D)*s,   "b2": np.zeros(D),
        "Wout": np.random.randn(D, C)*s,   "bout": np.zeros(C),
    }


def forward(x, p):
    mask = np.triu(np.full((T, T), -1e9), 1)
    Q = x @ p["Wq"]; K = x @ p["Wk"]; V = x @ p["Wv"]
    A = softmax(Q @ K.transpose(0, 2, 1) / H_dim**0.5 + mask)
    attn = A @ V @ p["Wo"]
    h = x + attn
    ff = np.maximum(0, h @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]
    h2 = h + ff
    logits = h2[:, -1, :] @ p["Wout"] + p["bout"]
    return logits, A, h, h2, V


def compute_grads(x, y, p):
    B_size = x.shape[0]
    logits, A, h, h2, V = forward(x, p)
    loss, probs = cross_entropy(logits, y)

    dlogits = probs.copy()
    dlogits[np.arange(B_size), y] -= 1
    dlogits /= B_size

    grads = {}
    grads["Wout"] = h2[:, -1, :].T @ dlogits
    grads["bout"] = dlogits.sum(0)

    dh2 = np.zeros_like(h2)
    dh2[:, -1, :] = dlogits @ p["Wout"].T

    grads["W2"] = np.einsum('bti,btj->ij', np.maximum(0, h @ p["W1"] + p["b1"]), dh2)
    grads["b2"] = dh2.sum((0, 1))

    dh_ff = (dh2 @ p["W2"].T) * (h @ p["W1"] + p["b1"] > 0)
    grads["W1"] = np.einsum('bti,btj->ij', h, dh_ff)
    grads["b1"] = dh_ff.sum((0, 1))

    dh = dh2 + dh_ff @ p["W1"].T
    attn_out = A @ V
    grads["Wo"] = np.einsum('bth,btd->hd', attn_out, dh)

    dattn_h = dh @ p["Wo"].T
    dV = np.einsum('bts,bsh->bth', A, dattn_h)
    grads["Wv"] = np.einsum('btd,bth->dh', x, dV)
    grads["Wq"] = np.einsum('btd,bth->dh', x, dattn_h) * 0.01
    grads["Wk"] = np.einsum('btd,bth->dh', x, dattn_h) * 0.01

    return loss, probs, grads


def eval_loss(x, y, p):
    logits = forward(x, p)[0]
    loss, probs = cross_entropy(logits, y)
    return loss, probs


def grad_norm(grads):
    return np.sqrt(sum(np.sum(g**2) for g in grads.values()))


# ── Adam ──────────────────────────────────────────
def init_adam_state(params):
    state = {}
    for k in params:
        state[k] = {"m": np.zeros_like(params[k]),
                     "v": np.zeros_like(params[k]), "t": 0}
    return state


def adam_update(params, grads, state, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    for k in params:
        s = state[k]
        s["t"] += 1
        s["m"] = beta1 * s["m"] + (1 - beta1) * grads[k]
        s["v"] = beta2 * s["v"] + (1 - beta2) * grads[k]**2
        m_hat = s["m"] / (1 - beta1**s["t"])
        v_hat = s["v"] / (1 - beta2**s["t"])
        params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)


# ── Dataset ───────────────────────────────────────
X_all = np.random.randn(N_TOTAL, T, D).astype(np.float32)
y_all = np.random.randint(0, C, N_TOTAL)
X_train, y_train = X_all[:N_TRAIN], y_all[:N_TRAIN]
X_val, y_val = X_all[N_TRAIN:], y_all[N_TRAIN:]

init_p = init_params()
rng = np.random.default_rng(123)

# ── Run ───────────────────────────────────────────
results = {}

for name in ["SGD (Fixed LR)", "Adam", "ADS Optimizer"]:
    p = copy.deepcopy(init_p)
    train_losses, val_losses = [], []
    lrs_log, alphas, entropies, gnorms = [], [], [], []

    if name == "ADS Optimizer":
        idx0 = rng.choice(N_TRAIN, BATCH, replace=False)
        _, p0_probs = eval_loss(X_train[idx0], y_train[idx0], p)
        _, alpha0 = entropy_of_probs(p0_probs, C)
        eta0_ads = ETA_TARGET * (1 + alpha0)

    if name == "Adam":
        adam_state = init_adam_state(p)

    for t in range(STEPS):
        # mini-batch sampling
        idx = rng.choice(N_TRAIN, BATCH, replace=False)
        xb, yb = X_train[idx], y_train[idx]

        loss, probs, grads = compute_grads(xb, yb, p)
        B_t, alpha_t = entropy_of_probs(probs, C)
        gn = grad_norm(grads)

        if name == "SGD (Fixed LR)":
            lr = ETA_TARGET
            for k in p:
                p[k] -= lr * grads[k]

        elif name == "Adam":
            lr = ETA_TARGET
            adam_update(p, grads, adam_state, lr)

        elif name == "ADS Optimizer":
            lr = eta0_ads / (1 + alpha_t)
            for k in p:
                p[k] -= lr * grads[k]

        # validation loss
        v_loss, v_probs = eval_loss(X_val, y_val, p)

        train_losses.append(loss)
        val_losses.append(v_loss)
        lrs_log.append(lr)
        alphas.append(alpha_t)
        entropies.append(B_t)
        gnorms.append(gn)

    results[name] = {
        "train_losses": train_losses, "val_losses": val_losses,
        "lrs": lrs_log, "alphas": alphas,
        "entropies": entropies, "gnorms": gnorms,
    }

# ── Plot (light theme) ───────────────────────────
colors = {
    "SGD (Fixed LR)": "#D35400",
    "Adam": "#27AE60",
    "ADS Optimizer": "#2E86C1",
}

fig = plt.figure(figsize=(18, 10), facecolor="white")
gs = gridspec.GridSpec(2, 3, hspace=0.38, wspace=0.30,
                       left=0.06, right=0.96, top=0.91, bottom=0.08)

axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
panel_titles = [
    "Training Loss (log scale)",
    "Validation Loss (log scale)",
    "Effective Learning Rate",
    "Log-Barrier  α = −log(1−B)",
    "Gradient L2 Norm (log scale)",
    "Final Validation Loss",
]

for ax, title in zip(axes, panel_titles):
    ax.set_facecolor("#FAFAFA")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8, color="#2C3E50")
    ax.tick_params(labelsize=9, colors="#2C3E50")
    for spine in ax.spines.values():
        spine.set_color("#BDC3C7")
    ax.grid(True, alpha=0.3, color="#95A5A6", linestyle="--")

steps_x = np.arange(1, STEPS + 1)

# 0: Train Loss
ax = axes[0]
for name, r in results.items():
    ax.semilogy(steps_x, r["train_losses"], color=colors[name], linewidth=2, label=name, alpha=0.85)
ax.set_xlabel("Step", color="#2C3E50"); ax.set_ylabel("Cross-Entropy Loss", color="#2C3E50")
ax.legend(fontsize=9, facecolor="white", edgecolor="#BDC3C7")

# 1: Val Loss
ax = axes[1]
for name, r in results.items():
    ax.semilogy(steps_x, r["val_losses"], color=colors[name], linewidth=2, label=name, alpha=0.85)
ax.set_xlabel("Step", color="#2C3E50"); ax.set_ylabel("Validation Loss", color="#2C3E50")
ax.legend(fontsize=9, facecolor="white", edgecolor="#BDC3C7")

# 2: LR
ax = axes[2]
for name, r in results.items():
    ax.plot(steps_x, r["lrs"], color=colors[name], linewidth=2, label=name, alpha=0.85)
ax.set_xlabel("Step", color="#2C3E50"); ax.set_ylabel("Learning Rate", color="#2C3E50")
ax.legend(fontsize=9, facecolor="white", edgecolor="#BDC3C7")

# 3: Alpha
ax = axes[3]
for name, r in results.items():
    ax.plot(steps_x, r["alphas"], color=colors[name], linewidth=2, label=name, alpha=0.85)
ax.set_xlabel("Step", color="#2C3E50"); ax.set_ylabel("α", color="#2C3E50")
ax.legend(fontsize=9, facecolor="white", edgecolor="#BDC3C7")

# 4: Grad norm
ax = axes[4]
for name, r in results.items():
    ax.semilogy(steps_x, r["gnorms"], color=colors[name], linewidth=2, label=name, alpha=0.85)
ax.set_xlabel("Step", color="#2C3E50"); ax.set_ylabel("‖∇L‖₂", color="#2C3E50")
ax.legend(fontsize=9, facecolor="white", edgecolor="#BDC3C7")

# 5: Bar chart
ax = axes[5]
names_list = list(results.keys())
final_vals = [results[n]["val_losses"][-1] for n in names_list]
bars = ax.bar(names_list, final_vals, color=[colors[n] for n in names_list],
              edgecolor="#2C3E50", linewidth=0.5, width=0.5)
for bar, val in zip(bars, final_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f"{val:.4f}", ha="center", va="bottom", color="#2C3E50",
            fontsize=11, fontweight="bold")
ax.set_ylabel("Final Val Loss", color="#2C3E50")
ax.set_ylim(0, max(final_vals) * 1.3)

fig.suptitle(
    "ADS Optimizer vs Adam vs SGD  —  Tiny GPT Ablation  (512 samples, 500 steps, val split)",
    fontsize=15, fontweight="bold", y=0.97, color="#2C3E50"
)

out_path = "docs/public/figures/ch12_ads_optimizer_ablation.png"
fig.savefig(out_path, dpi=180, facecolor="white")
plt.close()

print("=" * 65)
print("ADS Optimizer vs Adam vs SGD — Tiny GPT (Fair Comparison)")
print("=" * 65)
print(f"  Dataset: {N_TOTAL} total, {N_TRAIN} train, {N_VAL} val")
print(f"  Batch: {BATCH},  Steps: {STEPS},  eta_target: {ETA_TARGET}")
for name in results:
    r = results[name]
    print(f"\n  {name}:")
    print(f"    Final Train Loss: {r['train_losses'][-1]:.6f}")
    print(f"    Final Val Loss:   {r['val_losses'][-1]:.6f}")
    print(f"    Final LR:         {r['lrs'][-1]:.6f}")
    print(f"    Final Alpha:      {r['alphas'][-1]:.4f}")
    print(f"    Final Entropy B:  {r['entropies'][-1]:.4f}")
print(f"\n  Figure saved: {out_path}")
