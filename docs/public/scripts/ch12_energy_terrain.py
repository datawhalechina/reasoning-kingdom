"""
Energy Landscape of Reasoning: Closed-Form Curvature and Safe Step Size
========================================================================
Demonstration accompanying the Yonglin Limit (永霖极限) derivation.

Setup: belief simplex Delta^2 (3 classes). The reasoning energy is
    E(p) = D_KL(p* || p),  p* = target/attractor belief.
Inference = Euler steps of the gradient flow on E with projection to the
simplex:  p_{t+1} = proj_Delta(p_t - eta * grad E(p_t)).

Closed-form facts under study (true at the fixed point p = p*):
    Hess E(p)  = diag(q_i / p_i^2)   [general position; q_i = p*_i]
               -> diag(1/p_i)        [at p = p*]
    mu = lambda_min = 1/max_i p_i,  L = lambda_max = 1/min_i p_i
    kappa = L/mu = max_i p_i / min_i p_i          (spectral ratio / condition number)
    eta_max = 2 mu / L^2 = 2 / (L * kappa) = 2 min_i p_i^2 / max_i p_i

On the tangent space {v: sum v_i = 0} (the subspace preserved by the
projection) the exact local stability boundary is
    eta_crit = 2 / theta_max,  theta_max = largest eigenvalue of
    diag(1/q_i) restricted to the tangent space;
    for K = 3, theta_max solves  3 q0 q1 q2 th^2 - 2 (q0q1+q0q2+q1q2) th + 1 = 0.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patheffects import withStroke

rng = np.random.default_rng(42)


# ----------------------------- core helpers -----------------------------
def proj_simplex(v):
    """Euclidean projection onto probability simplex (vectorized over last axis).
    Standard algorithm: find the last index j with u_j - (css_j - 1)/j > 0."""
    v = np.asarray(v, float)
    shape = v.shape
    v = v.reshape(-1, shape[-1])
    K = shape[-1]
    u = np.sort(v, axis=1)[:, ::-1]
    css = np.cumsum(u, axis=1)
    avg = (css - 1.0) / np.arange(1, K + 1)
    cond = u - avg > 0
    j = np.where(cond)[1]    # column indices of True entries (rows repeat)
    # last True index per row:
    last = np.array([np.where(cond[i])[0][-1] for i in range(len(cond))])
    theta = (css[np.arange(len(v)), last] - 1.0) / (last + 1)
    out = np.maximum(v - theta[:, None], 0.0)
    return out.reshape(shape)


def energy(p, pstar):
    """E(p) = D_KL(p* || p), elementwise-safe (p clipped away from 0)."""
    pc = np.clip(p, 1e-300, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.sum(pstar * np.log(pstar / pc), axis=-1)
    return np.where(np.all(p > 0, axis=-1), e, np.inf)


def grad_energy(p, pstar):
    return -pstar / np.clip(p, 1e-12, 1.0)


def euler_step(p, pstar, eta):
    return proj_simplex(p - eta * grad_energy(p, pstar))


def eta_max_closed(pstar):
    """Closed-form safe step size at the fixed point: 2 min^2 / max."""
    return 2.0 * np.min(pstar) ** 2 / np.max(pstar)


def hessian_numeric(p, pstar, eps=1e-6):
    """Central-difference Hessian of E at p (K=3)."""
    K = len(p)
    H = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            ei = np.zeros(K); ej = np.zeros(K)
            ei[i] = eps; ej[j] = eps
            H[i, j] = (energy(p + ei + ej, pstar) - energy(p + ei - ej, pstar)
                       - energy(p - ei + ej, pstar) + energy(p - ei - ej, pstar)) / (4 * eps * eps)
    return H


def to_xy(p):
    """Barycentric -> Cartesian (A=e1 at (0,0), B=e2 at (1,0), C=e3 at (0.5,sqrt3/2))."""
    return p[..., 1] * 1.0 + p[..., 2] * 0.5, p[..., 2] * (np.sqrt(3) / 2)


def from_xy(x, y):
    s = np.sqrt(3.0) / 2.0
    p3 = np.clip(y / s, 0.0, 1.0)
    p2 = np.clip(x - y / (np.sqrt(3.0)), 0.0, 1.0)
    p1 = 1.0 - p2 - p3
    mask = (p1 >= -1e-9) & (p2 >= -1e-9) & (p3 >= -1e-9)
    p1 = np.clip(p1, 0.0, 1.0)
    return np.stack([p1, p2, p3], axis=-1), mask


# ----------------------------- figure panels -----------------------------
PSTAR = np.array([0.70, 0.20, 0.10])
EMAX = eta_max_closed(PSTAR)          # ~ 2*0.01/0.7 = 0.02857
ETA_ILLUS = 0.8 * EMAX

fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.4))
ps = withStroke(linewidth=3, foreground="white")

# ---------- (a) energy terrain + trajectories ----------
ax = axes[0, 0]
xgrid = np.linspace(-0.05, 1.05, 300)
ygrid = np.linspace(-0.05, np.sqrt(3) / 2 + 0.05, 260)
X, Y = np.meshgrid(xgrid, ygrid)
P, mask = from_xy(X, Y)
E = energy(P, PSTAR)
E = np.where(mask, E, np.nan)
lv = np.geomspace(1e-5, 50, 40)
cf = ax.contourf(X, Y, E, levels=lv, cmap="viridis", norm=LogNorm(vmin=1e-5, vmax=50))
cbar = fig.colorbar(cf, ax=ax, shrink=0.85)
cbar.set_label(r"$E(p)=D_{KL}(p^{*}\|p)$  [log scale]", fontsize=9)

starts = [np.array([1/3, 1/3, 1/3]), np.array([0.15, 0.60, 0.25]), np.array([0.45, 0.25, 0.30])]
for p0 in starts:
    p = p0.copy()
    xs, ys = [], []
    for t in range(120):
        p = euler_step(p, PSTAR, ETA_ILLUS)
        xs.append(to_xy(p)[0]); ys.append(to_xy(p)[1])
        if energy(p, PSTAR) < 1e-6:
            break
    ax.plot(xs, ys, lw=1.8, color="#D55E00", alpha=0.9)
    ax.plot(xs[0], ys[0], "o", ms=6, color="#D55E00", mfc="white")
    ax.annotate("$p_0$", (xs[0], ys[0]), textcoords="offset points", xytext=(-2, 8),
                fontsize=9, color="#D55E00")

px, py = to_xy(PSTAR)
ax.plot(px, py, "*", ms=22, color="#0072B2", mec="k", mew=0.8, zorder=10)
ax.annotate(r"$p^{*}$  (fixed point)", (px, py), textcoords="offset points",
            xytext=(8, -16), fontsize=11, color="#0072B2", fontweight="bold")
ax.text(0.98, 0.94, f"$\\eta=0.8\\,\\eta_{{max}}$ = {ETA_ILLUS:.4f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
ax.set_title("(a) The energy terrain of reasoning", fontsize=12, fontweight="bold")
ax.text(0.5, -0.14, "belief simplex $\\Delta^2$: $p_1$-vertex (left), $p_2$-vertex (right), $p_3$-vertex (top)",
        transform=ax.transAxes, ha="center", fontsize=8, color="0.35")
ax.set_aspect("equal"); ax.axis("off")

# ---------- (b) curvature spectrum along one trajectory ----------
ax = axes[0, 1]
p = np.array([1/3, 1/3, 1/3])
lmax_a, lmin_a, kap_a = [], [], []
lmax_n, lmin_n = [], []
p_rec = []
for t in range(80):
    p = euler_step(p, PSTAR, ETA_ILLUS)
    p_rec.append(p.copy())
for p in p_rec:
    lmax_a.append(np.max(PSTAR / p**2)); lmin_a.append(np.min(PSTAR / p**2))
    H = hessian_numeric(p, PSTAR)
    w = np.linalg.eigvalsh(H)
    lmax_n.append(w[-1]); lmin_n.append(w[0])
ts = np.arange(len(p_rec))
ax.plot(ts, lmax_a, color="#0072B2", lw=1.8, label=r"closed form  $\lambda_{max}$")
ax.plot(ts, lmin_a, color="#009E73", lw=1.8, label=r"closed form  $\lambda_{min}$")
ax.plot(ts, lmax_n, "o", ms=4.2, color="#0072B2", mfc="none", label=r"numerical Hessian  $\lambda_{max}$")
ax.plot(ts, lmin_n, "s", ms=4.2, color="#009E73", mfc="none", label=r"numerical Hessian  $\lambda_{min}$")
ax.set_yscale("log")
ax.set_xlabel("reasoning step  $t$", fontsize=10)
ax.set_ylabel("curvature eigenvalues  $\\lambda$", fontsize=10)
ax.set_title("(b) Curvature spectrum: closed form = numerical", fontsize=12, fontweight="bold")
ax.legend(fontsize=8, framealpha=0.9, loc="upper right")
ax.grid(alpha=0.3)
kap = PSTAR.max() / PSTAR.min()
ax.text(0.98, 0.08, r"$\kappa = L/\mu = %.2f = 0.70/0.10$ (at $p^{*}$)" % kap,
        transform=ax.transAxes, ha="right", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

# ---------- (c) analytic eta_max surface ----------
ax = axes[1, 0]
XS, YS = np.meshgrid(np.linspace(0, 1, 260), np.linspace(0, np.sqrt(3) / 2, 230))
PS, maskC = from_xy(XS, YS)
eta = np.where(maskC & np.all(PS > 1e-4, axis=-1),
               2.0 * np.min(PS, axis=-1) ** 2 / np.max(PS, axis=-1), np.nan)
eta = np.where(maskC, eta, np.nan)
lv2 = np.geomspace(1e-6, 0.25, 35)
cf2 = ax.contourf(XS, YS, eta, levels=lv2, cmap="magma", norm=LogNorm(vmin=1e-6, vmax=0.25))
cbar2 = fig.colorbar(cf2, ax=ax, shrink=0.85)
cbar2.set_label(r"$\eta_{max}(p)=2\min_i p_i^2/\max_i p_i$  [log]", fontsize=9)
ax.plot(px, py, "*", ms=22, color="#56B4E9", mec="k", mew=0.8, zorder=10)
ax.annotate(r"$p^{*}$", (px, py), textcoords="offset points", xytext=(6, 10),
            fontsize=11, color="#56B4E9", fontweight="bold")
ax.set_title("(c) Analytic safe step size on the simplex", fontsize=12, fontweight="bold")
ax.text(0.5, -0.14, "$\\eta_{max} \\to 0$ near the edges: the terrain dictates the pace",
        transform=ax.transAxes, ha="center", fontsize=8.5, color="0.35")
ax.set_aspect("equal"); ax.axis("off")

# ---------- (d) closed form vs numerical stability boundary ----------
ax = axes[1, 1]

def theta_max_tangent(q):
    """Largest eigenvalue of diag(q_i^{-1}) restricted to the tangent space {v: sum v_i = 0}.
    K = 3: analytic root of  3 q0 q1 q2 th^2 - 2 (q0q1+q0q2+q1q2) th + 1 = 0."""
    q0, q1, q2 = q
    a = 3 * q0 * q1 * q2
    b = -2 * (q0 * q1 + q0 * q2 + q1 * q2)
    c = 1.0
    disc = b * b - 4 * a * c
    th = (-b + np.sqrt(disc)) / (2 * a)
    return th


def eta_crit_tangent_closed(q):
    """Exact local stability boundary: eta_safe < 2 / lambda_max of Hessian on the tangent space."""
    return 2.0 / theta_max_tangent(q)


# ---------- (d) closed form vs numerical stability boundary ----------
ax = axes[1, 1]

etas_scan = np.logspace(-4.0, 0.6, 140)

# --- LOCAL analysis: perturbations of p*; exact prediction = tangent-space spectrum ---
local_ana, local_num = [], []
for _ in range(30):
    ps_ = rng.dirichlet(np.ones(3)) * 0.9 + 0.05
    ps_ = ps_ / ps_.sum()
    pert = [proj_simplex(ps_ + 0.02 * rng.normal(0, 1, 3) * np.array([1, 0.5, 0.25])) for _ in range(8)]
    # asymptotic stability: final distance does not grow
    worst = np.zeros(len(etas_scan))
    for p0 in pert:
        P = np.broadcast_to(p0.copy(), (len(etas_scan), 3)).copy()
        d0 = np.linalg.norm(p0 - ps_)
        for _ in range(500):
            P = proj_simplex(P - etas_scan[:, None] * (-ps_ / np.clip(P, 1e-12, 1.0)))
        worst = np.maximum(worst, np.linalg.norm(P - ps_, axis=1) / d0)
    idx = np.where(worst < 1.0)[0]
    local_num.append(etas_scan[idx[-1]] if len(idx) else etas_scan[0] / 2)
    local_ana.append(eta_crit_tangent_closed(ps_))

# --- GLOBAL analysis: from uniform prior; conservative closed form = full-space spectrum ---
glob_ana, glob_num = [], []
for _ in range(30):
    ps_ = rng.dirichlet(np.ones(3)) * 0.9 + 0.05
    ps_ = ps_ / ps_.sum()
    P = np.broadcast_to(np.full(3, 1 / 3), (len(etas_scan), 3)).copy()
    for _ in range(400):
        P = proj_simplex(P - etas_scan[:, None] * (-ps_ / np.clip(P, 1e-12, 1.0)))
    fk = np.sum(ps_ * np.log(np.clip(ps_, 1e-12, 1) / np.clip(P, 1e-12, 1)), axis=1)
    idx = np.where(fk < 1e-3)[0]
    glob_num.append(etas_scan[idx[-1]] if len(idx) else etas_scan[0] / 2)
    glob_ana.append(eta_max_closed(ps_))

local_ana, local_num = np.array(local_ana), np.array(local_num)
glob_ana, glob_num = np.array(glob_ana), np.array(glob_num)

ax.loglog(local_ana, local_num, "o", ms=7, color="#0072B2", alpha=0.85, zorder=6,
          label=r"local: $2/\lambda_{\max}$ on tangent space (exact)")
ax.loglog(glob_ana, glob_num, "^", ms=7, color="#CC79A7", alpha=0.85, zorder=5,
          label=r"global: conservative bound $2/(L\kappa)$")
lims = [min(local_ana.min(), local_num.min()) * 0.6, max(local_ana.max(), local_num.max()) * 1.6]
ax.loglog(lims, lims, "k--", lw=1.5, label="y = x (exact prediction)")
logR = np.corrcoef(np.log10(local_ana), np.log10(local_num))[0, 1] ** 2
med = np.median(np.abs(local_num - local_ana) / local_ana)
ax.set_xlabel(r"closed-form  $\eta_{crit}$", fontsize=10)
ax.set_ylabel("numerical stability threshold  $\\eta_{crit}$", fontsize=10)
ax.set_title("(d) Closed form = the exact local stability boundary", fontsize=12, fontweight="bold")
ax.legend(fontsize=8.5, loc="upper left")
ax.grid(alpha=0.3, which="both")
ax.text(0.03, 0.05, f"exact (tangent spectrum): $R^2$ = {logR:.4f}, median err = {med:.2%}\nconservative (full spectrum): $\\eta_{{crit}} \\geq 2/(L\\kappa)$ always",
        transform=ax.transAxes, fontsize=9, color="k", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))

fig.suptitle("The Energy Landscape of Reasoning: closed-form geometry and its verification\n"
             r"$E(p)=D_{KL}(p^{*}\|p)$,  inference $=$ Euler steps of $-\nabla E$ projected to the simplex",
             fontsize=13.5, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = "ch12_energy_terrain.png"   # saves next to the script
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved:", out)
print(f"eta_max closed form (p* = {PSTAR}): {EMAX:.6f}")
print(f"kappa at fixed point: {PSTAR.max()/PSTAR.min():.2f}")
print(f"panel d: local R^2(log-log) = {logR:.4f}, n = {len(local_ana)}, median err = {med:.2%}")
print(f"panel d: global: numerically stable region >= eta_max (projection adds stability)")
