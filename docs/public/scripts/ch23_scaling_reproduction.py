"""
Figure 23.8: reproducing the scaling-law straight line from the simplex.

Panel (a): the data-axis scaling law.  For a K-answer conditional distribution
q estimated from s samples, the leading finite-sample excess KL is

    E D_KL(q || q_hat) = (K - 1) / (2 s) + O(s^-2),

because  E[(q_hat - q)(q_hat - q)^T] = (diag(q) - q q^T) / s  and
tr(H_T (diag(q) - q q^T)) = K - 1  with H_T = P_T diag(1/q) P_T.
On log-log axes this is a straight line of slope -1; K only shifts the
intercept.

Panel (b): the other half of the same Hessian.  For near-vertex symmetric
anchors q(t) = (1-t)/K * 1 + t * e_1 the tangent condition number satisfies
kappa_T ~ K, so the linearised orbit decays as rho*^t with
rho* = (kappa_T - 1)/(kappa_T + 1).  Larger K means a longer road to the
same accuracy.

The figure is generated entirely from the formulas; it reproduces the
*format* of a scaling curve, not any particular laboratory's exponent.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------- tangent spectrum -----------------------------
def tangent_spectrum(q, iters=100):
    """Eigenvalues of P_T diag(1/q) P_T on T = {sum = 0}, via secular equation."""
    d = np.sort(1.0 / np.asarray(q, float))
    K = len(d)
    roots = np.empty(K - 1)
    for j in range(K - 1):
        lo, hi = d[j], d[j + 1]
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            with np.errstate(divide="ignore"):
                f = np.sum(1.0 / (d - mid))
            if f > 0:
                hi = mid
            else:
                lo = mid
        roots[j] = 0.5 * (lo + hi)
    return roots


def contraction_rate(q):
    lam = tangent_spectrum(q)
    kappa = lam.max() / lam.min()
    return kappa, (kappa - 1.0) / (kappa + 1.0)


# ----------------------------- panel data ---------------------------------
Ks = [3, 6, 12]
colors = ["#9ecae1", "#4292c6", "#08519c"]
s = np.logspace(1, 5, 120)

t_anchor = 0.9
rates = {}
for K in Ks:
    q = (1.0 - t_anchor) / K * np.ones(K)
    q[0] += t_anchor
    rates[K] = contraction_rate(q)

# ----------------------------- figure -------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.88, 4.28), dpi=200)

# (a) data-axis scaling law
ax = axes[0]
for K, c in zip(Ks, colors):
    excess = (K - 1.0) / (2.0 * s)
    ax.loglog(s, excess, color=c, lw=2.2,
              label=rf"$K={K}$")
    idx = np.array([0, 30, 60, 90, 119])
    ax.loglog(s[idx], excess[idx], "o", color=c, ms=4.2)

ax.set_xlabel("samples $s$", fontsize=13)
ax.set_ylabel(r"excess KL  $L(s)-H(q)$", fontsize=13)
ax.set_title("(a) data-axis scaling law: a plain $1/s$ count", fontsize=15, pad=9)
ax.grid(True, which="major", alpha=0.30)
ax.grid(True, which="minor", alpha=0.12)
ax.tick_params(labelsize=11.5)
ax.legend(fontsize=11.5, frameon=False, loc="lower left")

# slope guide
xs = np.array([2.0e4, 8.0e4])
ys = 3.0 / xs
ax.plot(xs, ys, color="#555555", lw=1.4, ls=(0, (4, 3)))
ax.annotate("slope $-1$", xy=(5.0e4, 3.0 / 5.0e4), xytext=(1.4e4, 5.5e-4),
            fontsize=11.5, color="#444444",
            arrowprops=dict(arrowstyle="-", color="#777777", lw=1.0))
ax.text(1.3e4, 4.0e-2, "same slope,\ndifferent intercept", fontsize=11,
        color="#444444", va="top", ha="left")

# (b) semi-log orbits
ax = axes[1]
t = np.arange(0, 31)
for K, c in zip(Ks, colors):
    kappa, rho = rates[K]
    ax.semilogy(t, rho ** t, color=c, lw=2.2, marker="o", ms=3.8,
                label=rf"$K={K}$  ($\rho^*={rho:.3f}$)")

ax.axhline(1e-4, color="#888888", lw=1.2, ls=(0, (4, 3)))
ax.text(30.5, 1.05e-4, r"same $\varepsilon$", fontsize=11, color="#666666",
        va="bottom", ha="right")
ax.set_xlabel("step $t$", fontsize=13)
ax.set_ylabel(r"error  $\|p_t-A\|_2$", fontsize=13)
ax.set_title(r"(b) the same Hessian: semi-log orbits, $\kappa_T\approx K$",
             fontsize=15, pad=9)
ax.set_xlim(0, 31)
ax.set_ylim(1e-10, 2.0)
ax.grid(True, which="major", alpha=0.30)
ax.grid(True, which="minor", alpha=0.12)
ax.tick_params(labelsize=11.5)
ax.legend(fontsize=10.8, frameon=False, loc="lower left")

fig.tight_layout(pad=0.7)
out = "docs/public/figures/ch23_scaling_reproduction.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", out)
print("rates:", {K: tuple(round(x, 4) for x in rates[K]) for K in Ks})
