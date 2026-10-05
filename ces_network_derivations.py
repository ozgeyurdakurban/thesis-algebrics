"""
Symbolic and numerical verification for the subsection
"The CES Specification on Networks".

Verifies:
  (1) the network CES first-order condition c_i = kappa_i X_i with
      kappa_i = ((1-delta)/(delta*theta_i))**sigma * Lambda_i**(sigma-1),
      sigma = 1/(1-rho);
  (2) the Cobb-Douglas limit kappa_i -> B_i/(A_i theta_i) as rho -> 0;
  (3) the sign result d log kappa / d log Lambda = sigma - 1;
  (4) coincidence of the act-based (Lambda = 1) and impact-based models
      at sigma = 1 on the star, and their divergence away from it;
  (5) robustness of the positional ordering (periphery > center) on the
      star over a grid of (a, beta, sigma).

Run:  python3 ces_network_derivations.py
"""

import numpy as np
import sympy as sp

# ------------------------------------------------------------------ (1)-(3)
rho, delta, theta, Lam = sp.symbols("rho delta theta Lambda", positive=True)
sigma = 1 / (1 - rho)
kappa = ((1 - delta) / (delta * theta)) ** sigma * Lam ** (sigma - 1)

# (1) FOC identity: delta*theta = (1-delta)*Lam**rho * kappa**(rho-1)
iden = (1 - delta) * Lam ** rho * kappa ** (rho - 1) - delta * theta
rng = np.random.default_rng(7)
for _ in range(500):
    subs = {rho: rng.uniform(-4, 0.95), delta: rng.uniform(0.05, 0.95),
            theta: rng.uniform(0.05, 0.95), Lam: rng.uniform(1.0, 3.5)}
    assert abs(complex(iden.evalf(subs=subs))) < 1e-8
print("(1) FOC identity holds (500 random draws).")

# (2) Cobb-Douglas limit
assert sp.simplify(sp.limit(kappa, rho, 0) - (1 - delta) / (delta * theta)) == 0
print("(2) rho -> 0 limit reproduces kappa = B/(A theta).")

# (3) elasticity of kappa wrt Lambda
elas = sp.simplify(sp.diff(sp.log(kappa), Lam) * Lam)
assert sp.simplify(elas - (sigma - 1)) == 0
print("(3) d log kappa / d log Lambda = sigma - 1.")

# ------------------------------------------------------------------ (4)-(5)
E, N = 15.0, 3


def kap(a, th, lam, s):
    return (a / ((1 - a) * th)) ** s * lam ** (s - 1)


def star_eq(a, beta, s, impact=True, n=N, e=E):
    th_c, th_l = 1 - beta / n, 1 - beta / 2
    lam_c = 1 + (n - 1) * beta / 2 if impact else 1.0
    lam_l = 1 + beta / n if impact else 1.0
    k_c, k_l = kap(a, th_c, lam_c, s), kap(a, th_l, lam_l, s)
    c_c = c_l = 1.0
    for _ in range(100000):
        n_c = min(e, k_c * (e + (beta / n) * (n - 1) * c_l) / (1 + k_c * th_c))
        n_l = min(e, k_l * (e + (beta / 2) * c_c) / (1 + k_l * th_l))
        if abs(n_c - c_c) < 1e-12 and abs(n_l - c_l) < 1e-12:
            break
        c_c, c_l = n_c, n_l
    return c_c, c_l


# (4) coincidence at sigma = 1; divergence otherwise
for a in (0.05, 0.10, 0.15):
    assert np.allclose(star_eq(a, 1.5, 1.0, True), star_eq(a, 1.5, 1.0, False),
                       atol=1e-9)
d_lo = star_eq(0.10, 1.5, 0.5, True)[0] - star_eq(0.10, 1.5, 0.5, False)[0]
d_hi = star_eq(0.10, 1.5, 2.0, True)[0] - star_eq(0.10, 1.5, 2.0, False)[0]
assert d_lo < -1e-3 < 1e-3 < d_hi
print("(4) act = impact at sigma = 1; impact < act for sigma < 1; "
      "impact > act for sigma > 1.")

# (5) positional ordering over the grid
viol = tot = 0
for beta in np.arange(1.25, 1.96, 0.05):
    for s in (0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0):
        for a in np.arange(0.02, 0.42, 0.04):
            c_c, c_l = star_eq(a, beta, s, True)
            tot += 1
            viol += c_c > c_l + 1e-8
print(f"(5) periphery >= center in {tot - viol}/{tot} grid points "
      f"(violations: {viol}).")
assert viol == 0
print("All checks passed.")

