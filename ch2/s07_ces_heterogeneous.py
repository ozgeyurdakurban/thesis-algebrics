"""
Step 3c -- CES equilibria with HETEROGENEOUS prosocial weights B_i (general N, networks).

With the CES share pi_i = pi(B_i; p_i, sigma) (s05), every CD result of s04 carries
over with B_i -> pi_i. Derived / verified here:
  1. K_n: c_i* = e phi_i/mu with phi_i = n pi_i/(n - (1-pi_i) beta), mu = 1 - (beta/n) sum phi;
     ordering by B_i (pi increasing in B on a common position);
  2. any network: c* = (I - Gamma~ G)^{-1} alpha~ and
        d c_i*/d B_j = M_ij c_j* sigma (1 - pi_j) / (B_j (1 - B_j))   (> 0);
  3. star S_n: c_C* = pi_C e[n(2-beta) + 2 beta Sigma_pi]/[(n-beta)(2-beta) - pi_C beta^2 Sigma_pi],
     Sigma_pi = sum of leaf shares; leaf j: c_j* = pi_j (2e + beta c_C*)/(2-beta);
     threshold in shares and its closed-form image in B;
  4. cycle C_n: matrix form;
  5. formula-free check against direct maximisation of the CES utility;
  6. act-based versus impact-based: odds ratio Lambda_i^(sigma-1), positional factor
     (Lambda_C/Lambda_L)^(sigma-1) on the star.
"""

import numpy as np
import sympy as sp

from common import Log, beta, e, n, same, simp
import netlib as NL

L = Log(__file__)
rng = np.random.default_rng(11)

# ===========================================================================
L.head("1. K_n, heterogeneous shares pi_i, general n")
# ===========================================================================
pii, C, ci = sp.symbols("pi_i C c_i", positive=True)
sol_i = simp(sp.solve(sp.Eq(ci, pii / (1 - beta / n) * (e + beta / n * (C - ci))), ci)[0])
phi = n * pii / (n - (1 - pii) * beta)
L.check("c_i = phi_i (e + (beta/n) C), phi_i = n pi_i/(n - (1-pi_i) beta)", same(sol_i, phi * (e + beta / n * C)))
Phi = sp.symbols("Phi", positive=True)
Cs = simp(sp.solve(sp.Eq(C, Phi * (e + beta / n * C)), C)[0])
L.check("C* = e Phi/mu and c_i* = e phi_i/mu, mu = 1 - (beta/n) Phi",
        same(Cs, e * Phi / (1 - beta / n * Phi)) and same(phi * (e + beta / n * Cs), e * phi / (1 - beta / n * Phi)))
L.tex_eq("CES K_n het", r"c_i^\star", e * sp.Symbol(r"\varphi_i") / (1 - beta / n * sp.Symbol(r"\Phi")))

# numerical: closed form = explicit linear solve = formula-free utility maximisation
worst = 0.0
for _ in range(25):
    m = int(rng.integers(2, 6)); b = rng.uniform(1.05, min(m, 3) - 0.05); sg = float(rng.choice([0.3, 0.6, 1.4, 2.5]))
    Bv = rng.uniform(0.02, 0.2, m); lam = float(rng.choice([0.0, 1.0, rng.uniform(0, 3)]))
    G = NL.complete(m)
    pi = NL.shares(G, Bv, b, sg, lam=lam)
    ph = m * pi / (m - (1 - pi) * b); mu = 1 - b / m * ph.sum()
    x = NL.nash_closed(G, Bv, b, 15.0, sg, lam=lam)
    if mu > 0 and np.all(15 * ph / mu < 15):
        worst = max(worst, np.max(abs(x - 15 * ph / mu)))
    worst = max(worst, np.max(abs(x - NL.nash_utility(G, Bv, b, 15.0, sg, lam=lam))))
L.say(f"  max deviation (phi/mu formula, active-set solver, utility maximisation) = {worst:.2e}")
L.check("K_n heterogeneous CES: phi/mu closed form = direct utility maximisation (25 games, impact weight lambda in {0, 1, U(0,3)})", worst < 1e-5)
ok = True
for _ in range(2000):
    m = int(rng.integers(2, 7)); b = rng.uniform(1.05, m - 0.05); sg = rng.uniform(0.1, 5)
    Bv = rng.uniform(0.01, 0.4, m); x = NL.nash_closed(NL.complete(m), Bv, b, 15.0, sg, lam=float(rng.uniform(0, 3)))
    o = np.argsort(Bv); ok &= bool(np.all(np.diff(x[o]) >= -1e-10))
L.check("K_n: contributions ordered by B_i at every sigma and common lambda (2000 draws)", ok)

# ===========================================================================
L.head("2. Any network: matrix form and comparative statics in B_j")
# ===========================================================================
ok = True
for _ in range(120):
    kind = int(rng.integers(0, 4)); m = int(rng.integers(3, 7))
    if kind == 3:
        G = np.zeros((m, m))
        for i in range(m - 1):
            G[i, i + 1] = G[i + 1, i] = 1          # path
    else:
        G = [NL.complete, NL.cycle, NL.star][kind](m)
    d = G.sum(1) + 1
    b = rng.uniform(1.05, min(d.min(), 3) - 0.05); sg = float(rng.uniform(0.3, 3)); lam = float(rng.choice([0.0, 1.0, rng.uniform(0, 3)]))
    Bv = rng.uniform(0.01, 0.06, m)
    x = NL.nash_closed(G, Bv, b, 15.0, sg, lam=lam)
    if x.max() >= 15 - 1e-6:
        continue
    pi = NL.shares(G, Bv, b, sg, lam=lam)
    gam = pi * b / (d - b)
    M = np.linalg.inv(np.eye(m) - gam[:, None] * G)
    for j in range(m):
        h = 1e-5 * Bv[j]
        Bp = Bv.copy(); Bp[j] += h; Bm = Bv.copy(); Bm[j] -= h
        fd = (NL.nash_closed(G, Bp, b, 15.0, sg, lam=lam) - NL.nash_closed(G, Bm, b, 15.0, sg, lam=lam)) / (2 * h)
        an = M[:, j] * x[j] * sg * (1 - pi[j]) / (Bv[j] * (1 - Bv[j]))
        ok &= np.allclose(fd, an, rtol=1e-4, atol=1e-6) and np.all(an > 0)
L.check("d c_i*/d B_j = M_ij c_j* sigma (1-pi_j)/(B_j(1-B_j)) > 0 (finite differences, K/C/S/path, general impact weight)", ok)

# ===========================================================================
L.head("3. Star S_n, heterogeneous shares, general n")
# ===========================================================================
piC, SigP, cC, pij = sp.symbols("pi_C Sigma_pi c_C pi_j", positive=True)
cCs = simp(sp.solve(sp.Eq(cC, piC / (1 - beta / n) * (e + beta / n * (2 * e * SigP / (2 - beta) + beta * SigP / (2 - beta) * cC))), cC)[0])
L.check("c_C* = pi_C e [n(2-beta) + 2 beta Sigma_pi] / [(n-beta)(2-beta) - pi_C beta^2 Sigma_pi]",
        same(cCs, piC * e * (n * (2 - beta) + 2 * beta * SigP) / ((n - beta) * (2 - beta) - piC * beta**2 * SigP)))
L.tex_eq("CES star centre het", r"c_C^\star", cCs)
L.tex_eq("CES star leaf het", r"c_j^\star", pij * (2 * e + beta * sp.Symbol("c_C^\\star")) / (2 - beta))
okS = True
for _ in range(300):
    m = int(rng.integers(3, 9)); b = rng.uniform(1.05, 1.95); sg = float(rng.uniform(0.2, 4)); lam = float(rng.choice([0.0, 1.0, rng.uniform(0, 3)]))
    Bv = rng.uniform(0.005, 0.08, m); G = NL.star(m)
    x = NL.nash_closed(G, Bv, b, 15.0, sg, lam=lam)
    if x.max() >= 15 - 1e-9:
        continue
    pi = NL.shares(G, Bv, b, sg, lam=lam); sp_ = pi[1:].sum()
    cc = pi[0] * 15 * (m * (2 - b) + 2 * b * sp_) / ((m - b) * (2 - b) - pi[0] * b**2 * sp_)
    okS &= abs(cc - x[0]) < 1e-9 and np.allclose(pi[1:] * (30 + b * cc) / (2 - b), x[1:], atol=1e-9)
L.check("general-n CES star formula = numerical equilibrium (300 draws, general impact weight)", okS)
# threshold in shares and its image in B
piP = sp.symbols("pi_P", positive=True)
thr_pi = 2 * piP * (n - beta) / (n * (2 - beta) + (n - 2) * piP * beta)
L.say("  threshold in shares (centre = leaves): pi_C* = 2 pi_P (n-beta)/[n(2-beta) + (n-2) pi_P beta]   (s04 with B -> pi)")
L.say("  image in B: odds(B_C*) = odds(pi_C*)^(1/sigma) * p_C^((sigma-1)/sigma)")
okT = True
for _ in range(200):
    m = int(rng.integers(3, 8)); b = rng.uniform(1.2, 1.95); sg = float(rng.uniform(0.3, 3)); lam = float(rng.choice([0.0, 1.0, rng.uniform(0, 3)]))
    BP = rng.uniform(0.01, 0.06)
    d, th, La, _ = NL.primitives(NL.star(m), b)
    p = th / (1 + lam * (La - 1))
    piPv = 1 / (1 + ((1 - BP) / BP) ** sg * p[1] ** (sg - 1))
    tpi = 2 * piPv * (m - b) / (m * (2 - b) + (m - 2) * piPv * b)
    oB = (tpi / (1 - tpi)) ** (1 / sg) * p[0] ** ((sg - 1) / sg)
    BCs = oB / (1 + oB)
    x = NL.nash_closed(NL.star(m), [BCs] + [BP] * (m - 1), b, 15.0, sg, lam=lam)
    if x.max() < 15 - 1e-9:
        okT &= abs(x[0] - x[1]) < 1e-8
L.check("closed-form CES threshold B_C*(B_P): centre and leaves contribute exactly the same (200 draws, general impact weight)", okT)

# ===========================================================================
L.head("4. Cycle C_n: matrix form; ordering by B only on C_3")
# ===========================================================================
ok = True
for _ in range(400):
    m = int(rng.integers(4, 9)); b = rng.uniform(1.05, 2.9); sg = float(rng.uniform(0.3, 3))
    Bv = rng.uniform(0.01, 0.12, m)
    x = NL.nash_closed(NL.cycle(m), Bv, b, 15.0, sg)
    d = np.full(m, 3.0)
    pi = NL.shares(NL.cycle(m), Bv, b, sg)
    if x.max() < 15 - 1e-9:
        y = np.linalg.solve(np.eye(m) - (pi * b / (3 - b))[:, None] * NL.cycle(m), pi * 15 / (1 - b / 3))
        ok &= np.allclose(x, y, atol=1e-9)
L.check("cycle: (I - Gamma~ A)^{-1} alpha~ = equilibrium (400 draws, n = 4..8)", ok)

# ===========================================================================
L.head("5. Act-based versus impact-based on the star")
# ===========================================================================
for m, b in ((3, 1.5), (5, 1.5)):
    d, th, La, _ = NL.primitives(NL.star(m), b)
    L.say(f"  n={m}, beta={b}: Lambda_C = {La[0]:.3f}, Lambda_L = {La[1]:.3f}; positional odds factor (impact/act) "
          f"= (Lambda_L/Lambda_C)^(sigma-1): sigma=0.5 -> {(La[1]/La[0])**(-0.5):.3f}, sigma=2 -> {(La[1]/La[0])**1:.3f}")
L.check("impact-based motivation amplifies the leaf-centre odds ratio under complements, compresses it under substitutes "
        "(factor (Lambda_L/Lambda_C)^(sigma-1), Lambda_C > Lambda_L)", True if NL.primitives(NL.star(3), 1.5)[2][0] > NL.primitives(NL.star(3), 1.5)[2][1] else False)

# ===========================================================================
L.head("6. Experimental illustration (e=15, n=3, beta=1.5), heterogeneous B")
# ===========================================================================
for Bv in ([0.05, 0.10, 0.15], [0.15, 0.05, 0.10]):
    for sg in (0.5, 1.0, 2.0):
        r = NL.nash_closed(NL.complete(3), Bv, 1.5, 15.0, sg); s_ = NL.nash_closed(NL.star(3), Bv, 1.5, 15.0, sg)
        sa = NL.nash_closed(NL.star(3), Bv, 1.5, 15.0, sg, False)
        L.say(f"  B={Bv}, sigma={sg}: regular {np.round(r, 3).tolist()} | star impact {np.round(s_, 3).tolist()} | star act {np.round(sa, 3).tolist()}")

L.finish()
