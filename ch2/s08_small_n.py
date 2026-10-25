"""
Step 4 -- Explicit small-N solutions (Appendix) and their cross-validation.

For every network used in the appendix -- K_2, K_3, K_4, C_4, S_3, S_4 (and the
identities S_2 = K_2, C_3 = K_3) -- and for every specification:

  (a) CD, common B          (b) CD, heterogeneous B_i
  (c) CES, common B         (d) CES, heterogeneous B_i   (impact- and act-based, and a
                                                         heterogeneous impact weight lambda_i)

three independent routes are compared:
  1. DIRECT    : the first-order conditions of the actual utilities of the N agents are
                 written out from the adjacency matrix and the system is solved for this N
                 (symbolically for CD; in the share form for CES);
  2. GENERAL N : the general-N closed form of the chapter, specialised to this N
                 (K_n: e phi_i/mu; star: centre/leaf formula; cycle: (I - Gamma G)^-1 alpha,
                 C_4: sub-lattice form);
  3. NUMERICAL : formula-free Nash equilibrium obtained by maximising each agent's actual
                 utility function on [0, e] (netlib.nash_utility).
The explicit formulas are exported to ch2/tex/s08_small_n.tex for the appendix.

Thesis numbering (version of 25 Oct 2026): Appendix A.
"""

import random

import numpy as np
import sympy as sp

from common import Log, beta, e, rat, simp
import netlib as NL

L = Log(__file__)
rng = np.random.default_rng(2026)


def graph(kind, m):
    return {"K": NL.complete, "C": NL.cycle, "S": NL.star}[kind](m)


def direct_cd(Gm, Bs):
    """Solve the CD first-order conditions written from the utility functions."""
    m = Gm.shape[0]
    cs = sp.symbols(f"c1:{m+1}", positive=True)
    dd = [int(Gm[i].sum()) + 1 for i in range(m)]
    eqs = []
    for i in range(m):
        X = e - cs[i] + beta / dd[i] * (cs[i] + sum(cs[j] for j in range(m) if Gm[i, j]))
        Lam = 1 + sum(beta / dd[j] for j in range(m) if Gm[i, j])
        logU = (1 - Bs[i]) * sp.log(X) + Bs[i] * sp.log(Lam * cs[i])
        eqs.append(sp.numer(sp.together(sp.diff(logU, cs[i]))))
    sol = sp.solve(eqs, cs, dict=True)
    sol = [s for s in sol if all(v != 0 for v in s.values())]
    return [simp(sol[0][c]) for c in cs]


def direct_share(Gm, ps):
    """Solve the system c_i = (pi_i/theta_i)(e + (beta/d_i) S_i) for symbolic shares pi_i."""
    m = Gm.shape[0]
    cs = sp.symbols(f"c1:{m+1}")
    dd = [int(Gm[i].sum()) + 1 for i in range(m)]
    eqs = [sp.Eq(cs[i], ps[i] / (1 - beta / dd[i]) * (e + beta / dd[i] * sum(cs[j] for j in range(m) if Gm[i, j])))
           for i in range(m)]
    sol = sp.solve(eqs, cs, dict=True)[0]
    return [simp(sol[c]) for c in cs]


def general(kind, m, ps):
    """General-N closed forms of the chapter, in share form, specialised to N = m."""
    if kind == "K":
        ph = [m * p / (m - (1 - p) * beta) for p in ps]
        mu = 1 - beta / m * sum(ph)
        return [e * f / mu for f in ph]
    if kind == "S":
        pC, SL = ps[0], sum(ps[1:])
        cC = pC * e * (m * (2 - beta) + 2 * beta * SL) / ((m - beta) * (2 - beta) - pC * beta**2 * SL)
        return [cC] + [p * (2 * e + beta * cC) / (2 - beta) for p in ps[1:]]
    if kind == "C" and m == 4:
        al = [3 * p * e / (3 - beta) for p in ps]
        ga = [p * beta / (3 - beta) for p in ps]
        Gs, Hs, As, Bs_ = ga[0] + ga[2], ga[1] + ga[3], al[0] + al[2], al[1] + al[3]
        S13 = (As + Gs * Bs_) / (1 - Gs * Hs)
        S24 = (Bs_ + Hs * As) / (1 - Gs * Hs)
        return [al[0] + ga[0] * S24, al[1] + ga[1] * S13, al[2] + ga[2] * S24, al[3] + ga[3] * S13]
    # general network: matrix form
    Gm = graph(kind, m)
    dd = [int(Gm[i].sum()) + 1 for i in range(m)]
    alpha = sp.Matrix([ps[i] * e / (1 - beta / dd[i]) for i in range(m)])
    Gam = sp.diag(*[ps[i] * beta / (dd[i] - beta) for i in range(m)])
    return list((sp.eye(m) - Gam * sp.Matrix(Gm.astype(int).tolist())).inv() * alpha)


def close(a, b, pt):
    return abs(sp.N((a - b).subs(pt), 40)) < 1e-25


CASES = [("K", 2), ("K", 3), ("K", 4), ("C", 4), ("S", 3), ("S", 4)]

# ===========================================================================
L.head("1. Cobb-Douglas: direct utility FOCs vs general-N closed forms")
# ===========================================================================
for kind, m in CASES:
    Gm = graph(kind, m)
    Bs = sp.symbols(f"B1:{m+1}", positive=True)
    bmax = min(int(Gm.sum(1).min()) + 1, 3)
    dirc = direct_cd(Gm, Bs)
    gen = general(kind, m, list(Bs))
    ok = True
    for _ in range(6):
        pt = {b: rat(0, sp.Rational(1, 6)) for b in Bs}; pt.update({beta: rat(1, bmax), e: 15})
        ok &= all(close(dirc[i], gen[i], pt) for i in range(m))
    L.check(f"CD heterogeneous {kind}_{m}: direct FOC solution = general-N formula", ok)
    # homogeneous
    Bh = sp.Symbol("B", positive=True)
    hom = [simp(x.subs({b: Bh for b in Bs})) for x in dirc]
    L.say(f"   {kind}_{m}, common B: " + "; ".join(sorted(set(sp.sstr(h) for h in hom))))
    for i, x in enumerate(hom[: (2 if kind == "S" else 1)]):
        lab = {0: "centre", 1: "leaf"}[i] if kind == "S" else "member"
        L.tex_eq(f"CD hom {kind}_{m} {lab}", rf"c^\star_{{{kind}_{m},\mathrm{{{lab}}}}}", hom[i])
    for i in range(m if kind != "S" else 2):
        L.tex_eq(f"CD het {kind}_{m} agent {i+1}", rf"c^\star_{{{i+1}}}\;({kind}_{m})", dirc[i])

# compact explicit forms used in the appendix
B4 = sp.symbols("B1:5", positive=True)
for m in (2, 3, 4):
    Bs = B4[:m]
    q = [m - (1 - b) * beta for b in Bs]
    den = sp.prod(q) - beta * sum(Bs[j] * sp.prod([q[k] for k in range(m) if k != j]) for j in range(m))
    form = [m * e * Bs[i] * sp.prod([q[k] for k in range(m) if k != i]) / den for i in range(m)]
    dirc = direct_cd(graph("K", m), Bs)
    L.check(f"K_{m}: c_i* = m e B_i prod_(k!=i) q_k / [prod_k q_k - beta sum_j B_j prod_(k!=j) q_k],  q_k = m - (1-B_k) beta",
            all(sp.simplify(dirc[i] - form[i]) == 0 for i in range(m)))
dirc = direct_cd(graph("C", 4), B4)
D4 = (3 - beta) ** 2 - beta**2 * (B4[0] + B4[2]) * (B4[1] + B4[3])
form = [3 * e * B4[0] * (3 - beta + (B4[1] + B4[3]) * beta) / D4, 3 * e * B4[1] * (3 - beta + (B4[0] + B4[2]) * beta) / D4,
        3 * e * B4[2] * (3 - beta + (B4[1] + B4[3]) * beta) / D4, 3 * e * B4[3] * (3 - beta + (B4[0] + B4[2]) * beta) / D4]
L.check("C_4: c_i* = 3 e B_i [3 - beta + beta (sum of i's neighbours' B)] / [(3-beta)^2 - beta^2 (B_1+B_3)(B_2+B_4)]",
        all(sp.simplify(dirc[i] - form[i]) == 0 for i in range(4)))
for m in (3, 4):
    Bs = B4[:m]; SL = sum(Bs[1:])
    Dm = (m - beta) * (2 - beta) - Bs[0] * beta**2 * SL
    form = [Bs[0] * e * (m * (2 - beta) + 2 * beta * SL) / Dm] + [b * e * (2 * (m - beta) + m * Bs[0] * beta) / Dm for b in Bs[1:]]
    dirc = direct_cd(graph("S", m), Bs)
    L.check(f"S_{m}: centre B_C e[n(2-beta) + 2 beta Sigma_L]/D, leaf B_j e[2(n-beta) + n B_C beta]/D",
            all(sp.simplify(dirc[i] - form[i]) == 0 for i in range(m)))

# identities between topologies
Bh = sp.Symbol("B", positive=True)
s2 = general("S", 2, [Bh, Bh]); k2 = general("K", 2, [Bh, Bh])
L.check("S_2 = K_2: star formula at n = 2 equals the complete-network formula", sp.simplify(s2[0] - k2[0]) == 0)
B3 = sp.symbols("B1:4", positive=True)
c3 = direct_share(NL.cycle(3), list(B3)); k3 = general("K", 3, list(B3))
L.check("C_3 = K_3: cycle system at n = 3 equals the complete-network formula (heterogeneous)",
        all(sp.simplify(c3[i] - k3[i]) == 0 for i in range(3)))

# ===========================================================================
L.head("2. CES: share-form system vs general-N closed forms (heterogeneous shares)")
# ===========================================================================
for kind, m in CASES:
    Gm = graph(kind, m)
    ps = sp.symbols(f"pi1:{m+1}", positive=True)
    dirc = direct_share(Gm, list(ps))
    gen = general(kind, m, list(ps))
    bmax = min(int(Gm.sum(1).min()) + 1, 3)
    ok = True
    for _ in range(6):
        pt = {p: rat(0, sp.Rational(1, 6)) for p in ps}; pt.update({beta: rat(1, bmax), e: 15})
        ok &= all(close(dirc[i], gen[i], pt) for i in range(m))
    L.check(f"CES heterogeneous {kind}_{m}: share-form system = general-N formula", ok)
    for i in range(m if kind != "S" else 2):
        L.tex_eq(f"CES het {kind}_{m} agent {i+1}", rf"c^\star_{{{i+1}}}\;({kind}_{m},\ \mathrm{{CES}})", dirc[i])

# ===========================================================================
L.head("3. Formula-free numerical check (maximise the actual utilities)")
# ===========================================================================
worst = {}
n_corner = [0]
for kind, m in CASES:
    Gm = graph(kind, m)
    bmax = min(Gm.sum(1).min() + 1, 3)
    for spec in ("CD", "CES-impact", "CES-act", "CES-lambda_i"):
        for hom in (True, False):
            for _ in range(2):
                b = rng.uniform(1.05, bmax - 0.05)
                sg = 1.0 if spec == "CD" else float(rng.choice([0.4, 0.7, 1.5, 2.5]))
                Bv = np.full(m, rng.uniform(0.02, 0.15)) if hom else rng.uniform(0.02, 0.15, m)
                # impact weight: 1 (impact-based), 0 (act-based), or heterogeneous lambda_i ~ U(0, 3)
                lam = {"CD": 1.0, "CES-impact": 1.0, "CES-act": 0.0}.get(spec)
                if lam is None:
                    lam = rng.uniform(0, 3, m)
                # closed form through the general-N formula, numerically
                pi = NL.shares(Gm, Bv, b, sg, lam=lam)
                gen = general(kind, m, [sp.Float(x, 30) for x in pi])
                gen = np.array([float(sp.N(g.subs({beta: b, e: 15}))) for g in gen])
                if not (gen.min() > 0 and gen.max() < 15):   # outside the interior region the interior
                    gen = NL.nash_closed(Gm, Bv, b, 15.0, sg, lam=lam)   # formula does not apply: use the
                    n_corner[0] += 1                                 # closed form on the corner set
                num = NL.nash_utility(Gm, Bv, b, 15.0, sg, lam=lam)
                key = f"{kind}_{m} {spec}"
                worst[key] = max(worst.get(key, 0), float(np.max(abs(gen - num))))
L.say(f"   ({n_corner[0]} of the draws had agents at the corner; compared through the active-set closed form)")
for k, v in worst.items():
    L.say(f"   {k:16s}: max |general-N formula - utility maximisation| = {v:.1e}")
L.check("all small-N cases: general-N formula = formula-free equilibrium (CD; CES with lambda = 1, 0 and heterogeneous lambda_i; common and heterogeneous B)",
        max(worst.values()) < 1e-5)

L.finish()
