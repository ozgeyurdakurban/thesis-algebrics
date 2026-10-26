"""
Step 6 -- The general model: CES with an impact weight lambda_i; altruism;
identification of (sigma, lambda).

Moral component (general form):

    Y_i = c_i + lambda_i * sum_{j in G_i} (beta/d_j) c_i = kappa_i c_i,
    kappa_i = 1 + lambda_i (Lambda_i - 1),    lambda_i >= 0,

lambda_i = 0 : act-based (pure warm glow, Andreoni 1990)
lambda_i = 1 : impact-based (the benchmark of the 5th-report model)

Derived here:
  1. Origin of the form: the c_i-dependent part of the linear utility of
     Anderson, Goeree & Holt (1998), pi_i + alpha sum_{j!=i} pi_j + g c_i, is
     g (1 + (alpha/g)(Lambda_i - 1)) c_i  =>  lambda_i = alpha/g.
  2. CES best response with kappa_i: share pi_i, price p_i = theta_i/kappa_i,
     sigma -> 1 limit (Cobb-Douglas), interiority ceiling on K_n.
  3. lambda-invariance: d log odds / d lambda_i = (sigma-1)(Lambda_i-1)/kappa_i,
     identically zero iff sigma = 1 (the Cobb-Douglas specification is the only
     member of the CES family whose predictions do not depend on lambda).
  4. Comparative statics in lambda: own share and the network effect
     dc_i*/dlambda_j = M~_ij c_j* (sigma-1)(1-pi_j)(Lambda_j-1)/kappa_j.
  5. Price elasticity of giving: eps = -[1 + (sigma-1)(1-pi)]; CD gives -1.
  6. Altruism: MRS between others' benefit E and own payoff X (Cox, Friedman &
     Sadiraj 2008), a = alpha_i (X/Y)^(1/sigma), alpha_i = lambda_i B_i/(1-B_i);
     linear limit = AGH coefficient; sign of d a/d lambda; value at the optimum
     a* = lambda p; MAT ordering conditions.
  7. Decomposition of giving into warm-glow and altruistic parts (OVX 2017).
  8. Identification of (sigma, lambda) from the n = 3 design (population level)
     and of lambda_i from the Star rotation (within subject).
  9. Positional orderings (H3, H4) under general lambda.
"""

import math
import random

import numpy as np
import sympy as sp

import netlib as nl
from common import Lam, Log, B, beta, d, e, rat, same, simp, sigma

L = Log(__file__)
c, S, W, th, lam, E, X, Y = sp.symbols("c S W theta lambda E X Y", positive=True)
alpha_, g_ = sp.symbols("alpha g", positive=True)
rho = 1 - 1 / sigma
kap = 1 + lam * (Lam - 1)


def fl(pt):
    return {k: sp.Float(v, 50) for k, v in pt.items()}


def num(expr, pt):
    return complex(sp.N(expr.subs(fl(pt)), 40)).real


def rel_ok(a_, b_, pt, tol=1e-25):
    """|a - b| / max(1, |b|) evaluated at 50-digit precision."""
    f = fl(pt)
    va, vb = sp.N(a_.subs(f), 50), sp.N(b_.subs(f), 50)
    return abs(complex(va - vb)) / max(1.0, abs(complex(vb))) < tol


def sampler():
    dd = random.randint(2, 9)
    b = rat(1, dd)
    return {d: dd, beta: b, th: 1 - b / dd, e: rat(1, 30), S: rat(0, 30), B: rat(0, 1),
            Lam: rat(1, 6), sigma: rat(sp.Rational(1, 10), 6), W: rat(1, 60), lam: rat(0, 5)}


# ---------------------------------------------------------------------------
L.head("1. Origin of the moral component: the agency-restricted part of AGH (1998)")
# ---------------------------------------------------------------------------
# Material payoffs on an arbitrary network, symbolic in beta; the c_i-dependent part of
# sum_{j != i} X_j is (Lambda_i - 1) c_i.  Checked on S_4 (centre and leaf) and C_5.
for name, G in [("S_4", nl.star(4)), ("C_5", nl.cycle(5)), ("K_4", nl.complete(4))]:
    n_ = G.shape[0]
    cs = sp.symbols(f"c0:{n_}", positive=True)
    dd = [int(G[i].sum()) + 1 for i in range(n_)]
    Xs = [e - cs[i] + beta / dd[i] * sum(cs[k] for k in range(n_) if k == i or G[i, k])
          for i in range(n_)]
    for i in ([0, 1] if name == "S_4" else [0]):
        others = sum(Xs[j] for j in range(n_) if j != i)
        coef = sp.diff(others, cs[i])
        Lam_i = 1 + sum(beta / dd[j] for j in range(n_) if G[i, j])
        L.check(f"{name}, node {i}: d/dc_i sum_(j!=i) X_j = Lambda_i - 1", simp(coef - (Lam_i - 1)) == 0)
        # AGH: U = X_i + alpha sum_{j!=i} X_j + g c_i ; c_i-dependent moral part
        moral = sp.expand(alpha_ * coef * cs[i] + g_ * cs[i])
        target = g_ * (1 + (alpha_ / g_) * (Lam_i - 1)) * cs[i]
        L.check(f"{name}, node {i}: AGH moral part = g (1 + (alpha/g)(Lambda_i-1)) c_i", simp(moral - target) == 0)
L.say("  => lambda = alpha/g: the impact weight is AGH's altruism-to-warm-glow ratio.")

# ---------------------------------------------------------------------------
L.head("2. CES best response with the impact weight")
# ---------------------------------------------------------------------------
Zs = (1 - B) * X**rho + B * Y**rho
foc = (sp.diff(Zs, X) * (-th) + sp.diff(Zs, Y) * kap) / rho
foc_c = foc.subs(Y, kap * c)
ratio = sp.symbols("r", positive=True)
eq = sp.simplify(foc_c.subs(c, ratio * X) / X**(rho - 1))
kt = (B / ((1 - B) * th))**sigma * kap**(sigma - 1)
L.show("FOC in r = c/X (divided by X^(rho-1))", eq)
deq = sp.diff(eq, ratio)
L.check("c/X = (B/((1-B) theta))^sigma kappa^(sigma-1) solves the FOC",
        all(abs(complex(sp.N(eq.subs(ratio, kt).subs(fl(pt)), 50))) < 1e-30 for pt in [sampler() for _ in range(30)]))
L.check("... and is its unique root (the FOC is strictly decreasing in r)",
        all(complex(sp.N(deq.subs(ratio, rat(sp.Rational(1, 100), 50)).subs(fl(pt)), 30)).real < 0
            for pt in [sampler() for _ in range(60)]))
cstar = kt * W / (1 + kt * th)
p = th / kap
pi = 1 / (1 + ((1 - B) / B)**sigma * p**(sigma - 1))
L.check("c = (pi/theta) W with pi = 1/(1+((1-B)/B)^sigma p^(sigma-1)), p = theta/kappa",
        all(rel_ok(cstar, pi * W / th, sampler()) for _ in range(30)))
L.tex_eq("share", r"\pi_i", pi.subs({th: sp.Symbol(r"\theta_i"), B: sp.Symbol("B_i")}))
pi_lim = sp.limit(pi.subs({B: sp.Rational(3, 10), th: sp.Rational(1, 2), lam: sp.Rational(7, 5), Lam: 2}),
                  sigma, 1)
L.check("sigma -> 1: pi -> B (Cobb-Douglas), for any lambda", pi_lim == sp.Rational(3, 10))
L.check("lambda = 1 reproduces the impact-based share (kappa = Lambda)", simp(kap.subs(lam, 1) - Lam) == 0)
L.check("lambda = 0 reproduces the act-based share (kappa = 1)", kap.subs(lam, 0) == 1)

# interiority ceiling on K_n (material condition pi < theta/Lambda)
nn = sp.symbols("n", positive=True)
thK, LamK = 1 - beta / nn, 1 + (nn - 1) * beta / nn
kK = 1 + lam * (LamK - 1)
Bbar = sp.symbols("Bbar", positive=True)
cond = sp.Eq(((Bbar / (1 - Bbar))**sigma * (thK / kK)**(1 - sigma)), (thK / LamK) / (1 - thK / LamK))
odds_ceiling = thK * kK**((1 - sigma) / sigma) * beta**(-1 / sigma)
ok = True
for _ in range(25):
    pt = {nn: random.randint(3, 9), lam: rat(0, 4), sigma: rat(sp.Rational(1, 5), 5)}
    pt[beta] = rat(1, pt[nn])
    lhs = (thK / LamK) / (1 - thK / LamK)
    # plug back: (Bbar/(1-Bbar))^sigma p_K^(1-sigma) == odds of the material ceiling theta/Lambda
    ok &= rel_ok(odds_ceiling**sigma * (thK / kK)**(1 - sigma), lhs, pt) and rel_ok(lhs, thK / beta, pt)
L.check("K_n ceiling: Bbar/(1-Bbar) = theta kappa^((1-sigma)/sigma) beta^(-1/sigma)  (pi < theta/Lambda)", ok)
dlog_ceil = simp(sp.diff(sp.log(odds_ceiling), sigma))
L.check("d log ceiling/d sigma = -log(kappa/beta)/sigma^2", all(rel_ok(dlog_ceil, -sp.log(kK / beta) / sigma**2,
        {nn: random.randint(3, 9), lam: rat(0, 4), sigma: rat(sp.Rational(1, 5), 5), beta: rat(1, 3)}) for _ in range(30)))
thr = sp.solve(sp.Eq(kK, beta), lam)
L.check("ceiling decreasing in sigma iff lambda > n(beta-1)/((n-1)beta); impact-based (lambda=1) always above",
        len(thr) == 1 and simp(thr[0] - nn * (beta - 1) / ((nn - 1) * beta)) == 0 and
        simp(1 - thr[0] - (nn - beta) / ((nn - 1) * beta)) == 0)

# ---------------------------------------------------------------------------
L.head("3. lambda-invariance is specific to sigma = 1")
# ---------------------------------------------------------------------------
lodds = sigma * sp.log(B / (1 - B)) + (1 - sigma) * sp.log(th) + (sigma - 1) * sp.log(kap)
dl = simp(sp.diff(lodds, lam))
L.show("d log odds / d lambda", dl)
L.check("d log odds/d lambda = (sigma-1)(Lambda-1)/kappa", simp(dl - (sigma - 1) * (Lam - 1) / kap) == 0)
L.check("vanishes for all (lambda, Lambda) iff sigma = 1", sp.solve(sp.Eq(dl, 0), sigma) == [1])
dLam = simp(sp.diff(lodds, Lam) * Lam)
L.check("d log odds/d log Lambda = (sigma-1) lambda Lambda/kappa", simp(dLam - (sigma - 1) * lam * Lam / kap) == 0)
kk = sp.Symbol("k", positive=True)
lodds_k = sigma * sp.log(B / (1 - B)) + (1 - sigma) * sp.log(th) + (sigma - 1) * sp.log(kk)
L.check("d log odds/d log kappa = sigma - 1", simp(sp.diff(lodds_k, kk) * kk - (sigma - 1)) == 0)
dsig = simp(sp.diff(lodds, sigma))
L.check("d log odds/d sigma = log(B kappa/((1-B) theta))  =>  threshold B > theta/(theta+kappa)",
        all(rel_ok(dsig, sp.log(B * kap / ((1 - B) * th)), sampler()) for _ in range(40)))

# ---------------------------------------------------------------------------
L.head("4. Comparative statics in the impact weight")
# ---------------------------------------------------------------------------
dpi_dlam = simp(sp.diff(pi, lam))
target = (sigma - 1) * pi * (1 - pi) * (Lam - 1) / kap
L.check("d pi/d lambda = (sigma-1) pi (1-pi) (Lambda-1)/kappa",
        all(abs(num(dpi_dlam - target, sampler())) < 1e-25 for _ in range(30)))
dlogc = (sigma - 1) * (1 - pi) * (Lam - 1) / kap
L.check("d log c/d lambda |_W = (sigma-1)(1-pi)(Lambda-1)/kappa (Hicks-Slutsky, kappa channel)",
        all(abs(num(sp.diff(sp.log(pi * W / th), lam) - dlogc, sampler())) < 1e-25 for _ in range(30)))

# network: finite differences on S_4 and a random connected graph
rng = np.random.default_rng(7)


def net_check(G, Bv, lamv, beta_v, sig, j, h=1e-6):
    n_ = G.shape[0]
    d_, th_, Lam_, _ = nl.primitives(G, beta_v)
    piv = nl.shares(G, Bv, beta_v, sig, lam=lamv)
    gam = piv * beta_v / (d_ - beta_v)
    M = np.linalg.inv(np.eye(n_) - gam[:, None] * G)
    c0 = nl.nash_closed(G, Bv, beta_v, 15.0, sig, lam=lamv)
    if np.any(c0 >= 15 - 1e-9):
        return None
    lp = lamv.copy(); lp[j] += h
    lm = lamv.copy(); lm[j] -= h
    fd = (nl.nash_closed(G, Bv, beta_v, 15.0, sig, lam=lp) - nl.nash_closed(G, Bv, beta_v, 15.0, sig, lam=lm)) / (2 * h)
    kj = 1 + lamv[j] * (Lam_[j] - 1)
    an = M[:, j] * c0[j] * (sig - 1) * (1 - piv[j]) * (Lam_[j] - 1) / kj
    return np.max(np.abs(fd - an))


errs = []
for G in [nl.star(4), nl.cycle(5), nl.complete(3)]:
    for _ in range(40):
        n_ = G.shape[0]
        Bv = rng.uniform(0.02, 0.15, n_)
        lamv = rng.uniform(0.0, 3.0, n_)
        r_ = net_check(G, Bv, lamv, 1.4, float(rng.choice([0.4, 0.8, 1.6, 3.0])), int(rng.integers(n_)))
        if r_ is not None:
            errs.append(r_)
L.say(f"  interior draws checked: {len(errs)}, max |FD - formula| = {max(errs):.2e}")
L.check("dc_i*/dlambda_j = M~_ij c_j* (sigma-1)(1-pi_j)(Lambda_j-1)/kappa_j on S_4, C_5, K_3",
        len(errs) > 50 and max(errs) < 1e-5)

# ---------------------------------------------------------------------------
L.head("5. Price elasticity of giving")
# ---------------------------------------------------------------------------
eps = simp(sp.diff(sp.log(pi * W / th), th) * th)
L.check("eps_(c,theta)|_W = -[1 + (sigma-1)(1-pi)]",
        all(abs(num(eps + 1 + (sigma - 1) * (1 - pi), sampler())) < 1e-25 for _ in range(30)))
L.check("income elasticity d log c/d log W = 1", simp(sp.diff(sp.log(pi * W / th), W) * W - 1) == 0)
L.check("Cobb-Douglas (sigma = 1): eps = -1", sp.simplify((-(1 + (sigma - 1) * (1 - sp.Symbol('q')))).subs(sigma, 1)) == -1)
L.say("  |eps| < 1 iff sigma < 1;  sigma = 1 + (-eps - 1)/(1 - pi).")
for eps_v, pi_v in [(-0.7, 0.02), (-0.7, 0.10), (-1.2, 0.02)]:
    L.say(f"    eps = {eps_v}, pi = {pi_v}:  implied sigma = {1 + (-eps_v - 1) / (1 - pi_v):.3f}")

# ---------------------------------------------------------------------------
L.head("6. Altruism: the willingness to pay for others' benefit (CFS 2008)")
# ---------------------------------------------------------------------------
# Preferences over (X, c, E): U = CES(X, c + lambda E).  a = U_E / U_X.
Zf = (1 - B) * X**rho + B * (c + lam * E)**rho
a = simp(sp.diff(Zf, E) / sp.diff(Zf, X))
a_claim = lam * (B / (1 - B)) * (X / (c + lam * E))**(1 / sigma)
smpA = lambda: {X: rat(1, 30), c: rat(0, 15), E: rat(0, 30), B: rat(0, 1), lam: rat(0, 5),
                sigma: rat(sp.Rational(1, 10), 6)}
L.check("a = lambda (B/(1-B)) (X/Y)^(1/sigma),  Y = c + lambda E",
        all(abs(num(a - a_claim, smpA())) < 1e-25 for _ in range(40)))
L.tex_eq("revealed altruism", r"a_i", a_claim)
alpha_i = lam * B / (1 - B)
L.check("alpha_i = lambda_i B_i/(1-B_i) is the MRS at a balanced bundle X = Y", simp(a_claim.subs(X, c + lam * E) - alpha_i) == 0)
# linear (perfect-substitutes) limit: sigma -> oo
L.check("linear limit (sigma -> oo): a -> lambda B/(1-B) = AGH coefficient alpha (with g = B/(1-B))",
        sp.limit(a_claim.subs({X: 3, c: 2, E: 5, B: sp.Rational(1, 4), lam: sp.Rational(1, 2)}), sigma, sp.oo)
        == sp.Rational(1, 2) * sp.Rational(1, 3))
L.check("a = 0 iff lambda = 0", a_claim.subs(lam, 0) == 0)
L.check("d a/d B > 0 for lambda > 0", all(num(sp.diff(a_claim, B), {**smpA(), lam: rat(sp.Rational(1, 10), 5)}) > 0 for _ in range(200)))
# sign of d a/d lambda at fixed (X, c) on the agent's own budget, E = (Lambda-1) c
a_pos = a_claim.subs(E, (Lam - 1) * c)
dlog_a = simp(sp.diff(sp.log(a_pos), lam) * lam * sigma * kap)
L.show("lambda sigma kappa d log a/d lambda", dlog_a)
L.check("sign(d a/d lambda) = sign(sigma + lambda (Lambda-1)(sigma-1))",
        simp(dlog_a - (sigma + lam * (Lam - 1) * (sigma - 1))) == 0)
L.say("  => a increases with lambda for sigma >= 1; for sigma < 1 iff lambda (Lambda-1) < sigma/(1-sigma).")
# value at the optimum: MRS_(Y,X) = p, so a* = lambda p
mrsYX = simp(sp.diff(Zf, c) / sp.diff(Zf, X))
okopt = True
for _ in range(30):
    pt = sampler()
    cv = num(cstar, pt)
    Xv = num(W - th * cstar, pt)
    # evaluate a at the chosen bundle: X = X*, c = c*, E = (Lambda-1) c*
    a_val = num(a_claim.subs({X: Xv, c: cv, E: (Lam - 1) * cv}), pt)
    okopt &= abs(a_val - num(lam * th / kap, pt)) < 1e-10 * max(1, abs(a_val))
L.check("at the interior optimum a* = lambda p = lambda theta/kappa (B drops out: tangency)", okopt)


def mat(Bp, lp, Bq, lq, sg, grid=4000):
    """Numerical MAT test: a(B',l') >= a(B,l) at every (X, c, E) (ratio depends on t = E/(c+E))."""
    t = np.linspace(0, 1, grid)
    ob = (Bp / (1 - Bp)) / (Bq / (1 - Bq))
    if lq == 0:
        return True
    if lp == 0:
        return False
    # ratio = ob (lp/lq) ((c + lq E)/(c + lp E))^(1/sg), c = 1 - t, E = t
    r_ = ob * (lp / lq) * (((1 - t) + lq * t) / ((1 - t) + lp * t))**(1 / sg)
    return bool(np.all(r_ >= 1 - 1e-12))


def mat_claim(Bp, lp, Bq, lq, sg):
    if lq == 0:
        return True
    if lp == 0:
        return False
    ob = (Bp / (1 - Bp)) / (Bq / (1 - Bq))
    return ob * (lp / lq) >= 1 - 1e-12 and ob * (lp / lq)**(1 - 1 / sg) >= 1 - 1e-12


rr = np.random.default_rng(11)
okm = True
for _ in range(4000):
    Bp, Bq = rr.uniform(0.02, 0.98, 2)
    lp, lq = rr.choice([0.0, 0.3, 0.5, 1.0, 2.0, 4.0], 2)
    sg = float(rr.choice([0.3, 0.7, 1.0, 1.5, 4.0]))
    okm &= mat(Bp, lp, Bq, lq, sg) == mat_claim(Bp, lp, Bq, lq, sg)
L.check("MAT: (B',l') more altruistic than (B,l) iff alpha' >= alpha and odds-ratio(B',B)(l'/l)^(1-1/sigma) >= 1", okm)
L.check("sigma = 1: MAT iff alpha' >= alpha and B' >= B",
        all(mat_claim(Bp, lp, Bq, lq, 1.0) == ((lp * Bp / (1 - Bp) >= lq * Bq / (1 - Bq) - 1e-12) and Bp >= Bq - 1e-12)
            for Bp, Bq in rr.uniform(0.02, 0.98, (500, 2)) for lp, lq in [(0.5, 1.0), (2.0, 1.0), (1.0, 3.0)]))

# ---------------------------------------------------------------------------
L.head("7. Decomposition of giving (OVX 2017 analogue)")
# ---------------------------------------------------------------------------
omega = lam * (Lam - 1) / kap
L.check("omega = lambda(Lambda-1)/kappa in [0,1), = 0 iff lambda = 0", omega.subs(lam, 0) == 0 and
        all(0 <= num(omega, sampler()) < 1 for _ in range(100)))
L.check("warm-glow + altruistic parts of Y add up: (1-omega) Y + omega Y = c + lambda (Lambda-1) c",
        simp((1 - omega) * kap * c + omega * kap * c - (c + lam * (Lam - 1) * c)) == 0 and
        simp((1 - omega) * kap * c - c) == 0)
for nm, Lv in [("regular K_3", 2), ("centre S_3", sp.Rational(5, 2)), ("leaf S_3", sp.Rational(3, 2))]:
    L.say(f"    lambda = 1, {nm}: omega = {sp.nsimplify(omega.subs({lam: 1, Lam: Lv}))}")

# ---------------------------------------------------------------------------
L.head("8. Identification of (sigma, lambda) from the n = 3 design")
# ---------------------------------------------------------------------------
# positions: K (regular, K_3), C (centre, S_3), L (leaf, S_3)
thK3, thC3, thL3 = 1 - beta / 3, 1 - beta / 3, 1 - beta / 2
LK, LC, LL = 1 + 2 * beta / 3, 1 + beta, 1 + beta / 3
kK3, kC3, kL3 = [1 + lam * (x_ - 1) for x_ in (LK, LC, LL)]
g1 = sp.log(kC3 / kK3)                      # D1 = logodds_C - logodds_K = (sigma-1) g1
g2 = sp.log(thK3 * kL3 / (thL3 * kK3))      # D2 = logodds_L - logodds_K = (sigma-1) g2
dg1 = simp(sp.diff(g1, lam))
dg2 = simp(sp.diff(g2, lam))
L.show("dg1/dlambda", dg1)
L.show("dg2/dlambda", dg2)
smpb = lambda: {beta: rat(sp.Rational(6, 5), 2), lam: rat(0, 20)}
L.check("g1(0) = 0 and g1 strictly increasing in lambda", g1.subs(lam, 0) == 0 and
        all(num(dg1, smpb()) > 0 for _ in range(300)))
L.check("g2 strictly decreasing in lambda, g2(0) = log(theta_K/theta_L) > 0",
        all(num(dg2, smpb()) < 0 for _ in range(300)) and all(num(g2.subs(lam, 0), smpb()) > 0 for _ in range(50)))
L.say("  => lambda -> (g1, g2) has a strictly monotone direction: (D1, D2) = (sigma-1)(g1, g2)")
L.say("     identifies sigma and lambda whenever sigma != 1; at sigma = 1, D = 0 and lambda is not identified.")


def recover(D1, D2, beta_v):
    """Invert (D1, D2) = (s-1)(g1(l), g2(l)); returns (s, l)."""
    if abs(D1) < 1e-14 and abs(D2) < 1e-14:
        return 1.0, float("nan")
    G1 = lambda l: math.log((1 + l * beta_v) / (1 + l * 2 * beta_v / 3))
    G2 = lambda l: math.log((1 - beta_v / 3) * (1 + l * beta_v / 3) / ((1 - beta_v / 2) * (1 + l * 2 * beta_v / 3)))
    sgn = 1.0 if (D1 > 0 or (abs(D1) < 1e-14 and D2 > 0)) else -1.0
    target = math.atan2(sgn * D1, sgn * D2)
    ang = lambda l: math.atan2(G1(l), G2(l))
    if abs(D1) < 1e-14:
        lh = 0.0
    else:
        lo, hi = 0.0, 1.0
        while ang(hi) < target and hi < 1e8:
            hi *= 2
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if ang(mid) < target else (lo, mid)
        lh = (lo + hi) / 2
    sh = 1 + (D2 / G2(lh) if abs(G2(lh)) > abs(G1(lh)) else D1 / G1(lh))
    return sh, lh


okid = True
worst = 0.0
for s_v, l_v, b_v in [(0.5, 1.0, 1.5), (2.0, 0.4, 1.5), (1.5, 3.0, 1.5), (0.7, 2.5, 1.3), (3.0, 0.0, 1.8),
                      (0.3, 6.0, 1.25), (1.2, 1.0, 1.9)]:
    for B_v in (0.05, 0.2):
        def lo_(thv, kv):
            return s_v * math.log(B_v / (1 - B_v)) + (1 - s_v) * math.log(thv) + (s_v - 1) * math.log(kv)
        kv = lambda Lv: 1 + l_v * (Lv - 1)
        lK = lo_(1 - b_v / 3, kv(1 + 2 * b_v / 3))
        lC = lo_(1 - b_v / 3, kv(1 + b_v))
        lL = lo_(1 - b_v / 2, kv(1 + b_v / 3))
        sh, lh = recover(lC - lK, lL - lK, b_v)
        err = max(abs(sh - s_v), abs(lh - l_v))
        worst = max(worst, err)
        okid &= err < 1e-8
L.check(f"round trip (sigma, lambda) -> log-odds -> (sigma, lambda): max error {worst:.1e}", okid)

# Formula-free: shares recovered from equilibrium contributions computed by utility maximisation
okff = True
for s_v, l_v in [(0.6, 1.0), (2.0, 0.5), (1.5, 2.0)]:
    B_v = 0.06
    cK = nl.nash_utility(nl.complete(3), [B_v] * 3, 1.5, 15.0, s_v, lam=l_v)
    cS = nl.nash_utility(nl.star(3), [B_v] * 3, 1.5, 15.0, s_v, lam=l_v)
    if np.any(cK > 15 - 1e-6) or np.any(cS > 15 - 1e-6):
        okff = False
        continue

    def lodd(cv, G, i):
        dd_, thv, _, _ = nl.primitives(G, 1.5)
        Wv = 15.0 + 1.5 / dd_[i] * (G[i] @ cv)
        piv = thv[i] * cv[i] / Wv
        return math.log(piv / (1 - piv))
    lK_, lC_, lL_ = lodd(cK, nl.complete(3), 0), lodd(cS, nl.star(3), 0), lodd(cS, nl.star(3), 1)
    sh, lh = recover(lC_ - lK_, lL_ - lK_, 1.5)
    L.say(f"    true (sigma, lambda) = ({s_v}, {l_v})  recovered from formula-free equilibria: ({sh:.6f}, {lh:.6f})")
    okff &= abs(sh - s_v) < 1e-5 and abs(lh - l_v) < 1e-4
L.check("identification from formula-free equilibria (utility maximisation on K_3 and S_3)", okff)

# testable restriction of the reference specification: (g1, g2) != (0, 0) for every lambda >= 0,
# so the three giving shares coincide iff sigma = 1
gn = [num(g1**2 + g2**2, {beta: rat(sp.Rational(6, 5), 2), lam: rat(0, 50)}) for _ in range(400)] + \
     [num((g1**2 + g2**2).subs(lam, 0), {beta: rat(sp.Rational(6, 5), 2)}) for _ in range(50)]
L.check("(g1, g2) != (0, 0) for all lambda >= 0: equal giving shares across K, C, L  iff  sigma = 1", min(gn) > 1e-6)
# the dictator decision (cost 1, impact 1): Cobb-Douglas share donated = B for every lambda
g_, Ebud = sp.symbols("g E_bud", positive=True)
foc_d = sp.diff((1 - B) * sp.log(Ebud - g_) + B * sp.log((1 + lam) * g_), g_)
L.check("dictator, sigma = 1: share donated g/E = B for every impact weight", simp(sp.solve(foc_d, g_)[0] / Ebud - B) == 0)

# within-subject (Star rotation): h(lambda) = log(p_C/p_L) strictly decreasing
h = sp.log(thC3 * kL3 / (thL3 * kC3))
dh = simp(sp.diff(h, lam))
L.check("h(lambda) = log(p_C/p_L) strictly decreasing => lambda_i identified from one rotation given sigma != 1",
        all(num(dh, smpb()) < 0 for _ in range(300)))
L.check("experimental beta = 3/2: h(lambda) = 0 at lambda = 2 (p_C = p_L)",
        sp.solve(sp.Eq(h.subs(beta, sp.Rational(3, 2)), 0), lam) == [2])

# ---------------------------------------------------------------------------
L.head("9. Positional orderings under a general impact weight")
# ---------------------------------------------------------------------------
lams = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
sigs = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0]
Bs = np.linspace(0.005, 0.3, 60)


def star_hom(nv, bv, sg, lv, Bv):
    G = nl.star(nv)
    pi_ = nl.shares(G, [Bv] * nv, bv, sg, lam=lv)
    pC, pL = pi_[0], pi_[1]
    D = (nv - bv) * (2 - bv) - (nv - 1) * pC * pL * bv**2
    cC = pC * 15 * (nv * (2 - bv) + 2 * (nv - 1) * pL * bv) / D
    cL = pL * 15 * (2 * (nv - bv) + nv * pC * bv) / D
    return D, cC, cL


def k_hom(nv, bv, sg, lv, Bv):
    G = nl.complete(nv)
    pK = nl.shares(G, [Bv] * nv, bv, sg, lam=lv)[0]
    den = nv - bv - pK * bv * (nv - 1)
    return den, pK * nv * 15 / den


res_i, res_ii, res_iii = True, {}, {}
for lv in lams:
    viol_LC3, viol_CK3 = 0, []
    rev_n4 = []
    for nv in range(3, 9):
        for bv in np.linspace(1.25 if nv == 3 else 1.05, 1.95, 10):
            if nv == 3 and bv < 1.2:
                continue
            for sg in sigs:
                for Bv in Bs:
                    D, cC, cL = star_hom(nv, bv, sg, lv, Bv)
                    if not (D > 0 and 0 < cC < 15 and 0 < cL < 15):
                        continue
                    if nv == 3:
                        if cL <= cC:
                            viol_LC3 += 1
                        dK, cK = k_hom(3, bv, sg, lv, Bv)
                        if dK > 0 and 0 < cK < 15 and cC <= cK:
                            viol_CK3.append(sg)
                    elif cL <= cC:
                        rev_n4.append(sg)
    res_i &= viol_LC3 == 0
    res_ii[lv] = (min(viol_CK3) if viol_CK3 else None, max(viol_CK3) if viol_CK3 else None)
    res_iii[lv] = min(rev_n4) if rev_n4 else None
    L.say(f"  lambda = {lv}: n=3 leaf<=centre cases: {viol_LC3};  n=3 centre<=regular at sigma in "
          f"{res_ii[lv]};  n>=4 leaf<=centre at sigma >= {res_iii[lv]}")
L.say("  (the n = 3 leaf-centre reversals at lambda >= 2 are characterised exactly below)")
L.check("(ii) lambda = 0: centre > regular at every sigma; lambda > 0: violations only for sigma < 1",
        res_ii[0.0] == (None, None) and all(v[1] is None or v[1] < 1 for k, v in res_ii.items()))
L.check("(iii) n >= 4 leaf-centre reversal: never at lambda = 0; only at sigma > 1 when lambda > 0",
        res_iii[0.0] is None and all(v is None or v > 1 for v in res_iii.values()))
# limit behind (ii)
okl = True
for lv in [0.25, 1.0, 4.0]:
    for sg in [0.3, 0.7]:
        Bv = 1e-40
        _, cC, _ = star_hom(3, 1.5, sg, lv, Bv)
        _, cK = k_hom(3, 1.5, sg, lv, Bv)
        lim = ((1 + lv * 1.0) / (1 + lv * 1.5))**(1 - sg)    # (kappa_K/kappa_C)^(1-sigma)
        okl &= abs(cC / cK - lim) < 1e-4
L.check("B -> 0: c_C/c_K -> (kappa_K/kappa_C)^(1-sigma) < 1 for lambda > 0, sigma < 1", okl)

# ---- exact leaf-centre comparison on S_n (any sigma, lambda, heterogeneous B) -------------
# Derived: with shares pi_C, pi_L on the interior, c_L > c_C  <=>  odds(pi_L)/odds(pi_C) > theta_L/theta_C.
nS, pC_, pL_ = sp.symbols("n_S pi_C pi_L", positive=True)
lhs_ = pL_ * (2 * (nS - beta) + nS * pC_ * beta) - pC_ * (nS * (2 - beta) + 2 * (nS - 1) * pL_ * beta)
thC_, thL_ = 1 - beta / nS, 1 - beta / 2
oC, oL = pC_ / (1 - pC_), pL_ / (1 - pL_)
rhs_ = (oL / oC - thL_ / thC_)
# lhs_ / (pi_C pi_L) = 2(n-beta)/o_C - n(2-beta)/o_L, a positive multiple of (o_L/o_C - theta_L/theta_C)
ident = simp(lhs_ / (pC_ * pL_) - (2 * (nS - beta) / oC - nS * (2 - beta) / oL))
L.check("c_L - c_C has the sign of 2(n-beta)/o_C - n(2-beta)/o_L (identity)", ident == 0)
L.check("... = (n(2-beta)/o_C) (o_L/o_C)^(-1)... i.e. c_L > c_C iff odds(pi_L)/odds(pi_C) > theta_L/theta_C",
        simp((2 * (nS - beta) / oC - nS * (2 - beta) / oL) - (nS * (2 - beta) / oL) * (oL / oC) * (2 * (nS - beta) / (nS * (2 - beta))) + nS * (2 - beta) / oL) == 0
        and simp(thL_ / thC_ - nS * (2 - beta) / (2 * (nS - beta))) == 0)
# common B: odds ratio = (p_L/p_C)^(1-sigma); threshold impact weight lbar_n(beta) at which p_C = p_L
LC_, LL_ = 1 + (nS - 1) * beta / 2, 1 + beta / nS
lbar = sp.solve(sp.Eq(thC_ * (1 + lam * (LL_ - 1)), thL_ * (1 + lam * (LC_ - 1))), lam)
L.show("lbar_n(beta) (p_C = p_L)", [simp(v) for v in lbar])
L.check("n = 3: lbar = 3/(12 - 7 beta); at beta = 3/2, lbar = 2",
        len(lbar) == 1 and simp(lbar[0].subs(nS, 3) - 3 / (12 - 7 * beta)) == 0 and
        simp(lbar[0].subs({nS: 3, beta: sp.Rational(3, 2)})) == 2)
L.tex_eq("lbar_n", r"\bar\lambda_n(\beta)", simp(lbar[0]))
L.check("n = 3: lbar_3(9/7) = 1 (impact-based spec inside the robust region iff beta >= 9/7)",
        simp((3 / (12 - 7 * beta)).subs(beta, sp.Rational(9, 7))) == 1)
b12 = 1.2
pC12, pL12 = (1 - b12 / 3) / (1 + b12), (1 - b12 / 2) / (1 + b12 / 3)
sst12 = 1 + math.log((1 - b12 / 3) / (1 - b12 / 2)) / math.log(pL12 / pC12)
L.say(f"  impact-based, n = 3, beta = 1.2: sigma* = {sst12:.3f}")
L.check("impact-based, n = 3, beta = 1.2: leaf-centre reversal threshold sigma* ~ 9.7", abs(sst12 - 9.72) < 0.01)
# vectorised verification of the exact condition and of the sigma* threshold
np.seterr(all="ignore")
okx, cnt, okss = True, 0, True
lvg = np.linspace(0, 8, 81)[:, None, None]
sgg = np.concatenate([np.linspace(0.05, 1, 20), np.linspace(1.05, 40, 120)])[None, :, None]
Bg = np.geomspace(1e-8, 0.5, 60)[None, None, :]
for nv in range(3, 9):
    for bv in np.linspace(1.05 if nv > 3 else 1.2, 1.99, 20):
        thC, thL = 1 - bv / nv, 1 - bv / 2
        kC, kL = 1 + lvg * (nv - 1) * bv / 2, 1 + lvg * bv / nv
        shr = lambda t, k: 1 / (1 + ((1 - Bg) / Bg)**sgg * (t / k)**(sgg - 1))
        pC, pL = shr(thC, kC), shr(thL, kL)
        D = (nv - bv) * (2 - bv) - (nv - 1) * pC * pL * bv**2
        cC = pC * 15 * (nv * (2 - bv) + 2 * (nv - 1) * pL * bv) / D
        cL = pL * 15 * (2 * (nv - bv) + nv * pC * bv) / D
        ok_ = (D > 0) & (cC > 0) & (cC < 15) & (cL > 0) & (cL < 15)
        pred = ((thL / kL) / (thC / kC))**(1 - sgg) > thL / thC
        okx &= bool(np.all((cL > cC)[ok_] == np.broadcast_to(pred, ok_.shape)[ok_]))
        cnt += int(ok_.sum())
        # sigma* form: reversal iff p_C < p_L and sigma > 1 + log(thC/thL)/log(p_L/p_C)
        pCm, pLm = thC / kC, thL / kL
        with np.errstate(divide="ignore", invalid="ignore"):
            sst = np.where(pCm < pLm, 1 + np.log(thC / thL) / np.log(pLm / pCm), np.inf)
        rev_pred = np.broadcast_to(sgg > sst, ok_.shape)
        okss &= bool(np.all((cL <= cC)[ok_] == rev_pred[ok_]))
L.say(f"  interior star equilibria checked: {cnt}")
L.check("S_n, common B: c_L > c_C  iff  (p_L/p_C)^(1-sigma) > theta_L/theta_C  (exact)", okx)
L.check("reversal iff lambda > lbar_n(beta) and sigma > sigma* = 1 + log(theta_C/theta_L)/log(p_L/p_C)", okss)
# heterogeneous weights: odds form reproduces the Cobb-Douglas threshold B_C*(B_P)
BP, BC = sp.symbols("B_P B_C", positive=True)
Bstar = sp.solve(sp.Eq((BC / (1 - BC)) / (BP / (1 - BP)), thC_ / thL_), BC)
L.check("sigma = 1: odds condition gives B_C* = 2B_P(n-beta)/[n(2-beta)+(n-2)B_P beta] (Prop. star_interaction)",
        len(Bstar) == 1 and simp(Bstar[0] - 2 * BP * (nS - beta) / (nS * (2 - beta) + (nS - 2) * BP * beta)) == 0)
# leaf vs regular member (H4, first link) at n = 3 under general lambda.
# p_L <= p_K  iff  lambda <= lt(beta) = 3/(2(3 - 2 beta))  (infinite for beta >= 3/2)
thK3_, thL3_ = 1 - beta / 3, 1 - beta / 2
lt = sp.solve(sp.Eq(thL3_ * (1 + lam * 2 * beta / 3), thK3_ * (1 + lam * beta / 3)), lam)
L.check("n = 3: p_L = p_K at lambda = 3/(2(3-2 beta)); p_L < p_K for every lambda when beta >= 3/2",
        len(lt) == 1 and simp(lt[0] - sp.Rational(3, 2) / (3 - 2 * beta)) == 0)
okLK, okLK2 = True, True
for lv in lams:
    for bv in np.linspace(1.2, 1.99, 15):
        for sg in sigs + [12.0, 20.0]:
            for Bv in Bs:
                D, cC, cL = star_hom(3, bv, sg, lv, Bv)
                dK, cK = k_hom(3, bv, sg, lv, Bv)
                if D > 0 and 0 < cL < 15 and 0 < cC < 15 and dK > 0 and 0 < cK < 15 and not cL > cK:
                    okLK &= lv > 1.5 / (3 - 2 * bv)      # violations only above lt(beta)
                    okLK2 &= sg > 1                      # ... and only under gross substitutes
L.check("n = 3: leaf > regular member whenever lambda <= 3/(2(3-2 beta)) (always at beta >= 3/2)", okLK)
L.check("n = 3: leaf <= regular occurs only for sigma > 1", okLK2)

# experimental prices by lambda (table)
L.say("\n  prices of the moral good p = theta/kappa at e=15, n=3, beta=1.5:")
for lv in [0.0, 0.5, 1.0, 2.0]:
    pr = [(0.5 / (1 + lv * 1.0)), (0.5 / (1 + lv * 1.5)), (0.25 / (1 + lv * 0.5))]
    L.say(f"    lambda = {lv}: regular {pr[0]:.4f}, centre {pr[1]:.4f}, leaf {pr[2]:.4f}")

L.finish()
