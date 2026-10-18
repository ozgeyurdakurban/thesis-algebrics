"""
Step 3b -- CES equilibria under a COMMON prosocial weight B on K_n, C_n, S_n (general n).

By s05, the CES best response is c_i = (pi_i/theta_i) W_i with the share
    pi_i = 1/(1 + ((1-B)/B)^sigma p_i^(sigma-1)),
    p_i = theta_i/Lambda_i (impact-based) or theta_i (act-based).
With a common B the share is common on regular graphs but POSITION-SPECIFIC on the
star (p_C != p_L), so the CES star with a common weight is a star with heterogeneous
shares.

Derived here:
  K_n, C_n : equilibrium in pi; interiority pi < theta/Lambda (full income at the full-
             contribution profile equals Lambda e); closed-form ceilings in B for both
             specifications; their sigma-monotonicity; sigma comparative statics.
  S_n      : centre/leaf equilibrium with shares (pi_C, pi_L); leaf-vs-centre condition;
             positional ordering over a grid; mixed-regime centre ceiling (closed form);
             numerical leaf ceiling.
  Checks   : formula-free Nash computation (maximising the actual CES utility).
"""

import numpy as np
import sympy as sp

from common import Log, beta, e, n, rat, same, simp, sigma
import netlib as NL

L = Log(__file__)
c, pi_, piC, piL, cC, cL, B = sp.symbols("c pi pi_C pi_L c_C c_L B", positive=True)
rng = np.random.default_rng(7)


def share(Bv, pv, sg):
    return 1 / (1 + ((1 - Bv) / Bv) ** sg * pv ** (sg - 1))


# ===========================================================================
L.head("1. K_n with a common share pi (general n)")
# ===========================================================================
thK, LamK = 1 - beta / n, 1 + (n - 1) * beta / n
cK = simp(sp.solve(sp.Eq(c, pi_ / thK * (e + beta / n * (n - 1) * c)), c)[0])
L.show("c*_K(pi)", cK)
L.check("c*_K = pi n e/(n - beta - pi beta (n-1))  (= CD formula with B -> pi)", same(cK, pi_ * n * e / (n - beta - pi_ * beta * (n - 1))))
piceil = simp(sp.solve(sp.Eq(cK, e), pi_)[0])
L.show("interiority ceiling on the share", piceil)
L.check("interior  <=>  pi < theta/Lambda = (n-beta)/(n + beta(n-1))", same(piceil, thK / LamK))
L.check("full income at the full-contribution profile = Lambda e on K_n (W = e + beta(n-1)e/n)", same(e + beta * (n - 1) * e / n, LamK * e))

# ceilings in B
Bimp = sp.symbols("Bbar", positive=True)
pK = thK / LamK
odds_imp_ceiling = thK * LamK ** ((1 - sigma) / sigma) * beta ** (-1 / sigma)
odds_act_ceiling = thK * beta ** (-1 / sigma)
okc = True
for _ in range(60):
    pt = {n: rng.integers(2, 9)}
    pt[beta] = rng.uniform(1.02, pt[n] - 0.02); pt[sigma] = rng.uniform(0.15, 5)
    th_v = 1 - pt[beta] / pt[n]; La_v = 1 + (pt[n] - 1) * pt[beta] / pt[n]
    oi = th_v * La_v ** ((1 - pt[sigma]) / pt[sigma]) * pt[beta] ** (-1 / pt[sigma])
    oa = th_v * pt[beta] ** (-1 / pt[sigma])
    Bi_, Ba_ = oi / (1 + oi), oa / (1 + oa)
    okc &= abs(share(Bi_, th_v / La_v, pt[sigma]) - th_v / La_v) < 1e-10
    okc &= abs(share(Ba_, th_v, pt[sigma]) - th_v / La_v) < 1e-10
L.check("closed-form B-ceilings: odds(impact) = theta Lambda^((1-sigma)/sigma) beta^(-1/sigma), "
        "odds(act) = theta beta^(-1/sigma)  (share hits theta/Lambda exactly)", okc)
L.check("both ceilings reduce to the CD ceiling (n-beta)/(n+beta(n-1)) at sigma = 1",
        same((odds_imp_ceiling / (1 + odds_imp_ceiling)).subs(sigma, 1), (n - beta) / (n + beta * (n - 1)))
        and same((odds_act_ceiling / (1 + odds_act_ceiling)).subs(sigma, 1), (n - beta) / (n + beta * (n - 1))))
L.tex_eq("odds of impact ceiling", r"\frac{\bar B^{K}_{\mathrm{imp}}}{1-\bar B^{K}_{\mathrm{imp}}}", odds_imp_ceiling)
L.tex_eq("odds of act ceiling", r"\frac{\bar B^{K}_{\mathrm{act}}}{1-\bar B^{K}_{\mathrm{act}}}", odds_act_ceiling)
dlog_imp = simp(sp.diff(sp.log(odds_imp_ceiling), sigma))
dlog_act = simp(sp.diff(sp.log(odds_act_ceiling), sigma))
L.show("d log odds(impact ceiling)/d sigma", dlog_imp)
L.show("d log odds(act ceiling)/d sigma", dlog_act)
L.check("impact ceiling: d log odds/d sigma = -log(Lambda/beta)/sigma^2",
        same(dlog_imp, -sp.log(LamK / beta) / sigma**2) or same(sp.expand_log(dlog_imp, force=True), sp.expand_log(-sp.log(LamK / beta) / sigma**2, force=True)))
L.check("Lambda_K > beta on the admissible region (so the impact ceiling widens under complements)",
        all((1 + (m - 1) * b / m) > b for m in range(2, 30) for b in np.linspace(1.01, m - 0.01, 30)))
L.check("act ceiling INcreasing in sigma (d log odds/d sigma = log(beta)/sigma^2 > 0): the mirror image",
        same(sp.expand_log(dlog_act, force=True), sp.log(beta) / sigma**2))

# sigma comparative statics on K_n
L.say("  On K_n, c* is increasing in pi, so sign(dc*/dsigma) = sign(d pi/d sigma) = sign(B - theta/(theta+Lambda)) (s05).")
ex = {n: 3, beta: sp.Rational(3, 2)}
Bsig = (thK / (thK + LamK)).subs(ex)
L.check("experiment (regular): contributions rise with sigma iff B > 1/5; interior (CD) iff B < 1/4  => band (0.2, 0.25)",
        Bsig == sp.Rational(1, 5) and pK.subs(ex) == sp.Rational(1, 4))
G3 = NL.complete(3)
up = [NL.nash_closed(G3, [0.22] * 3, 1.5, 15.0, s)[0] for s in (0.8, 1.0, 1.25)]
dn = [NL.nash_closed(G3, [0.10] * 3, 1.5, 15.0, s)[0] for s in (0.8, 1.0, 1.25)]
L.say(f"  B = 0.22: c*(sigma = 0.8, 1, 1.25) = {np.round(up, 4).tolist()}")
L.say(f"  B = 0.10: c*(sigma = 0.8, 1, 1.25) = {np.round(dn, 4).tolist()}")
L.check("B = 0.22: contributions INCREASE in sigma; B = 0.10: DECREASE  (the 5th-report claim of monotone decline is not general)",
        up[0] < up[1] < up[2] and dn[0] > dn[1] > dn[2])

# beta comparative statics under CES: the Lambda channel has sign sigma - 1
neg_sig, tot = [], 0
for _ in range(3000):
    m = int(rng.integers(2, 9)); b = rng.uniform(1.05, m - 0.05); sg = rng.uniform(0.1, 3); Bv = rng.uniform(0.01, 0.5)
    f = lambda bb: NL.nash_closed(NL.complete(m), [Bv] * m, bb, 15.0, sg)[0]
    c0, c1 = f(b - 1e-5), f(b + 1e-5)
    if c1 >= 15 - 1e-9:
        continue
    tot += 1
    if (c1 - c0) / 2e-5 < -1e-7:
        neg_sig.append(sg)
L.say(f"  K_n: dc*/dbeta < 0 in {len(neg_sig)} of {tot} interior draws (n = 2..8, sigma in (0.1, 3)); sigmas: {np.round(neg_sig, 3).tolist()}")
fex = lambda bb: NL.nash_closed(NL.complete(8), [0.0148] * 8, bb, 15.0, 0.17)[0]
dex = (fex(1.4012) - fex(1.4010)) / 2e-4
L.say(f"  example K_8, B = 0.0148, sigma = 0.17, beta = 1.401: dc*/dbeta = {dex:.4f}")
L.check("dc*/dbeta can be NEGATIVE under strong complements (only sigma < 1 among the draws); "
        "the 5th-report statement that beta raises contributions through the Lambda channel holds only for sigma >= 1",
        dex < 0 and all(sg < 1 for sg in neg_sig))

# ===========================================================================
L.head("2. Cycle C_n (d = 3): same forms with n -> 3 in (theta, Lambda); independent of n")
# ===========================================================================
cCyc = simp(sp.solve(sp.Eq(c, pi_ / (1 - beta / 3) * (e + beta / 3 * 2 * c)), c)[0])
L.check("c*_cyc = 3 pi e/(3 - beta - 2 pi beta), interior iff pi < (3-beta)/(3+2beta)",
        same(cCyc, 3 * pi_ * e / (3 - beta - 2 * pi_ * beta)) and same(sp.solve(sp.Eq(cCyc, e), pi_)[0], (3 - beta) / (3 + 2 * beta)))
ok = True
for m in (4, 5, 6, 7):
    for sg in (0.5, 1.7):
        x = NL.nash_closed(NL.cycle(m), [0.12] * m, 1.5, 15.0, sg)
        y = NL.nash_closed(NL.complete(3), [0.12] * 3, 1.5, 15.0, sg)
        ok &= np.allclose(x, x[0]) and abs(x[0] - y[0]) < 1e-10
L.check("C_n equilibrium under CES is symmetric and equals K_3's for every n (both have d = 3, Lambda = 1 + 2beta/3)", ok)

# ===========================================================================
L.head("3. Star S_n with a common B: position-specific shares (pi_C, pi_L)")
# ===========================================================================
sol = sp.solve([sp.Eq(cC, piC / (1 - beta / n) * (e + beta / n * (n - 1) * cL)),
                sp.Eq(cL, piL / (1 - beta / 2) * (e + beta / 2 * cC))], [cC, cL], dict=True)[0]
cCs, cLs = simp(sol[cC]), simp(sol[cL])
DSp = (n - beta) * (2 - beta) - (n - 1) * piC * piL * beta**2
L.check("c*_C = pi_C e [n(2-beta) + 2(n-1) pi_L beta] / [(n-beta)(2-beta) - (n-1) pi_C pi_L beta^2]",
        same(cCs, piC * e * (n * (2 - beta) + 2 * (n - 1) * piL * beta) / DSp))
L.check("c*_L = pi_L e [2(n-beta) + n pi_C beta] / same denominator",
        same(cLs, piL * e * (2 * (n - beta) + n * piC * beta) / DSp))
L.tex_eq("CES star centre", r"c^\star_{S,C}", cCs)
L.tex_eq("CES star leaf", r"c^\star_{S,L}", cLs)
gap = simp(cLs - cCs)
L.show("c*_L - c*_C", gap)
L.check("leaf > centre  <=>  pi_L [2(n-beta) + n pi_C beta] > pi_C [n(2-beta) + 2(n-1) pi_L beta]",
        same(sp.numer(sp.together(gap)) / e, sp.expand(piL * (2 * (n - beta) + n * piC * beta) - piC * (n * (2 - beta) + 2 * (n - 1) * piL * beta)))
        or same(-sp.numer(sp.together(gap)) / e, sp.expand(piL * (2 * (n - beta) + n * piC * beta) - piC * (n * (2 - beta) + 2 * (n - 1) * piL * beta))))
# positions: p_C, p_L
pC_imp = (1 - beta / n) / (1 + (n - 1) * beta / 2)
pL_imp = (1 - beta / 2) / (1 + beta / n)
vals = [(m, b, float(sp.N(pL_imp.subs({n: m, beta: b}))), float(sp.N(pC_imp.subs({n: m, beta: b})))) for m in (3, 5) for b in (1.2, 1.5, 1.9)]
for v in vals:
    L.say(f"  n={v[0]}, beta={v[1]}: p_L = {v[2]:.4f}, p_C = {v[3]:.4f}")

# ordering over a grid (both specifications)
def ordering_scan(ns, betas, sigmas, Bs, imps):
    out = []
    for m in ns:
        for b in betas:
            for sg in sigmas:
                for imp in imps:
                    for Bv in Bs:
                        x = NL.nash_closed(NL.star(m), [Bv] * m, b, 15.0, sg, imp)
                        if x[1] >= 15 - 1e-9:      # leaf at the corner: ordering only weak
                            continue
                        out.append((m, b, sg, imp, Bv, x[1] - x[0]))
    return out


SIG = np.concatenate([np.linspace(0.1, 1, 7), np.linspace(1.25, 8, 10)])
BB = np.linspace(0.005, 0.6, 18)
r_exp = ordering_scan([3], np.linspace(1.2, 1.95, 7), SIG, BB, (True, False))
L.say(f"  (a) experiment n = 3, beta in [1.2, 1.95]: {len(r_exp)} interior-leaf profiles, violations = {sum(g <= 0 for *_, g in r_exp)}")
L.check("(a) n = 3 on the admissible band beta >= 6/5: leaf > centre at EVERY sigma, both specifications",
        all(g > 0 for *_, g in r_exp))
r_act = ordering_scan(range(3, 9), np.linspace(1.05, 1.95, 7), SIG, BB, (False,))
L.check("(b) act-based specification: leaf > centre for all n = 3..8, beta in (1.05, 1.95), all sigma",
        all(g > 0 for *_, g in r_act))
r_imp = ordering_scan(range(3, 9), np.linspace(1.05, 1.95, 7), SIG, BB, (True,))
viol = [r for r in r_imp if r[5] <= 0]
L.say(f"  (c) impact-based, n = 3..8, beta in (1.05, 1.95): violations {len(viol)} of {len(r_imp)}; "
      f"min sigma among violations = {min(v[2] for v in viol):.2f}; max beta = {max(v[1] for v in viol):.2f}")
if viol:
    v = max(viol, key=lambda r: -r[5])
    L.say(f"      example: n={v[0]}, beta={v[1]:.2f}, sigma={v[2]:.2f}, B={v[4]:.3f}: leaf - centre = {v[5]:.4f}")
L.check("(c) impact-based: the ordering CAN reverse, only under substitutes well above unit elasticity (sigma >= 2 on the grid) "
        "-- the moral channel Lambda_C > Lambda_L overturns the price channel",
        len(viol) > 0 and min(v[2] for v in viol) >= 2 - 1e-9 and all(v[3] for v in viol))

# H4 at n = 3: leaf > centre > regular member
def h4_scan(sigmas, imps):
    out = []
    for b in np.linspace(1.2, 1.95, 6):
        for sg in sigmas:
            for imp in imps:
                for Bv in np.linspace(0.005, 0.5, 20):
                    s_ = NL.nash_closed(NL.star(3), [Bv] * 3, b, 15.0, sg, imp)
                    r_ = NL.nash_closed(NL.complete(3), [Bv] * 3, b, 15.0, sg, imp)[0]
                    if s_[0] >= 15 - 1e-9 or r_ >= 15 - 1e-9:
                        continue
                    out.append((b, sg, imp, Bv, s_[1] - r_, s_[0] - r_))
    return out


h4 = h4_scan(SIG, (True, False))
L.check("(d) n = 3: leaf > regular member at every sigma, both specifications", all(x[4] > 0 for x in h4))
L.check("(e) n = 3: centre > regular member at every sigma under the ACT-based specification",
        all(x[5] > 0 for x in h4 if not x[2]))
L.check("(f) n = 3: centre > regular member under the IMPACT-based specification whenever sigma >= 1",
        all(x[5] > 0 for x in h4 if x[2] and x[1] >= 1))
rev = [x for x in h4 if x[2] and x[1] < 1 and x[5] <= 0]
L.say(f"  (g) impact-based, sigma < 1: centre <= regular in {len(rev)} of {sum(1 for x in h4 if x[2] and x[1] < 1)} profiles")
# analytical limit as B -> 0: c_C/c_K -> pi_C/pi_K -> (p_C/p_K)^(1-sigma)
lim_ok = True
for b in (1.2, 1.5, 1.8):
    for sg in (0.3, 0.6, 0.9, 1.5):
        d_, th_, La_S, _ = NL.primitives(NL.star(3), b)
        _, _, La_K, _ = NL.primitives(NL.complete(3), b)
        ratio_lim = ((th_[0] / La_S[0]) / (th_[0] / La_K[0])) ** (1 - sg)
        Bv = 1e-20
        rr = NL.nash_closed(NL.star(3), [Bv] * 3, b, 15.0, sg)[0] / NL.nash_closed(NL.complete(3), [Bv] * 3, b, 15.0, sg)[0]
        lim_ok &= abs(rr - ratio_lim) < 1e-3
L.check("(g) as B -> 0, c_C*/c_K* -> (p_C/p_K)^(1-sigma) = (Lambda_K/Lambda_C)^(1-sigma) < 1 for sigma < 1: "
        "under impact-based complements the centre falls below the regular member at small B", lim_ok and len(rev) > 0)

# mixed regime: leaves at e -> centre ceiling in closed form
L.say("  mixed regime: c_C = pi_C Lambda_K e/theta_C  =>  centre interior iff pi_C < theta_C/Lambda_K = (n-beta)/(n+beta(n-1))")
okm = True
for _ in range(300):
    m = int(rng.integers(3, 9)); b = rng.uniform(1.05, 1.95); sg = rng.uniform(0.2, 4)
    thC = 1 - b / m; LaK = 1 + (m - 1) * b / m; LaC = 1 + (m - 1) * b / 2
    q = thC / LaK
    for imp in (True, False):
        pC = thC / LaC if imp else thC
        odds = (q / (1 - q)) ** (1 / sg) * pC ** ((sg - 1) / sg)
        Bc = odds / (1 + odds)
        okm &= abs(share(Bc, pC, sg) - q) < 1e-10
L.check("closed-form centre ceiling: odds = (theta_C/beta)^(1/sigma) p_C^((sigma-1)/sigma)  (q/(1-q) = theta_C/beta)", okm)

# ===========================================================================
L.head("4. Formula-free check: maximise the actual CES utility (no closed forms)")
# ===========================================================================
worst = 0.0
for _ in range(40):
    kind = int(rng.integers(0, 3)); m = int(rng.integers(3, 6))
    G = [NL.complete, NL.cycle, NL.star][kind](m)
    b = rng.uniform(1.05, 1.95); sg = float(rng.choice([0.4, 0.7, 1.0, 1.5, 2.5])); imp = bool(rng.integers(0, 2))
    Bv = [rng.uniform(0.02, 0.3)] * m
    x = NL.nash_closed(G, Bv, b, 15.0, sg, imp); y = NL.nash_utility(G, Bv, b, 15.0, sg, imp)
    worst = max(worst, np.max(abs(x - y)))
L.say(f"  max |closed form - utility maximisation| = {worst:.2e}")
L.check("closed forms = direct utility maximisation (40 random games, all topologies, both specs)", worst < 1e-5)

# ===========================================================================
L.head("5. Experimental values (e = 15, n = 3, beta = 1.5, B = 0.10)")
# ===========================================================================
L.say("  sigma | impact: regular  centre   leaf   gap | act: regular  centre   leaf   gap")
rows = {}
for sg in (0.5, 0.75, 1.0, 1.5, 2.0):
    ri = NL.nash_closed(NL.complete(3), [0.1] * 3, 1.5, 15.0, sg, True)[0]
    si = NL.nash_closed(NL.star(3), [0.1] * 3, 1.5, 15.0, sg, True)
    ra = NL.nash_closed(NL.complete(3), [0.1] * 3, 1.5, 15.0, sg, False)[0]
    sa = NL.nash_closed(NL.star(3), [0.1] * 3, 1.5, 15.0, sg, False)
    rows[sg] = (ri, si, ra, sa)
    L.say(f"  {sg:4.2f}  | {ri:7.3f} {si[0]:7.3f} {si[1]:7.3f} {si[1]-si[0]:6.3f} | {ra:7.3f} {sa[0]:7.3f} {sa[1]:7.3f} {sa[1]-sa[0]:6.3f}")
L.check("5th-report numbers reproduced: sigma=2 centre 2.28 (impact) vs 0.87 (act); sigma=1.5 gaps 2.61 vs 2.60",
        abs(rows[2.0][1][0] - 2.28) < 5e-3 and abs(rows[2.0][3][0] - 0.865) < 5e-3
        and abs(rows[1.5][1][1] - rows[1.5][1][0] - 2.605) < 5e-3 and abs(rows[1.5][3][1] - rows[1.5][3][0] - 2.595) < 5e-3)
L.say("  ceilings on B (impact / act) by sigma, experiment:")
for sg in (0.5, 0.75, 1.0, 1.5, 2.0):
    th_, La_ = 0.5, 2.0
    oi = th_ * La_ ** ((1 - sg) / sg) * 1.5 ** (-1 / sg); oa = th_ * 1.5 ** (-1 / sg)
    # star leaf ceiling numerically (bisection on the common B)
    def leaf_ceiling(imp):
        lo, hi = 1e-6, 0.999
        for _ in range(80):
            mid = (lo + hi) / 2
            if NL.nash_closed(NL.star(3), [mid] * 3, 1.5, 15.0, sg, imp)[1] >= 15 - 1e-10:
                hi = mid
            else:
                lo = mid
        return hi
    L.say(f"    sigma={sg:4.2f}: regular {oi/(1+oi):.4f} / {oa/(1+oa):.4f};  star leaf {leaf_ceiling(True):.4f} / {leaf_ceiling(False):.4f}")

L.finish()
