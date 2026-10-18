"""
Step 3a -- The CES individual problem at an arbitrary network position.

Notation (unified with Cobb-Douglas): the prosocial weight is B_i = 1 - delta_i.

    U = [ (1-B) X^rho + B Y^rho ]^(1/rho),  rho in (-oo,1)\\{0},  sigma = 1/(1-rho)
    X = W - theta c,  W = e + (beta/d) S,  Y = Lambda c.

Derived here:
  * FOC => c = kappa~ X,  kappa~ = (B/((1-B) theta))^sigma Lambda^(sigma-1);
  * best response as a CES demand:  c = (pi/theta) W, with expenditure share
        pi = 1 / (1 + ((1-B)/B)^sigma p^(sigma-1)),   p = theta/Lambda
    (p = price of one unit of the moral good in units of private consumption);
  * limits: sigma -> 1 gives pi = B (Cobb-Douglas); sigma -> 0 (Leontief) gives
    pi = p/(1+p), independent of B; sigma -> oo gives a bang-bang share;
  * concavity along the budget line, no lower corner;
  * the moral channel  d log odds(pi)/d log Lambda = sigma - 1;
  * the sigma channel  d log odds(pi)/d sigma = log(B Lambda/((1-B) theta)),
    hence own giving rises with sigma iff B > theta/(theta + Lambda);
  * act-based (Lambda = 1) vs impact-based shares coincide iff sigma = 1.
"""

import random

import numpy as np
import sympy as sp

from common import Lam, Log, B, beta, d, e, rat, same, simp, sigma

L = Log(__file__)
c, S, W, th, p = sp.symbols("c S W theta p", positive=True)
r = sp.symbols("r", positive=True)
rho = 1 - 1 / sigma


def fl(pt):
    """Exact rationals -> 50-digit floats (avoids exact Rational**Rational root extraction)."""
    return {k: sp.Float(v, 50) for k, v in pt.items()}


def num(expr, pt):
    return complex(sp.N(expr.subs(fl(pt)), 40))


def sampler():
    dd = random.randint(2, 9)
    b = rat(1, dd)
    return {d: dd, beta: b, th: 1 - b / dd, e: rat(1, 30), S: rat(0, 30), B: rat(0, 1),
            Lam: rat(1, 6), sigma: rat(sp.Rational(1, 10), 6), W: rat(1, 60)}


# ---------------------------------------------------------------------------
L.head("1. First-order condition and the CES demand for giving")
# ---------------------------------------------------------------------------
Xs, Ys = sp.symbols("X Y", positive=True)
Zs = (1 - B) * Xs**rho + B * Ys**rho
# chain rule along the budget line: dX/dc = -theta, dY/dc = Lambda
foc = (sp.diff(Zs, Xs) * (-th) + sp.diff(Zs, Ys) * Lam) / rho
foc = foc.subs(Ys, Lam * c)
L.show("dZ/dc / rho", foc)
foc_r = sp.simplify(foc.subs(c, r * Xs) / Xs ** (rho - 1))
foc_r = sp.powsimp(sp.expand_power_base(foc_r, force=True), force=True)
L.show("FOC divided by X^(rho-1), with c = r X", foc_r)
L.check("the FOC depends on (c, X) only through the ratio r = c/X", Xs not in sp.simplify(foc_r).free_symbols)
X = W - th * c
Y = Lam * c
Z = (1 - B) * X**rho + B * Y**rho
kt = (B / ((1 - B) * th)) ** sigma * Lam ** (sigma - 1)
okfoc = all(abs(num(foc_r.subs(r, kt), pt)) < 1e-25 for pt in [sampler() for _ in range(60)])
L.check("FOC holds at c/X = kappa~ = (B/((1-B) theta))^sigma Lambda^(sigma-1)", okfoc)
# uniqueness of the root: foc_r is monotone in r (derivative sign constant)
dfr = sp.diff(foc_r, r)
L.check("FOC(r) is strictly decreasing in r  => unique ratio (40 random points)", all(
    num(dfr, {**pt, r: rat(0, 20)}).real < 0 for pt in [sampler() for _ in range(40)]))

c_br = kt * W / (1 + kt * th)
pi = 1 / (1 + ((1 - B) / B) ** sigma * p ** (sigma - 1))
okpi = all(abs(num(th * c_br / W - pi.subs(p, th / Lam), pt)) < 1e-25 for pt in [sampler() for _ in range(80)])
L.check("BR = kappa~ W/(1 + kappa~ theta) = (pi/theta) W,  pi = 1/(1 + ((1-B)/B)^sigma p^(sigma-1)), p = theta/Lambda", okpi)
L.tex_eq("pi", r"\pi_i", pi.subs({B: sp.Symbol("B_i"), p: sp.Symbol("p_i")}))
L.check("X at the optimum = (1 - pi) W  (expenditure share of private consumption)",
        all(abs(num(W - th * c_br - (1 - pi.subs(p, th / Lam)) * W, pt)) < 1e-25 for pt in [sampler() for _ in range(40)]))
L.say("  => alpha_i = pi_i e/theta_i, gamma_i = pi_i beta/(d_i - beta): the CD apparatus applies with B_i -> pi_i")

# ---------------------------------------------------------------------------
L.head("2. Limits")
# ---------------------------------------------------------------------------
L.check("sigma -> 1: pi -> B  (Cobb-Douglas)", sp.simplify(sp.limit(pi, sigma, 1) - B) == 0)
pi0 = sp.limit(pi.subs({B: sp.Rational(2, 7), p: sp.Rational(3, 5)}), sigma, 0, "+")
L.check("sigma -> 0 (Leontief): pi -> p/(1+p), independent of B", sp.simplify(pi0 - sp.Rational(3, 5) / (1 + sp.Rational(3, 5))) == 0)
L.check("sigma -> oo (perfect substitutes): pi -> 1 if B/(1-B) > p, -> 0 if B/(1-B) < p",
        sp.limit(pi.subs({B: sp.Rational(1, 2), p: sp.Rational(1, 2)}), sigma, sp.oo) == 1
        and sp.limit(pi.subs({B: sp.Rational(1, 5), p: sp.Rational(1, 2)}), sigma, sp.oo) == 0)
L.check("kappa~ -> B/((1-B) theta) = kappa as sigma -> 1", sp.simplify(sp.limit(kt, sigma, 1) - B / ((1 - B) * th)) == 0)

# ---------------------------------------------------------------------------
L.head("3. Concavity and the lower corner")
# ---------------------------------------------------------------------------
U = Z ** (1 / rho)
ok = True
for _ in range(150):
    pt = sampler()
    cc = rat(0, 1) * pt[W] / pt[th]
    v = sp.N(sp.diff(U, c, 2).subs(fl({**pt, c: cc})), 30)
    ok &= v < 0
L.check("d^2 U/dc^2 < 0 on the budget line (150 random points, sigma in (0.1, 6))", ok)
ok = True
for sg in (sp.Rational(3, 2), 4):          # gross substitutes: infinite marginal utility at c = 0
    pt = {B: sp.Rational(1, 10), th: sp.Rational(1, 2), Lam: 2, W: 15, sigma: sg}
    ok &= sp.limit(sp.diff(U, c).subs(pt), c, 0, "+") == sp.oo
for sg in (sp.Rational(1, 3), sp.Rational(2, 3)):   # gross complements: U(0) = 0 < U(c) for c > 0
    pt = {B: sp.Rational(1, 10), th: sp.Rational(1, 2), Lam: 2, W: 15, sigma: sg}
    ok &= sp.limit(U.subs(pt), c, 0, "+") == 0 and sp.N(U.subs(pt).subs(c, sp.Rational(1, 100))) > 0
L.check("no lower corner: dU/dc -> +oo at c=0 if sigma > 1; U(0) = 0 < U(c>0) if sigma < 1", ok)

# ---------------------------------------------------------------------------
L.head("4. Comparative statics of the share")
# ---------------------------------------------------------------------------
odds = (B / (1 - B)) ** sigma * p ** (1 - sigma)
def rel_close(a, b, pt, tol=1e-30):
    va, vb = num(a, pt), num(b, pt)
    return abs(va - vb) <= tol * max(1.0, abs(vb))


L.check("pi/(1-pi) = (B/(1-B))^sigma p^(1-sigma)",
        all(rel_close(pi / (1 - pi), odds, {**pt, p: rat(0, 2)}) for pt in [sampler() for _ in range(40)]))
dpiB = sp.diff(pi, B)
L.check("d pi/dB = sigma pi (1-pi)/(B(1-B)) > 0",
        all(abs(num(dpiB - sigma * pi * (1 - pi) / (B * (1 - B)), {**pt, p: rat(0, 2)})) < 1e-25 for pt in [sampler() for _ in range(40)]))
lo_Lam = sp.simplify(sp.diff(sp.log(odds.subs(p, th / Lam)), Lam) * Lam)
L.show("d log odds / d log Lambda", lo_Lam)
L.check("moral channel: d log odds/d log Lambda = sigma - 1  (>0 substitutes, <0 complements, 0 at CD)",
        sp.simplify(lo_Lam - (sigma - 1)) == 0)
lo_sig = sp.simplify(sp.expand_log(sp.diff(sp.log(odds.subs(p, th / Lam)), sigma), force=True))
L.show("d log odds / d sigma", lo_sig)
L.check("sigma channel: d log odds/d sigma = log(B Lambda/((1-B) theta))",
        all(abs(num(lo_sig - sp.log(B * Lam / ((1 - B) * th)), pt)) < 1e-25 for pt in [sampler() for _ in range(40)]))
Bsig = sp.solve(sp.Eq(B * Lam / ((1 - B) * th), 1), B)[0]
L.show("threshold: share rises with sigma iff B >", Bsig)
L.check("threshold B^sigma = theta/(theta + Lambda) (impact); theta/(theta + 1) (act-based)",
        same(Bsig, th / (th + Lam)))
L.tex_eq("sigma threshold", r"B^{\sigma}_i", th / (th + Lam))
L.say("  note: p_i = theta_i/Lambda_i falls with beta at every position (theta_i falls, Lambda_i rises)")

# ---------------------------------------------------------------------------
L.head("5. Act-based (Y = c) versus impact-based (Y = Lambda c)")
# ---------------------------------------------------------------------------
ratio = sp.simplify(odds.subs(p, th / Lam) / odds.subs(p, th))
L.show("odds(impact)/odds(act)", ratio)
L.check("odds(impact)/odds(act) = Lambda^(sigma-1): identical shares (and contributions) iff sigma = 1 (Lambda > 1)",
        sp.simplify(ratio - Lam ** (sigma - 1)) == 0)

L.finish()
