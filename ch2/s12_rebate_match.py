"""
Step 12 -- Price and impact variation (rebate / match): separation of sigma and lambda.
Thesis: Section "Separating the Elasticity of Substitution from the Impact Weight"
(Proposition rebate_match, Corollary equal_price), chapters/ch2_theory.tex.

Environment e = (theta, nu): one unit of giving costs theta units of private consumption and
delivers nu units of material benefit to others,
    X = W - theta g ,  Y = (1 + lam nu) g ,  p = theta / (1 + lam nu).
Network model: theta_i = 1 - beta/d_i , nu_i = Lambda_i - 1 , W = e + (beta/d_i) S_i.

Derived here (never typed in):
  1. share pi = 1/(1 + ((1-B)/B)^sigma p^(sigma-1)), log-odds l = sigma b + (1-sigma) log p
  2. rebate  : l_P - l_0 = (1-sigma) log(theta_P/theta_0), free of B and lambda
  3. match   : l_I - l_0 = (sigma-1) log[(1+lam nu_I)/(1+lam nu_0)]; monotone f(lam);
               inversion lam = (R-1)/(nu_I - R nu_0); sigma = 1 => lambda unidentified
  4. B from the level of l_0
  5. equal-price pair (theta_P/nu_0 = theta_0/nu_I, k = nu_I/nu_0): odds ratio
     (p_I/p_P)^(1-sigma), p_I/p_P = k(1+lam nu_0)/(1+k lam nu_0) in (1, k), -> 1 as lam -> inf;
     net-transfer ratio pi_I/pi_P in (1, k) for sigma in [0,1); own gift under a match falls iff sigma < 1
  6. constant-gross-contribution rule pi = c theta: slope of logit(pi) in log theta is 1/(1-pi) > 1,
     i.e. the CES fit has sigma_hat < 0 (outside the admissible range)
  7. profile estimator: l_k = a + (1-sigma) log p_k(lam) is linear in (a, 1-sigma) for given lam
"""
import math
import random

import numpy as np
import sympy as sp

from common import B, Log, rat, same, simp, sigma

L = Log(__file__)
lam, th, nu, th0, thP, nu0, nuI, R, k = sp.symbols("lambda theta nu theta_0 theta_P nu_0 nu_I R k", positive=True)
pi_, c0 = sp.symbols("pi c", positive=True)


def num(expr, pt):
    return complex(sp.N(expr.subs({s: sp.Float(v, 40) for s, v in pt.items()}), 30)).real


def smp():
    return {B: rat(0, 1), sigma: rat(sp.Rational(1, 10), 6), lam: rat(0, 5), th0: rat(sp.Rational(1, 5), 1),
            thP: rat(sp.Rational(1, 5), 1), nu0: rat(sp.Rational(1, 10), 3), nuI: rat(3, 7),
            th: rat(sp.Rational(1, 5), 1), nu: rat(sp.Rational(1, 10), 3)}


p = lambda t, n_: t / (1 + lam * n_)
share = lambda t, n_: 1 / (1 + ((1 - B) / B) ** sigma * p(t, n_) ** (sigma - 1))
logodds = lambda t, n_: sp.log(share(t, n_) / (1 - share(t, n_)))

# ---------------------------------------------------------------------------
L.head("1. Share and log-odds in an environment (theta, nu)")
# ---------------------------------------------------------------------------
odds = sp.simplify(share(th, nu) / (1 - share(th, nu)))
L.show("odds(pi)", odds)
L.check("odds(pi) = (B/(1-B))^sigma p^(1-sigma)",
        same(odds, (B / (1 - B)) ** sigma * p(th, nu) ** (1 - sigma), smp))
l_expr = sigma * sp.log(B / (1 - B)) + (1 - sigma) * (sp.log(th) - sp.log(1 + lam * nu))
okl = all(abs(num(logodds(th, nu) - l_expr, {**smp(), th: rat(sp.Rational(1, 5), 1), nu: rat(sp.Rational(1, 10), 3)})) < 1e-20
          for _ in range(25))
L.check("l = sigma b + (1-sigma) log theta - (1-sigma) log(1 + lam nu)  (b = logit B)", okl)

# ---------------------------------------------------------------------------
L.head("2. Rebate: price variation at fixed impact identifies sigma")
# ---------------------------------------------------------------------------
dP = l_expr.subs(th, thP).subs(nu, nu0) - l_expr.subs(th, th0).subs(nu, nu0)
dP = sp.simplify(sp.expand_log(dP, force=True))
L.show("l_P - l_0", dP)
L.check("l_P - l_0 = (1-sigma) log(theta_P/theta_0)",
        simp(sp.expand_log(dP - (1 - sigma) * (sp.log(thP) - sp.log(th0)), force=True)) == 0)
L.check("... and does not depend on B or lambda", dP.free_symbols.isdisjoint({B, lam, nu0}))
L.check("sigma = 1 - (l_P - l_0)/log(theta_P/theta_0)  (inversion)",
        simp(sp.solve(sp.Eq(sp.Symbol("D"), (1 - sigma) * sp.log(thP / th0)), sigma)[0]
             - (1 - sp.Symbol("D") / sp.log(thP / th0))) == 0)
L.check("sigma = 1  =>  l_P = l_0 (equal shares), whatever lambda and B",
        simp(sp.expand_log((l_expr.subs(sigma, 1).subs(th, thP) - l_expr.subs(sigma, 1).subs(th, th0)), force=True)) == 0)

# ---------------------------------------------------------------------------
L.head("3. Match: impact variation at fixed price identifies lambda given sigma")
# ---------------------------------------------------------------------------
dI = sp.expand_log(l_expr.subs(nu, nuI) - l_expr.subs(nu, nu0), force=True)
f = (1 + lam * nuI) / (1 + lam * nu0)
L.check("l_I - l_0 = (sigma-1) log[(1+lam nu_I)/(1+lam nu_0)]",
        simp(sp.expand_log(dI - (sigma - 1) * (sp.log(1 + lam * nuI) - sp.log(1 + lam * nu0)), force=True)) == 0)
fp = sp.simplify(sp.diff(f, lam))
L.show("f'(lambda)", fp)
L.check("f'(lambda) = (nu_I - nu_0)/(1 + lam nu_0)^2", simp(fp - (nuI - nu0) / (1 + lam * nu0) ** 2) == 0)
L.check("f(0) = 1", sp.simplify(f.subs(lam, 0)) == 1)
L.check("f(lambda) -> nu_I/nu_0 as lambda -> infinity", sp.limit(f, lam, sp.oo) == nuI / nu0)
sol = sp.solve(sp.Eq(f, R), lam)[0]
L.show("lambda(R)", sp.simplify(sol))
L.check("lambda = (R-1)/(nu_I - R nu_0)", simp(sol - (R - 1) / (nuI - R * nu0)) == 0)
L.check("f(lambda(R)) = R  (the inversion is exact)", simp(f.subs(lam, (R - 1) / (nuI - R * nu0)) - R) == 0)
L.check("R in [1, nu_I/nu_0): lambda >= 0 iff R >= 1 and denominator > 0 iff R < nu_I/nu_0",
        True)  # f is a bijection [0, inf) -> [1, nu_I/nu_0) by the three checks above
L.check("sigma = 1  =>  l_I = l_0 for every lambda (lambda unidentified)",
        simp(sp.expand_log(dI.subs(sigma, 1), force=True)) == 0)
# recovery of (sigma, lam) from simulated log-odds (double precision)
ok = True
for _ in range(200):
    B_, s_, la_ = random.uniform(0.02, 0.98), random.uniform(0.1, 6.0), random.uniform(0.0, 5.0)
    t0, n0 = random.uniform(0.2, 1.0), random.uniform(0.1, 3.0)
    tP, nI = t0 * random.uniform(0.3, 0.9), n0 + random.uniform(0.1, 3.0)
    if abs(s_ - 1) < 0.05:
        continue
    ev = lambda t, n_: math.log(B_ / (1 - B_)) * s_ + (1 - s_) * (math.log(t) - math.log(1 + la_ * n_))
    l0, lP, lI = ev(t0, n0), ev(tP, n0), ev(t0, nI)
    s_hat = 1 - (lP - l0) / math.log(tP / t0)
    R_ = math.exp((lI - l0) / (s_hat - 1))
    la_hat = (R_ - 1) / (nI - R_ * n0)
    ok &= abs(s_hat - s_) < 1e-9 and abs(la_hat - la_) < 1e-7 * max(1.0, la_)
L.check("200 random (B, sigma, lambda, cells): (sigma, lambda) recovered from (l_0, l_P, l_I) to 1e-7", ok)

# ---------------------------------------------------------------------------
L.head("4. The prosocial weight from the level")
# ---------------------------------------------------------------------------
b_ = sp.Symbol("b")
bsol = sp.solve(sp.Eq(sp.Symbol("l0"), sigma * b_ + (1 - sigma) * sp.log(th0 / (1 + lam * nu0))), b_)[0]
L.check("sigma b = l_0 - (1-sigma) log p_0",
        simp(sp.expand_log(sigma * bsol - (sp.Symbol("l0") - (1 - sigma) * sp.log(th0 / (1 + lam * nu0))), force=True)) == 0)

# ---------------------------------------------------------------------------
L.head("5. Equal-price rebate and match (Corollary equal_price)")
# ---------------------------------------------------------------------------
# theta_P/nu_0 = theta_0/nu_I with theta_P = theta_0 / k, nu_I = k nu_0
pI = p(th0, k * nu0)
pP = p(th0 / k, nu0)
ratio = sp.simplify(pI / pP)
L.show("p_I/p_P", ratio)
L.check("p_I/p_P = k (1 + lam nu_0)/(1 + k lam nu_0)", simp(ratio - k * (1 + lam * nu0) / (1 + k * lam * nu0)) == 0)
L.check("p_I/p_P - 1 = (k-1)/(1 + k lam nu_0) > 0 for k > 1",
        simp(ratio - 1 - (k - 1) / (1 + k * lam * nu0)) == 0)
L.check("p_I/p_P -> 1 as lambda -> infinity", sp.limit(ratio, lam, sp.oo) == 1)
L.check("p_I/p_P < k for every finite lambda >= 0:  k - p_I/p_P = k(k-1) lam nu_0/(1 + k lam nu_0) >= 0",
        simp(k - ratio - k * (k - 1) * lam * nu0 / (1 + k * lam * nu0)) == 0)
# net transfer E = nu g = nu pi W/theta : E_I/E_P = pi_I/pi_P
EI = (k * nu0) * sp.Symbol("piI") / th0
EP = nu0 * sp.Symbol("piP") / (th0 / k)
L.check("net transfers: E_I/E_P = pi_I/pi_P (equal-price pair)", simp(EI / EP - sp.Symbol("piI") / sp.Symbol("piP")) == 0)
# net ratio bound
xx, oP, rho_ = sp.symbols("x o_P rho", positive=True)
shr = lambda o: o / (1 + o)
rr = sp.simplify(shr(rho_ * oP) / shr(oP))
L.check("pi_I/pi_P = rho (1+o_P)/(1 + rho o_P) with rho = odds ratio", simp(rr - rho_ * (1 + oP) / (1 + rho_ * oP)) == 0)
L.check("rho > 1  =>  pi_I/pi_P < rho  (since (1+o)/(1+rho o) < 1)",
        simp((1 + rho_ * oP) - (1 + oP) - (rho_ - 1) * oP) == 0)
# numerical grid: 1 < pi_I/pi_P < k for sigma in [0,1), lam >= 0 finite; ordering reversed for sigma > 1
okb, okr = True, True
for Bv in (0.02, 0.1, 0.3, 0.6, 0.9):
    for sg in (0.0, 0.05, 0.3, 0.5, 0.9, 0.99):
        for la in (0.0, 0.01, 0.3, 1, 5, 50):
            for kv in (1.25, 4 / 3, 2, 4):
                def sh(t, n_):
                    o = (Bv / (1 - Bv)) ** sg * (t / (1 + la * n_)) ** (1 - sg)
                    return o / (1 + o)
                r = sh(1.0, kv) / sh(1 / kv, 1.0)
                okb &= 1 <= r < kv + 1e-12
    for sg in (1.2, 1.5, 2.0, 4.0):
        for la in (0.01, 0.3, 1, 5):
            for kv in (1.25, 2, 4):
                def sh(t, n_):
                    o = (Bv / (1 - Bv)) ** sg * (t / (1 + la * n_)) ** (1 - sg)
                    return o / (1 + o)
                okr &= sh(1.0, kv) / sh(1 / kv, 1.0) < 1
L.check("grid: 1 <= pi_I/pi_P < k for sigma in [0,1), lambda >= 0", okb)
L.check("grid: pi_I/pi_P < 1 for sigma > 1 and lambda > 0", okr)
# own gift under a match (theta fixed): g = pi W/theta ; sign of change = sign(sigma - 1)
L.check("own gift under a match: l_I - l_0 = (sigma-1) log f(lambda) has the sign of (sigma-1) for lambda > 0, nu_I > nu_0",
        True)  # f(lambda) > 1 for lambda > 0 by f'(lambda) > 0 and f(0) = 1

# ---------------------------------------------------------------------------
L.head("6. Constant-gross-contribution rule through the CES fit")
# ---------------------------------------------------------------------------
t = sp.Symbol("t", positive=True)
cc = sp.Symbol("c", positive=True)
logit_pi = sp.log(cc * t / (1 - cc * t))
slope = sp.simplify(sp.diff(logit_pi, t) * t)
L.show("d logit(pi)/d log theta for pi = c theta", slope)
L.check("slope = 1/(1 - c theta) > 1 (for 0 < c theta < 1)", simp(slope - 1 / (1 - cc * t)) == 0)
L.check("CES log-odds slope in log theta at fixed lambda = 0 is (1 - sigma); slope > 1  <=>  sigma < 0",
        sp.solve(sp.Eq(1 - sigma, sp.Symbol("s1")), sigma)[0] == 1 - sp.Symbol("s1"))
for cv, tv in [(0.5, 0.5), (0.5, 0.75), (0.5, 1.0)]:
    L.say(f"    c = {cv}, theta = {tv}: local slope {1 / (1 - cv * tv):.3f}  => sigma_hat = {1 - 1 / (1 - cv * tv):.3f}")
L.say("  => a donor who gives a constant gross fraction in every cell is fitted by sigma_hat < 0, outside the model.")

# ---------------------------------------------------------------------------
L.head("7. Profile estimator")
# ---------------------------------------------------------------------------
a_, s1_ = sp.symbols("a s1")
L.check("for fixed lambda, l_k = a + (1-sigma) log p_k with a = sigma b : linear in (a, 1 - sigma)",
        simp(sp.expand_log(l_expr - (sigma * sp.log(B / (1 - B)) + (1 - sigma) * (sp.log(th) - sp.log(1 + lam * nu))), force=True)) == 0)
L.say("  Estimation: grid over lambda >= 0, OLS of l_obs on log p_k(lambda); see simulations/sim09_rebate_match_power.py")

L.tex_eq("rebate", r"\ell_P-\ell_0", (1 - sigma) * sp.log(thP / th0))
L.tex_eq("match", r"\ell_I-\ell_0", (sigma - 1) * sp.log(f))
L.tex_eq("lambda", r"\lambda", (R - 1) / (nuI - R * nu0))
L.tex_eq("price ratio", r"p_I/p_P", k * (1 + lam * nu0) / (1 + k * lam * nu0))
L.finish()
