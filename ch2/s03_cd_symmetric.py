"""
Step 1c -- Cobb-Douglas equilibria under a COMMON prosocial weight B
           ("symmetric" in preferences) on K_n, C_n and S_n, general n.

Normalisation A = 1 - B (so the expenditure share b = B/(A+B) equals B).
For arbitrary A, B > 0 every formula holds with B replaced by b = B/(A+B).

Best response (s02):  c_i = (B/theta_i) (e + (beta/d_i) S_i).

Derived here:
  K_n : symmetric equilibrium, existence, interiority ceiling, uniqueness,
        comparative statics in (B, e, beta, n), the full-corner regime;
  C_n : the same with d = 3 (independent of n), C_3 = K_3, cycle vs complete;
  S_n : centre/leaf equilibrium, the centre-periphery gap (general n),
        ordering against K_n and C_n, the leaf ceiling (closed form), the
        mixed regime (leaves at e, centre interior), the centre ceiling,
        contraction, comparative statics;
  numerical: least and greatest equilibria coincide (uniqueness) on a grid.
"""

import random

import numpy as np
import sympy as sp

from common import EXP, Log, B, beta, e, n, positive_on, rat, same, simp

L = Log(__file__)
c, cC, cL = sp.symbols("c c_C c_L", positive=True)


def theta(dd):
    return 1 - beta / dd


def BR(dd, S):
    return B / theta(dd) * (e + beta / dd * S)


# ===========================================================================
L.head("1. Complete network K_n (no network)")
# ===========================================================================
cK = simp(sp.solve(sp.Eq(c, BR(n, (n - 1) * c)), c)[0])
L.show("c*_K", cK)
DK = n - beta - B * beta * (n - 1)
L.check("c*_K = B n e / (n - beta - B beta (n-1))", same(cK, B * n * e / DK))
L.tex_eq("c*_K", r"c^\star_{K}", cK)

# existence: denominator > 0
bexist = sp.solve(sp.Eq(DK, 0), B)[0]
L.show("existence bound on B (D_K > 0)", simp(bexist))
# interiority ceiling: c* = e
BK = simp(sp.solve(sp.Eq(cK, e), B)[0])
L.show("interiority ceiling B^K(beta,n)", BK)
L.check("B^K = (n - beta)/(n + beta (n-1))", same(BK, (n - beta) / (n + beta * (n - 1))))
L.tex_eq("ceiling K_n", r"\bar B^{K}", BK)


def sK():
    nn = sp.Integer(random.randint(2, 12))
    bb = rat(1, nn)
    return {n: nn, beta: bb, e: rat(1, 40), B: rat(0, (nn - bb) / (nn + bb * (nn - 1)))}


L.check("ceiling < existence bound  (interiority implies D_K > 0)",
        positive_on(simp(bexist - BK), sK))
# contraction: rho(Gamma A) = (n-1) gamma, gamma = B beta/(n-beta)
L.check("interiority implies contraction (n-1) B beta/(n-beta) < 1  => unique equilibrium",
        positive_on(1 - (n - 1) * B * beta / (n - beta), sK))
# full corner: if B >= B^K, c = e for all is an equilibrium (BR at S=(n-1)e is >= e)
corner_gap = simp(BR(n, (n - 1) * e) - e)
L.show("BR(S=(n-1)e) - e", corner_gap)
L.check("BR(S=(n-1)e) >= e  <=>  B >= B^K  (full-corner regime above the ceiling)",
        same(sp.solve(sp.Eq(corner_gap, 0), B)[0], BK))

# comparative statics
for var, name in ((B, "B"), (e, "e"), (beta, "beta"), (n, "n")):
    der = simp(sp.diff(cK, var))
    L.show(f"dc*_K/d{name}", der)
dB_K = simp(sp.diff(cK, B))
L.check("dc*_K/dB = n e (n-beta)/D^2 > 0", same(dB_K, n * e * (n - beta) / DK**2) and positive_on(dB_K, sK))
L.check("dc*_K/de = c*/e > 0 (contributions proportional to e)", same(sp.diff(cK, e), cK / e))
dbeta_K = simp(sp.diff(cK, beta))
L.check("dc*_K/dbeta = B n e (1 + B(n-1))/D^2 > 0", same(dbeta_K, B * n * e * (1 + B * (n - 1)) / DK**2))
dn_K = simp(sp.diff(cK, n))
L.check("dc*_K/dn = -B (1-B) beta e / D^2 < 0  (group-size effect)", same(dn_K, -B * (1 - B) * beta * e / DK**2)
        and positive_on(-dn_K, sK))
L.tex_eq("dc/dn K", r"\partial c^\star_K/\partial n", dn_K)
L.check("d ceiling/d beta = -n^2/(n+beta(n-1))^2 < 0", same(sp.diff(BK, beta), -n**2 / (n + beta * (n - 1))**2))
L.check("d ceiling/d n = beta^2/(n+beta(n-1))^2 > 0", same(sp.diff(BK, n), beta**2 / (n + beta * (n - 1))**2))
L.check("lim_{n->oo} c*_K = B e/(1 - B beta)  (finite large-group limit)",
        same(sp.limit(cK, n, sp.oo), B * e / (1 - B * beta)))

# ===========================================================================
L.head("2. Cycle C_n (d_i = 3 for every n)")
# ===========================================================================
cCyc = simp(sp.solve(sp.Eq(c, BR(3, 2 * c)), c)[0])
L.show("c*_cyc", cCyc)
L.check("c*_cyc = 3 B e/(3 - beta - 2 B beta), independent of n", same(cCyc, 3 * B * e / (3 - beta - 2 * B * beta)))
L.check("C_3 = K_3: c*_cyc = c*_K at n = 3", same(cCyc, cK.subs(n, 3)))
Bcyc = simp(sp.solve(sp.Eq(cCyc, e), B)[0])
L.show("ceiling B^cyc", Bcyc)
L.check("B^cyc = (3 - beta)/(3 + 2 beta)", same(Bcyc, (3 - beta) / (3 + 2 * beta)))
L.tex_eq("c*_cyc", r"c^\star_{C}", cCyc)


def sCyc():
    bb = rat(1, 3)
    return {beta: bb, e: rat(1, 40), B: rat(0, (3 - bb) / (3 + 2 * bb)), n: sp.Integer(random.randint(4, 12))}


L.check("interiority implies contraction 2 gamma = 2 B beta/(3-beta) < 1", positive_on(1 - 2 * B * beta / (3 - beta), sCyc))


def sCK():  # admissible for both C_n and K_n with n >= 4: 1 < beta < 3, B below both ceilings
    nn = sp.Integer(random.randint(4, 12))
    bb = rat(1, 3)
    return {n: nn, beta: bb, e: rat(1, 40), B: rat(0, min((3 - bb) / (3 + 2 * bb), (nn - bb) / (nn + bb * (nn - 1))))}


L.check("n > 3: cycle member contributes more than a complete-network member (smaller d)",
        positive_on(simp(cCyc - cK), sCK))

# ===========================================================================
L.head("3. Star S_n, common weight B: centre (d=n) and n-1 leaves (d=2)")
# ===========================================================================
sol = sp.solve([sp.Eq(cC, BR(n, (n - 1) * cL)), sp.Eq(cL, BR(2, cC))], [cC, cL], dict=True)[0]
cCs, cLs = simp(sol[cC]), simp(sol[cL])
L.show("c*_centre", cCs)
L.show("c*_leaf", cLs)
DS = (n - beta) * (2 - beta) - (n - 1) * B**2 * beta**2
L.check("common denominator Delta_S = (n-beta)(2-beta) - (n-1) B^2 beta^2",
        same(cCs * DS / (B * e), simp(cCs * DS / (B * e))) and sp.simplify(sp.denom(sp.together(cCs)) / DS).is_number)
L.check("c*_centre = B e [n(2-beta) + 2(n-1) B beta] / Delta_S",
        same(cCs, B * e * (n * (2 - beta) + 2 * (n - 1) * B * beta) / DS))
L.check("c*_leaf   = B e [2(n-beta) + n B beta] / Delta_S",
        same(cLs, B * e * (2 * (n - beta) + n * B * beta) / DS))
L.tex_eq("c*_centre", r"c^\star_{S,C}", cCs)
L.tex_eq("c*_leaf", r"c^\star_{S,L}", cLs)
L.check("contraction rho(Gamma A)^2 = (n-1) gamma_C gamma_L < 1  <=>  Delta_S > 0",
        same(1 - (n - 1) * (B * beta / (n - beta)) * (B * beta / (2 - beta)), DS / ((n - beta) * (2 - beta))))

gap = simp(cLs - cCs)
L.show("c*_leaf - c*_centre", gap)
L.check("gap = (n-2) B (1-B) beta e / Delta_S", same(gap, (n - 2) * B * (1 - B) * beta * e / DS))
L.check("n = 3: gap = A B beta e/(beta^2 - 5 beta + 6 - 2 B^2 beta^2)  [thesis Prop. star_order]",
        same(gap.subs(n, 3), (1 - B) * B * beta * e / (beta**2 - 5 * beta + 6 - 2 * B**2 * beta**2)))
L.tex_eq("gap", r"c^\star_{S,L}-c^\star_{S,C}", gap)

# leaf ceiling: c*_leaf = e (leaves reach the corner first)
rootsL = sp.solve(sp.Eq(cLs, e), B)
rootsL = [simp(r) for r in rootsL]
L.say("  roots of c*_leaf = e in B:", [sp.sstr(r) for r in rootsL])
BL = [r for r in rootsL if sp.N(r.subs({n: 3, beta: sp.Rational(3, 2)})) > 0][0]
L.show("leaf ceiling B^S_L(beta,n)", BL)
L.check("n=3, beta=3/2: leaf ceiling = 1/6", sp.nsimplify(BL.subs({n: 3, beta: sp.Rational(3, 2)})) == sp.Rational(1, 6))
L.tex_eq("leaf ceiling", r"\bar B^{S}_{L}", BL)

# mixed regime: leaves at e, centre interior
cC_mix = simp(BR(n, (n - 1) * e))
L.show("centre contribution when leaves are at e", cC_mix)
BCmix = simp(sp.solve(sp.Eq(cC_mix, e), B)[0])
L.show("centre ceiling (mixed regime)", BCmix)
L.check("centre ceiling in the mixed regime equals the K_n ceiling (n-beta)/(n+beta(n-1))", same(BCmix, BK))


def sS():  # star, admissible: 1 < beta < 2, B below the leaf ceiling
    nn = sp.Integer(random.randint(3, 10))
    bb = rat(1, 2)
    up = sp.N(BL.subs({n: nn, beta: bb}), 30)
    return {n: nn, beta: bb, e: rat(1, 40), B: sp.nsimplify(rat(0, 1) * sp.Rational(str(up)[:12]))}


L.check("on the fully interior region: Delta_S > 0 (contraction holds)", positive_on(DS, sS))
B_rho = sp.solve(sp.Eq(DS.subs(EXP), 0), B)
B_rho = [r for r in B_rho if r.is_positive][0]
L.show("experiment: Delta_S > 0 iff B <", sp.N(B_rho, 6))
L.check("experiment: contraction (Delta_S > 0) holds for B < 0.408 > 1/4, i.e. across all three regimes up to the full corner",
        abs(sp.N(B_rho) - 0.40825) < 1e-4 and B_rho > sp.Rational(1, 4))
L.check("on the fully interior region: leaf > centre (gap > 0)", positive_on(gap, sS))
L.check("leaf ceiling < centre ceiling (leaves corner first)",
        positive_on(BK - BL, lambda: {k: v for k, v in sS().items() if k != B}))
L.check("on the fully interior region: centre > complete-network member with the same n",
        positive_on(simp(cCs - cK), sS))
L.check("closed form: c*_centre - c*_K = (n-1)(n-2) B^2 (1-B) beta^2 e / (D_K Delta_S)",
        same(cCs - cK, (n - 1) * (n - 2) * B**2 * (1 - B) * beta**2 * e / (DK * DS)))
L.tex_eq("centre minus K_n member", r"c^\star_{S,C}-c^\star_{K}", simp(cCs - cK))
# the leaf ceiling in readable form: positive root of beta(n+beta(n-1)) B^2 + 2(n-beta) B - (n-beta)(2-beta) = 0
BL_read = (-(n - beta) + sp.sqrt((n - beta) * ((n - beta) + beta * (2 - beta) * (n + beta * (n - 1))))) / (beta * (n + beta * (n - 1)))
L.check("leaf ceiling = [-(n-beta) + sqrt((n-beta)((n-beta) + beta(2-beta)(n+beta(n-1))))] / [beta(n+beta(n-1))]",
        all(abs(sp.N((BL - BL_read).subs({n: m, beta: bb}))) < 1e-25
            for m in range(3, 9) for bb in (sp.Rational(11, 10), sp.Rational(3, 2), sp.Rational(19, 10))))
L.check("leaf ceiling solves beta(n+beta(n-1)) B^2 + 2(n-beta) B - (n-beta)(2-beta) = 0",
        all(abs(sp.N((beta * (n + beta * (n - 1)) * B**2 + 2 * (n - beta) * B - (n - beta) * (2 - beta))
                     .subs(B, BL).subs({n: m, beta: sp.Rational(3, 2)}))) < 1e-25 for m in range(3, 9)))
for var, name in ((B, "B"), (beta, "beta")):
    L.check(f"dc*_centre/d{name} > 0 and dc*_leaf/d{name} > 0 on the interior region",
            positive_on(simp(sp.diff(cCs, var)), sS) and positive_on(simp(sp.diff(cLs, var)), sS))
L.check("d gap/dB > 0 on the interior region (gap widens with the prosocial weight)",
        positive_on(simp(sp.diff(gap, B)), sS))

# centre vs cycle member for n > 3 (different d): sign is not uniform
diff_cc = simp(cCs - cCyc)
signs = set()
for _ in range(400):
    p = sS()
    p[n] = sp.Integer(random.randint(4, 10))
    up = min(sp.N(BL.subs({n: p[n], beta: p[beta]})), sp.N(Bcyc.subs(beta, p[beta])))
    p[B] = sp.nsimplify(up * rat(0, 1))
    signs.add(bool(sp.N(diff_cc.subs(p)) > 0))
L.say(f"  sign(centre(S_n) - member(C_n)) for n >= 4 takes values {signs}")
L.check("for n >= 4 the centre-vs-cycle ranking is parameter dependent (H4 ordering is an n=3 result)", signs == {True, False})

# ===========================================================================
L.head("4. Experimental values (e = 15, n = 3, beta = 3/2)")
# ===========================================================================
L.say("  ceilings: B^K = B^cyc = B^S_C =", BK.subs(EXP), "; B^S_L =", sp.nsimplify(BL.subs(EXP)))
L.say("    B   | regular | star centre | star leaf | leaf - centre")
for Bv in (sp.Rational(1, 20), sp.Rational(1, 10), sp.Rational(3, 20)):
    vals = [sp.N(f.subs(EXP).subs(B, Bv), 6) for f in (cK, cCs, cLs, gap)]
    L.say(f"  {float(Bv):.2f}  | " + " | ".join(f"{float(v):9.4f}" for v in vals))
L.check("B=0.10: regular 3.75, centre 4.4681, leaf 7.3404",
        abs(sp.N(cK.subs(EXP).subs(B, sp.Rational(1, 10))) - 3.75) < 1e-12
        and abs(sp.N(cCs.subs(EXP).subs(B, sp.Rational(1, 10))) - 4.468085106) < 1e-8
        and abs(sp.N(cLs.subs(EXP).subs(B, sp.Rational(1, 10))) - 7.340425532) < 1e-8)
L.check("H4 ordering at n=3: leaf > centre > regular on the interior region",
        positive_on(simp((cLs - cCs).subs(n, 3)), lambda: {**sS(), n: 3})
        and positive_on(simp((cCs - cK).subs(n, 3)), lambda: {**sS(), n: 3}))

# ===========================================================================
L.head("5. Numerical uniqueness: least (from 0) and greatest (from e) equilibria coincide")
# ===========================================================================
def adj(kind, m):
    if kind == "K":
        return np.ones((m, m)) - np.eye(m)
    if kind == "C":
        a = np.zeros((m, m))
        for i in range(m):
            a[i, (i + 1) % m] = a[i, (i - 1) % m] = 1
        return a
    a = np.zeros((m, m)); a[0, 1:] = a[1:, 0] = 1
    return a


def eq(adjm, Bv, bv, ev, start):
    dd = adjm.sum(1) + 1
    th = 1 - bv / dd
    x = np.full(len(dd), start, float)
    for _ in range(100000):
        xn = np.minimum(ev, Bv / th * (ev + bv / dd * (adjm @ x)))
        if np.max(np.abs(xn - x)) < 1e-12:
            return xn
        x = xn
    return xn


worst = 0.0
for kind in ("K", "C", "S"):
    for m in (3, 4, 5, 6):
        a = adj(kind, m)
        bmax = 2.0 if kind == "S" else min(3.0, float(m)) if kind == "C" else float(m)
        for bv in np.linspace(1.05, bmax - 0.05, 8):
            for Bv in np.linspace(0.01, 0.6, 15):
                lo, hi = eq(a, Bv, bv, 15.0, 0.0), eq(a, Bv, bv, 15.0, 15.0)
                worst = max(worst, float(np.max(np.abs(lo - hi))))
L.say(f"  max |c_least - c_greatest| over the grid = {worst:.2e}")
L.check("unique equilibrium (incl. corner and mixed regimes) on the grid", worst < 1e-8)

L.finish()
