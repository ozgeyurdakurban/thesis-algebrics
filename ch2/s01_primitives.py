"""
Step 1a -- Model primitives on a general network (Chapter 2, Section "Model").

Derives, for an arbitrary undirected network g without self-loops:
  * the material payoff  X_i = e - theta_i c_i + (beta/d_i) sum_{j in G_i} c_j,
    theta_i = 1 - beta/d_i;
  * the impact-written moral component Y_i = Lambda_i c_i,
    Lambda_i = 1 + sum_{j in G_i} beta/d_j  (impact multiplier);
  * social efficiency SE = sum_i X_i = n e + sum_i (s_i - 1) c_i,
    s_i = sum_{k in N[i]} beta/d_k  (total marginal social return);
  * the identity Lambda_i = 1 + s_i - beta/d_i linking the two;
  * the closed forms of (d_i, theta_i, Lambda_i, s_i) on K_n, C_n, S_n for general n,
    and the dilemma / social-productivity bands for beta.

Everything is derived from the adjacency matrix of explicit graphs (n = 3..8)
and from the general-n structure, and the two are cross-checked.

Thesis numbering (version of 25 Oct 2026): eq. (2.18) and the positional primitives of Section 2.3;
the effective impact multiplier Lambda~ = 1 + lambda (Lambda - 1) is treated in s10.
"""

import sympy as sp

from common import EXP, Log, beta, e, n, simp

L = Log(__file__)


# ---------------------------------------------------------------------------
# Graph constructors (explicit adjacency matrices)
# ---------------------------------------------------------------------------
def complete(m):
    return sp.Matrix(m, m, lambda i, j: 0 if i == j else 1)


def cycle(m):
    return sp.Matrix(m, m, lambda i, j: 1 if (i - j) % m in (1, m - 1) else 0)


def star(m):  # node 0 = center
    return sp.Matrix(m, m, lambda i, j: 1 if (i != j and (i == 0 or j == 0)) else 0)


def primitives(adj):
    """Return per-node d, theta, Lambda, s and the symbolic X, Y, SE."""
    m = adj.shape[0]
    cs = sp.symbols(f"c0:{m}", positive=True)
    G = [[j for j in range(m) if adj[i, j] == 1] for i in range(m)]
    dd = [len(G[i]) + 1 for i in range(m)]
    X = [e - cs[i] + beta / dd[i] * (cs[i] + sum(cs[j] for j in G[i])) for i in range(m)]
    Y = [cs[i] + sum(beta / dd[j] for j in G[i]) * cs[i] for i in range(m)]
    theta = [1 - beta / dd[i] for i in range(m)]
    Lam = [1 + sum(beta / dd[j] for j in G[i]) for i in range(m)]
    s = [beta / dd[i] + sum(beta / dd[j] for j in G[i]) for i in range(m)]
    return dict(c=cs, G=G, d=dd, X=X, Y=Y, theta=theta, Lam=Lam, s=s, m=m)


# ---------------------------------------------------------------------------
L.head("1. Identities that hold on EVERY network (checked on K_n, C_n, S_n, n=3..8)")
# ---------------------------------------------------------------------------
ok_X = ok_Y = ok_SE = ok_link = True
for build in (complete, cycle, star):
    for m in range(3, 9):
        P = primitives(build(m))
        cs, G = P["c"], P["G"]
        for i in range(m):
            # X_i = e - theta_i c_i + (beta/d_i) sum_{G_i} c_j
            rhs = e - P["theta"][i] * cs[i] + beta / P["d"][i] * sum(cs[j] for j in G[i])
            ok_X &= sp.simplify(P["X"][i] - rhs) == 0
            # Y_i = Lambda_i c_i
            ok_Y &= sp.simplify(P["Y"][i] - P["Lam"][i] * cs[i]) == 0
            # Lambda_i = 1 + s_i - beta/d_i
            ok_link &= sp.simplify(P["Lam"][i] - (1 + P["s"][i] - beta / P["d"][i])) == 0
        # SE = n e + sum (s_i - 1) c_i, derived by collecting coefficients
        SE = sp.expand(sum(P["X"]))
        derived = {cs[i]: sp.simplify(SE.coeff(cs[i])) for i in range(m)}
        ok_SE &= all(sp.simplify(derived[cs[i]] - (P["s"][i] - 1)) == 0 for i in range(m))
        ok_SE &= sp.simplify(SE.subs({ci: 0 for ci in cs}) - m * e) == 0

L.check("X_i = e - theta_i c_i + (beta/d_i) sum_{j in G_i} c_j", ok_X)
L.check("Y_i = Lambda_i c_i with Lambda_i = 1 + sum_{j in G_i} beta/d_j", ok_Y)
L.check("SE = sum_i X_i = n e + sum_i (s_i - 1) c_i  (coefficients derived)", ok_SE)
L.check("Lambda_i = 1 + s_i - beta/d_i  (impact multiplier = 1 + external social return)", ok_link)

# ---------------------------------------------------------------------------
L.head("2. Closed forms by topology for general n (derived from the structure)")
# ---------------------------------------------------------------------------
# K_n: every node has n-1 neighbours, each with d=n.
K = dict(d=n, theta=1 - beta / n, Lam=1 + (n - 1) * beta / n, s=beta / n + (n - 1) * beta / n)
# C_n: two neighbours, each with d=3.
C = dict(d=3, theta=1 - beta / 3, Lam=1 + 2 * beta / 3, s=beta / 3 + 2 * beta / 3)
# S_n: center has n-1 leaves (d=2); a leaf has one neighbour, the center (d=n).
SC = dict(d=n, theta=1 - beta / n, Lam=1 + (n - 1) * beta / 2, s=beta / n + (n - 1) * beta / 2)
SL = dict(d=2, theta=1 - beta / 2, Lam=1 + beta / n, s=beta / 2 + beta / n)

ok = True
for m in range(3, 9):
    for build, forms in ((complete, [K]), (cycle, [C]), (star, [SC, SL])):
        P = primitives(build(m))
        nodes = [0] if build is not star else [0, 1]
        for node, F in zip(nodes, forms):
            for key in ("d", "theta", "Lam", "s"):
                ok &= sp.simplify(sp.sympify(P[key][node]) - sp.sympify(F[key]).subs(n, m)) == 0
L.check("general-n closed forms reproduce the explicit graphs for n = 3..8", ok)

for name, F in (("K_n", K), ("C_n", C), ("S_n center", SC), ("S_n leaf", SL)):
    L.say(f"  {name:11s}: d={sp.sstr(F['d'])}, theta={sp.sstr(simp(F['theta']))}, "
          f"Lambda={sp.sstr(simp(F['Lam']))}, s={sp.sstr(simp(F['s']))}")
L.check("s_i = beta on regular graphs (K_n, C_n)", simp(K["s"]) == beta and simp(C["s"]) == beta)

L.tex_eq("Lambda K_n", r"\Lambda^{K}", simp(K["Lam"]))
L.tex_eq("Lambda C_n", r"\Lambda^{C}", simp(C["Lam"]))
L.tex_eq("Lambda star center", r"\Lambda_{C}^{S}", simp(SC["Lam"]))
L.tex_eq("Lambda star leaf", r"\Lambda_{L}^{S}", simp(SL["Lam"]))
L.tex_eq("s star center", r"s_{C}", simp(SC["s"]))
L.tex_eq("s star leaf", r"s_{L}", simp(SL["s"]))

# Ordering of the impact multiplier on the star
gapLam = simp(SC["Lam"] - SL["Lam"])
L.show("Lambda_C - Lambda_L", gapLam)
L.check("Lambda_C > Lambda_L on S_n for n >= 3 (since (n-1)/2 - 1/n > 0)",
        all(gapLam.subs({n: m, beta: sp.Rational(3, 2)}) > 0 for m in range(3, 30)))

# ---------------------------------------------------------------------------
L.head("3. Admissible beta: dilemma (1<beta<d_i) and social productivity (s_i>1)")
# ---------------------------------------------------------------------------
bL = sp.solve(sp.Eq(SL["s"], 1), beta)[0]
bC = sp.solve(sp.Eq(SC["s"], 1), beta)[0]
L.show("leaf social-productivity floor  beta_L", simp(bL))
L.show("center social-productivity floor beta_C", simp(bC))
L.check("leaf floor = 2n/(n+2)", sp.simplify(bL - 2 * n / (n + 2)) == 0)
L.check("center floor = 2n/(n^2-n+2)", sp.simplify(bC - 2 * n / (n**2 - n + 2)) == 0)
L.check("n=3: leaf floor 6/5, center floor 3/4", bL.subs(n, 3) == sp.Rational(6, 5) and bC.subs(n, 3) == sp.Rational(3, 4))
L.check("leaf floor is the binding one for n>=3 (beta_L >= 1 >= beta_C)",
        all(bL.subs(n, m) >= 1 >= bC.subs(n, m) for m in range(3, 40)))
L.check("star beta-band (2n/(n+2), 2) is non-empty for all n but shrinks to 0 as n->oo",
        sp.limit(2 - bL, n, sp.oo) == 0 and all(2 - bL.subs(n, m) > 0 for m in range(3, 40)))
L.tex_eq("leaf floor", r"\underline{\beta}_{L}", simp(bL))

# ---------------------------------------------------------------------------
L.head("4. Experimental values (e=15, n=3, beta=3/2)")
# ---------------------------------------------------------------------------
for name, F in (("K_3=C_3", K), ("S_3 center", SC), ("S_3 leaf", SL)):
    vals = {k: sp.nsimplify(sp.sympify(F[k]).subs(EXP)) for k in ("d", "theta", "Lam", "s")}
    L.say(f"  {name:11s}: d={vals['d']}, theta={vals['theta']}, MPCR={sp.nsimplify(1 - vals['theta'])}, "
          f"Lambda={vals['Lam']}, s={vals['s']}")
L.check("experiment: theta=1/2 at d=3, theta=1/4 at d=2",
        K["theta"].subs(EXP) == sp.Rational(1, 2) and SL["theta"].subs(EXP) == sp.Rational(1, 4))
L.check("experiment: s_C = 2, s_L = 5/4 (>1), s_reg = 3/2",
        SC["s"].subs(EXP) == 2 and SL["s"].subs(EXP) == sp.Rational(5, 4) and K["s"].subs(EXP) == sp.Rational(3, 2))

L.finish()
