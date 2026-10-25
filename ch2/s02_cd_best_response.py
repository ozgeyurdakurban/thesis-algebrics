"""
Step 1b -- The Cobb-Douglas individual problem at an arbitrary network position.

Agent i at a position with closed-neighbourhood size d (theta = 1 - beta/d),
impact multiplier Lambda, and neighbours' total contribution S = sum_{j in G_i} c_j:

    X = e - theta c + (beta/d) S  =  W - theta c ,   W = e + (beta/d) S  ("full income")
    Y = Lambda c
    U = X^A Y^B ,   A, B > 0.

Derived here (no formula typed in):
  * FOC, its solution, and the equivalence FOC <=> c = kappa X, kappa = B/(A theta);
  * the best response as a Cobb-Douglas demand: c = b W / theta, b = B/(A+B);
    hence BR = alpha + gamma S with alpha = b e/theta, gamma = b beta/(d theta);
  * irrelevance of Lambda (it scales the moral good, not its price);
  * SOC (strict log-concavity), absence of the lower corner, the upper-corner rule;
  * strategic complementarity, scale invariance in (A,B), monotonicity in B.

Thesis numbering (version of 25 Oct 2026): Lemma 2.4.6, Remark 2.4.7 (the sigma = 1 reference
specification); Remark 2.3.1 (normalization). Lambda here stands for any positive conversion
factor, so the neutrality result covers the effective multiplier Lambda~ for every impact weight.
"""

import random

import sympy as sp

from common import Lam, Log, A, B, beta, d, e, rat, same, simp

L = Log(__file__)
c = sp.symbols("c", positive=True)
S = sp.symbols("S", nonnegative=True)      # neighbours' total contribution
t = sp.symbols("t", positive=True)

theta = 1 - beta / d
W = e + beta / d * S
X = W - theta * c
Y = Lam * c
logU = A * sp.log(X) + B * sp.log(Y)


def sampler():
    dd = sp.Integer(random.randint(2, 9))
    return {d: dd, beta: rat(1, dd), e: rat(1, 30), S: rat(0, 40), A: rat(0, 3), B: rat(0, 3),
            Lam: rat(1, 6), c: rat(0, 5)}


# ---------------------------------------------------------------------------
L.head("1. First-order condition and its solution")
# ---------------------------------------------------------------------------
foc = sp.diff(logU, c)
L.show("d log U / dc", sp.simplify(foc))
L.check("FOC does not depend on Lambda (d/dLambda of FOC = 0)", sp.simplify(sp.diff(foc, Lam)) == 0)

sol = sp.solve(sp.Eq(foc, 0), c)
L.check("FOC has a unique root", len(sol) == 1)
c_br = simp(sol[0])
L.show("BR (derived)", c_br)

b = B / (A + B)
L.check("BR = b W / theta  with b = B/(A+B)   [Cobb-Douglas demand]", same(c_br, b * W / theta, sampler))

kappa = B / (A * theta)
L.check("FOC <=> c = kappa X, kappa = B/(A theta)",
        same(sp.solve(sp.Eq(c, kappa * X), c)[0], c_br, sampler))
L.check("thesis form kappa(e+(beta/d)S)/(1+kappa theta) is the same BR",
        same(kappa * W / (1 + kappa * theta), c_br, sampler))

X_opt = simp(X.subs(c, c_br))
L.show("X at the optimum", X_opt)
L.check("material payoff at the optimum = (A/(A+B)) W  (expenditure share of X)", same(X_opt, A / (A + B) * W, sampler))
L.check("moral good at the optimum Y = Lambda b W/theta (Lambda scales Y, not c)",
        same(Y.subs(c, c_br), Lam * b * W / theta, sampler))

# Linear form BR = alpha + gamma S
alpha = simp(c_br.subs(S, 0))
gamma = simp(sp.diff(c_br, S))
L.show("alpha", alpha)
L.show("gamma", gamma)
L.check("BR is affine in S: BR = alpha + gamma S", same(c_br, alpha + gamma * S, sampler))
L.check("alpha = b e / theta", same(alpha, b * e / theta, sampler))
L.check("gamma = b beta/(d theta) = b beta/(d - beta)", same(gamma, b * beta / (d - beta), sampler))
L.tex_eq("alpha", r"\alpha_i", (B / (A + B)) * e / sp.Symbol(r"\theta_i"))
L.tex_eq("gamma", r"\gamma_i", (B / (A + B)) * beta / (sp.Symbol("d_i") - beta))

# ---------------------------------------------------------------------------
L.head("2. Second-order condition and boundary behaviour")
# ---------------------------------------------------------------------------
soc = simp(sp.diff(logU, c, 2))
L.show("d^2 log U / dc^2", soc)
L.check("SOC = -A theta^2/X^2 - B/c^2", same(soc, -A * theta**2 / X**2 - B / c**2, sampler))
L.check("SOC < 0 on the feasible region (strict log-concavity)",
        all(sp.N(soc.subs(p)) < 0 for p in [sampler() for _ in range(300)]))
lim0 = sp.limit(foc.subs({A: 1, B: sp.Rational(1, 5), Lam: 2, d: 3, beta: sp.Rational(3, 2), e: 15, S: 4}), c, 0, "+")
L.check("lim_{c->0+} dlogU/dc = +oo  => no lower corner (c* > 0 always)", lim0 == sp.oo)
# Upper corner: the truncated BR is min{e, BR}; BR >= e  <=>  b W >= theta e
cond = sp.simplify(c_br - e)
L.check("BR >= e  <=>  b (e + (beta/d) S) >= theta e", same(cond * theta, b * W - theta * e, sampler))

# ---------------------------------------------------------------------------
L.head("3. Properties of the best response")
# ---------------------------------------------------------------------------
L.check("strategic complements: dBR/dS = gamma > 0 (d > beta)", sp.simplify(gamma.subs(d, beta + t)).is_positive)
L.check("scale invariance: (A,B) -> (tA,tB) leaves BR unchanged",
        sp.simplify(c_br.subs({A: t * A, B: t * B}, simultaneous=True) - c_br) == 0)
L.check("Lambda-irrelevance: dBR/dLambda = 0", sp.diff(c_br, Lam) == 0)
cbB = simp(c_br.subs(A, 1 - B))
L.show("BR under A = 1 - B", cbB)
L.check("under A+B=1: BR = (B/theta)(e + (beta/d) S)  -- linear in the prosocial weight",
        same(cbB, B / theta * W, sampler))
L.check("dBR/dB > 0 (A=1-B)", sp.simplify(sp.diff(cbB, B) * theta / W) == 1)
L.check("d alpha/dB > 0 and d gamma/dB > 0 (general A,B: via b = B/(A+B))",
        sp.simplify(sp.diff(b, B) - A / (A + B) ** 2) == 0)
L.check("dBR/dd < 0: a larger neighbourhood lowers the BR at given S, e (theta up, share down)",
        all(sp.N(sp.diff(c_br, d).subs(p)) < 0 for p in [sampler() for _ in range(300)]))
L.tex_eq("BR under A+B=1", r"\mathrm{BR}_i", (B / sp.Symbol(r"\theta_i")) * (e + beta / sp.Symbol("d_i") * sp.Symbol("S_i")))

L.finish()
