"""
Step 5a -- The additive-baseline warm-glow benchmark, U = X^A (T + c)^B, on K_n.

Derived here:
  * FOC: T + c = kappa X,  kappa = B/(A theta)   (the baseline survives the FOC);
  * common weight: c*_WG = (B n e - A (n-beta) T)/D_K and c*_IB - c*_WG = A (n-beta) T/D_K;
    crowding-out d c*_WG/dT < 0; same existence condition;
  * heterogeneous weights: c_i*_WG = phi_i (e - (beta/n) T Q)/mu - T/(1+kappa_i),
    Q = sum_j 1/(1+kappa_j), and the agent-specific gap (Appendix C);
  * n = 2: (gap_1 - gap_2)/(B_1 - B_2) closed form.

Thesis numbering (version of 25 Oct 2026): Proposition 2.4.29 and Appendix C (warm-glow section);
the act-based specification (lambda = 0) is the T = 0 case.
"""

import sympy as sp

from common import Log, beta, e, n, same, simp, T

L = Log(__file__)
A, B, c, S = sp.symbols("A B c S", positive=True)
th = 1 - beta / n

# FOC
X = e - th * c + beta / n * S
logU = A * sp.log(X) + B * sp.log(T + c)
sol = simp(sp.solve(sp.Eq(sp.diff(logU, c), 0), c)[0])
kap = B / (A * th)
L.check("warm-glow FOC: T + c = kappa X (baseline survives)", same(sol, sp.solve(sp.Eq(T + c, kap * X), c)[0]))

# common weight on K_n (A = 1 - B)
cs = sp.symbols("cs")
symm = simp(sp.solve(sp.Eq(cs, sol.subs(S, (n - 1) * cs)), cs)[0].subs(A, 1 - B))
DK = n - beta - B * beta * (n - 1)
L.check("c*_WG = (B n e - A (n-beta) T)/D_K", same(symm, (B * n * e - (1 - B) * (n - beta) * T) / DK))
L.check("c*_IB - c*_WG = A (n-beta) T/D_K", same(B * n * e / DK - symm, (1 - B) * (n - beta) * T / DK))
L.check("crowding-out: d c*_WG/dT = -A (n-beta)/D_K", same(sp.diff(symm, T), -(1 - B) * (n - beta) / DK))

# heterogeneous weights, n = 2 and 3: direct solve vs Appendix C formula
for m in (2, 3):
    Bs = sp.symbols(f"B1:{m+1}", positive=True)
    xs = sp.symbols(f"x1:{m+1}")
    thm = 1 - beta / m
    kaps = [b / ((1 - b) * thm) for b in Bs]
    eqs = [sp.Eq(T + xs[i], kaps[i] * (e - thm * xs[i] + beta / m * (sum(xs) - xs[i]))) for i in range(m)]
    sl = sp.solve(eqs, xs, dict=True)[0]
    phis = [k / (1 + k) for k in kaps]
    mu = 1 - beta / m * sum(phis)
    Q = sum(1 / (1 + k) for k in kaps)
    ok = all(sp.simplify(sl[xs[i]] - (phis[i] * (e - beta / m * T * Q) / mu - T / (1 + kaps[i]))) == 0 for i in range(m))
    L.check(f"n = {m}: c_i*_WG = phi_i (e - (beta/n) T Q)/mu - T/(1 + kappa_i)", ok)
    gap = [simp(e * phis[i] / mu - sl[xs[i]]) for i in range(m)]
    ok = all(sp.simplify(gap[i] - T * (beta / m * phis[i] * Q / mu + 1 / (1 + kaps[i]))) == 0 for i in range(m))
    L.check(f"n = {m}: gap_i = T((beta/n) phi_i Q/mu + 1/(1+kappa_i))", ok)
    if m == 2:
        r = simp((gap[0] - gap[1]) / (Bs[0] - Bs[1]))
        L.show("n=2: (gap_1 - gap_2)/(B_1 - B_2)", r)
        L.check("n=2: ratio = 2T(beta-2)(beta-1)/(B_1 B_2 beta^2 - beta^2 + 4 beta - 4)",
                same(r, 2 * T * (beta - 2) * (beta - 1) / (Bs[0] * Bs[1] * beta**2 - beta**2 + 4 * beta - 4)))
L.check("n=2, beta=3/2: denominator < 0 iff B_1 B_2 < 1/9; the K_2 ceiling is 1/7, so the ratio is positive on the interior",
        sp.solve(sp.Eq(sp.Symbol("P") * sp.Rational(9, 4) - sp.Rational(9, 4) + 6 - 4, 0), sp.Symbol("P"))[0] == sp.Rational(1, 9)
        and sp.Rational(1, 7) ** 2 < sp.Rational(1, 9))

L.finish()
