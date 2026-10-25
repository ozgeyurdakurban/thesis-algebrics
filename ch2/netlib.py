"""
Numerical companion to the symbolic scripts (numpy only).

All equilibrium computations of Chapter 2 reduce to one object: the share of
full income an agent devotes to giving,

    Cobb-Douglas : pi_i = B_i                                 (A_i + B_i = 1)
    CES          : pi_i = 1 / (1 + ((1-B_i)/B_i)^sigma * p_i^(sigma-1)),
                   p_i = theta_i / kappa_i,  kappa_i = 1 + lambda_i (Lambda_i - 1),
                   lambda_i >= 0 the impact weight (lambda = 1: impact-based,
                   lambda = 0: act-based; `impact=True/False` is kept as a shortcut),

and the best response c_i = min{e, (pi_i/theta_i) (e + (beta/d_i) S_i)}.

Two independent solvers are provided:
  * nash_closed   : the linear-algebra solution on a given set of cornered agents
                    (active-set iteration), using the formulas derived in the paper;
  * nash_utility  : FORMULA-FREE. Each best response is found by maximising the
                    actual utility function U_i(c_i) on [0, e] (golden section on
                    log U, which is strictly concave), iterated to a fixed point.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
def complete(n):
    return np.ones((n, n)) - np.eye(n)


def cycle(n):
    a = np.zeros((n, n))
    for i in range(n):
        a[i, (i + 1) % n] = a[i, (i - 1) % n] = 1
    return a


def star(n):  # node 0 is the centre
    a = np.zeros((n, n))
    a[0, 1:] = a[1:, 0] = 1
    return a


def primitives(G, beta):
    d = G.sum(1) + 1
    theta = 1 - beta / d
    Lam = 1 + G @ (beta / d)
    s = beta / d + G @ (beta / d)
    return d, theta, Lam, s


# ---------------------------------------------------------------------------
# Shares
# ---------------------------------------------------------------------------
def impact_weight(n, impact=True, lam=None):
    """Vector of impact weights lambda_i (scalar broadcast); default from `impact`."""
    if lam is None:
        lam = 1.0 if impact else 0.0
    return np.broadcast_to(np.asarray(lam, float), (n,)).copy()


def conversion(G, beta, impact=True, lam=None):
    """kappa_i = 1 + lambda_i (Lambda_i - 1): units of moral good per unit given."""
    _, _, Lam, _ = primitives(G, beta)
    return 1.0 + impact_weight(G.shape[0], impact, lam) * (Lam - 1.0)


def shares(G, B, beta, sigma=1.0, impact=True, lam=None):
    B = np.asarray(B, float)
    d, theta, Lam, _ = primitives(G, beta)
    if abs(sigma - 1.0) < 1e-12:
        return B.copy()
    p = theta / conversion(G, beta, impact, lam)
    return 1.0 / (1.0 + ((1 - B) / B) ** sigma * p ** (sigma - 1))


# ---------------------------------------------------------------------------
# Solver 1: closed form on the active set
# ---------------------------------------------------------------------------
def closed_on_set(G, B, beta, e, K, sigma=1.0, impact=True, lam=None):
    """Closed form given the set K of agents at the upper corner:
    c_K = e,  c_I = (I - Gamma_I G_II)^{-1} (alpha_I + Gamma_I G_IK e 1)."""
    n = G.shape[0]
    d, theta, _, _ = primitives(G, beta)
    pi = shares(G, B, beta, sigma, impact, lam)
    alpha, gamma = pi * e / theta, pi * beta / (d - beta)
    K = np.asarray(K, bool)
    I = ~K
    c = np.full(n, e, float)
    if I.any():
        GI = G[np.ix_(I, I)]
        rhs = alpha[I] + gamma[I] * (G[np.ix_(I, K)] @ np.full(K.sum(), e))
        Mat = np.eye(I.sum()) - gamma[I][:, None] * GI
        try:
            c[I] = np.linalg.solve(Mat, rhs)
        except np.linalg.LinAlgError:          # singular interior block: no interior solution on this set
            c[I] = np.inf
    return c


def nash_closed(G, B, beta, e, sigma=1.0, impact=True, max_rounds=60, lam=None):
    """Equilibrium by active-set iteration on the closed form; if the active set
    cycles (possible only outside the contraction region), the corner set is taken
    from monotone best-response iteration and the closed form is applied on it."""
    n = G.shape[0]
    d, theta, _, _ = primitives(G, beta)
    pi = shares(G, B, beta, sigma, impact, lam)
    alpha, gamma = pi * e / theta, pi * beta / (d - beta)
    K = np.zeros(n, bool)
    for _ in range(max_rounds):
        c = closed_on_set(G, B, beta, e, K, sigma, impact, lam)
        if not np.all(np.isfinite(c)):
            break
        newK = (alpha + gamma * (G @ np.minimum(c, e))) >= e - 1e-12
        if np.array_equal(newK, K) and np.all(c <= e + 1e-9) and np.all(c >= 0):
            return np.minimum(c, e)
        K = newK
    it_sol = nash_iterate(G, B, beta, e, sigma, impact, lam=lam)
    K = it_sol >= e - 1e-9
    c = closed_on_set(G, B, beta, e, K, sigma, impact, lam)
    return np.minimum(c, e) if np.all(np.isfinite(c)) else it_sol


def nash_iterate(G, B, beta, e, sigma=1.0, impact=True, start=0.0, tol=1e-13, it=200000, lam=None):
    d, theta, _, _ = primitives(G, beta)
    pi = shares(G, B, beta, sigma, impact, lam)
    alpha, gamma = pi * e / theta, pi * beta / (d - beta)
    c = np.full(G.shape[0], float(start))
    for _ in range(it):
        cn = np.minimum(e, alpha + gamma * (G @ c))
        if np.max(np.abs(cn - c)) < tol:
            return cn
        c = cn
    return cn


# ---------------------------------------------------------------------------
# Solver 2: formula-free (maximise the actual utility)
# ---------------------------------------------------------------------------
def _make_utility(i, G, B, beta, e, sigma, impact, prim, kap):
    d, theta, Lam, _ = prim
    lam = kap[i]
    Bi = B[i]
    r = 1 - 1 / sigma if abs(sigma - 1.0) > 1e-12 else 0.0

    def f(ci, S):
        X = e - ci + beta / d[i] * (ci + S)
        Y = lam * ci
        if X <= 0 or Y <= 0:
            return -np.inf
        if r == 0.0:
            return (1 - Bi) * np.log(X) + Bi * np.log(Y)
        return np.log((1 - Bi) * X ** r + Bi * Y ** r) / r   # log of the CES aggregate
    return f


def _argmax(f, lo, hi, tol=1e-12):
    g = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    x1, x2 = b - g * (b - a), a + g * (b - a)
    f1, f2 = f(x1), f(x2)
    while b - a > tol:
        if f1 < f2:
            a, x1, f1 = x1, x2, f2
            x2 = a + g * (b - a)
            f2 = f(x2)
        else:
            b, x2, f2 = x2, x1, f1
            x1 = b - g * (b - a)
            f1 = f(x1)
    x = (a + b) / 2
    return hi if f(hi) >= f(x) else x


def nash_utility(G, B, beta, e, sigma=1.0, impact=True, tol=1e-10, it=5000, lam=None):
    """Formula-free Nash equilibrium: Gauss-Seidel best responses, each obtained by
    maximising the agent's actual utility on [0, e] (golden-section search)."""
    B = np.asarray(B, float)
    n = G.shape[0]
    prim = primitives(G, beta)
    kap = conversion(G, beta, impact, lam)
    fs = [_make_utility(i, G, B, beta, e, sigma, impact, prim, kap) for i in range(n)]
    c = np.zeros(n)
    for _ in range(it):
        old = c.copy()
        for i in range(n):
            S = G[i] @ c
            c[i] = _argmax(lambda x: fs[i](x, S), 1e-14, e)
        if np.max(np.abs(c - old)) < tol:
            return c
    return c
