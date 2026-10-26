"""
Step 2 -- Cobb-Douglas equilibria with HETEROGENEOUS prosocial weights B_i, general N.

Best response (s02, A_i + B_i = 1):  c_i = (B_i/theta_i)(e + (beta/d_i) S_i)
                                     = alpha_i + gamma_i S_i,
  alpha_i = B_i e/theta_i ,  gamma_i = B_i beta/(d_i - beta).

Derived here:
  1. K_n (no network), general n: phi_i, aggregate C*, c_i* = e phi_i / mu,
     mu = 1 - (beta/n) sum phi;  mu > 0 <=> rho(Gamma G) < 1; interiority;
     ordering and cross effects;
  2. ANY network: c* = (I - Gamma G)^{-1} alpha = M alpha (weighted Bonacich form),
     and the comparative statics  d c_i*/d B_j = M_ij c_j*/B_j  (>0 on connected g);
  3. Star S_n, general n: centre depends on the leaves only through Sigma_L = sum of
     leaf weights; leaf-vs-centre condition; position-preference threshold B_C*(B_P, n);
  4. Cycle C_n, general n: matrix/Neumann form; ordering by B holds on C_3 but NOT in
     general on C_n, n >= 4 (counterexample);
  5. Mixed regimes (agents at e): active-set characterisation;
  6. Consistency: every heterogeneous formula reduces to the homogeneous one of s03.

Thesis numbering (version of 25 Oct 2026): Propositions 2.4.15-2.4.20, Lemma 2.4.17.
"""

import random

import numpy as np
import sympy as sp

from common import EXP, Log, beta, e, n, positive_on, rat, same, simp
import netlib as NL

L = Log(__file__)


def th(dd):
    return 1 - beta / dd


# ===========================================================================
L.head("1. Complete network K_n, heterogeneous B_i, general n")
# ===========================================================================
Bi, C, ci = sp.symbols("B_i C c_i", positive=True)
# agent i: S_i = C - c_i
sol_i = simp(sp.solve(sp.Eq(ci, Bi / th(n) * (e + beta / n * (C - ci))), ci)[0])
L.show("c_i as a function of the aggregate C", sol_i)
phi = simp(n * Bi / (n - (1 - Bi) * beta))
L.check("c_i = phi_i (e + (beta/n) C),  phi_i = n B_i/(n - (1-B_i) beta)", same(sol_i, phi * (e + beta / n * C)))
L.check("phi_i = B_i/(A_i theta + B_i) (thesis notation)", same(phi, Bi / ((1 - Bi) * th(n) + Bi)))
Phi = sp.symbols("Phi", positive=True)       # Phi = sum_j phi_j
Cstar = simp(sp.solve(sp.Eq(C, Phi * (e + beta / n * C)), C)[0])
L.show("C* (aggregate)", Cstar)
mu = 1 - beta / n * Phi
L.check("C* = e Phi / mu,  mu = 1 - (beta/n) Phi", same(Cstar, e * Phi / mu))
ci_star = simp(phi * (e + beta / n * Cstar))
L.check("c_i* = e phi_i / mu", same(ci_star, e * phi / mu))
L.tex_eq("phi_i", r"\varphi_i", phi)
L.tex_eq("c_i* K_n het", r"c_i^\star", e * sp.Symbol(r"\varphi_i") / (1 - beta / n * sp.Symbol(r"\Phi")))

# explicit check for n = 2..6: solve the n linear equations directly
ok = True
for m in range(2, 7):
    Bs = sp.symbols(f"B1:{m+1}", positive=True)
    cs = sp.symbols(f"c1:{m+1}", positive=True)
    eqs = [sp.Eq(cs[i], Bs[i] / th(m) * (e + beta / m * (sum(cs) - cs[i]))) for i in range(m)]
    sol = sp.solve(eqs, cs, dict=True)[0]
    ph = [m * b / (m - (1 - b) * beta) for b in Bs]
    mu_m = 1 - beta / m * sum(ph)
    for i in range(m):
        pt = {b: rat(0, sp.Rational(1, 3)) for b in Bs}
        pt.update({beta: rat(1, min(m, 3)), e: 15})
        ok &= abs(sp.N((sol[cs[i]] - e * ph[i] / mu_m).subs(pt), 30)) < 1e-20
L.check("general-n formula = direct solution of the n equations, n = 2..6", ok)

# spectral radius <=> mu > 0 (secular equation of Gamma (J - I))
okr = True
rng = np.random.default_rng(1)
for _ in range(400):
    m = int(rng.integers(2, 9)); b = float(rng.uniform(1.01, m - 0.01)); Bv = rng.uniform(0.01, 0.9, m)
    G = NL.complete(m)
    gam = Bv * b / (m - b)
    rho = max(abs(np.linalg.eigvals(gam[:, None] * G)))
    phv = m * Bv / (m - (1 - Bv) * b)
    okr &= (rho < 1) == (1 - b / m * phv.sum() > 0)
L.check("on K_n: rho(Gamma G) < 1  <=>  mu > 0  (400 random draws)", okr)
L.check("interiority c_i* < e  <=>  phi_i < mu", same(sp.solve(sp.Eq(e * phi / mu, e), Phi)[0], (1 - phi) * n / beta) )
# ordering and cross effects
dphi = simp(sp.diff(phi, Bi))
L.show("d phi_i / d B_i", dphi)
L.check("d phi_i/d B_i = n(n - beta)/(n - (1-B_i) beta)^2 > 0", same(dphi, n * (n - beta) / (n - (1 - Bi) * beta) ** 2))
L.say("  => c_i* - c_j* = (e/mu)(phi_i - phi_j), so sign(c_i* - c_j*) = sign(B_i - B_j) since phi is increasing")
L.check("cross effect: c_i* rises with every B_j (through mu)", same(sp.diff(e * phi / mu, Phi), e * phi * beta / n / mu**2))
L.check("homogeneous limit: B_i = B gives c*_K of s03",
        same((e * phi / mu).subs(Phi, n * phi).subs(Bi, sp.Symbol("B", positive=True)),
             sp.Symbol("B", positive=True) * n * e / (n - beta - sp.Symbol("B", positive=True) * beta * (n - 1))))

# ===========================================================================
L.head("2. Any network: matrix solution and comparative statics")
# ===========================================================================
def sym_network_check(Gm, label):
    m = Gm.shape[0]
    Bs = sp.symbols(f"B1:{m+1}", positive=True)
    dd = [int(sum(Gm[i, :])) + 1 for i in range(m)]
    alpha = sp.Matrix([Bs[i] * e / th(dd[i]) for i in range(m)])
    Gam = sp.diag(*[Bs[i] * beta / (dd[i] - beta) for i in range(m)])
    Gs = sp.Matrix(Gm.astype(int).tolist())
    M = (sp.eye(m) - Gam * Gs).inv()
    cstar = M * alpha
    # direct solve of the best-response system
    cs = sp.symbols(f"c1:{m+1}")
    eqs = [sp.Eq(cs[i], Bs[i] / th(dd[i]) * (e + beta / dd[i] * sum(Gs[i, j] * cs[j] for j in range(m)))) for i in range(m)]
    sol = sp.solve(eqs, cs, dict=True)[0]
    pt = {b: rat(0, sp.Rational(1, 6)) for b in Bs}
    pt.update({beta: sp.Rational(3, 2), e: 15})
    ok1 = all(abs(sp.N((sol[cs[i]] - cstar[i]).subs(pt), 30)) < 1e-20 for i in range(m))
    # comparative statics d c_i / d B_j = M_ij c_j / B_j
    ok2 = True
    Mn = M.subs(pt); cn = cstar.subs(pt)
    for j in range(m):
        dc = sp.diff(cstar, Bs[j]).subs(pt)
        for i in range(m):
            ok2 &= abs(sp.N(dc[i] - Mn[i, j] * cn[j] / pt[Bs[j]], 30)) < 1e-20
            ok2 &= sp.N(dc[i]) > 0
    L.check(f"{label}: (I - Gamma G)^-1 alpha = direct solution", ok1)
    L.check(f"{label}: d c_i*/d B_j = M_ij c_j*/B_j > 0 for all i, j", ok2)


sym_network_check(NL.star(4), "S_4")
sym_network_check(NL.cycle(4), "C_4")
sym_network_check(NL.cycle(5), "C_5")
path = np.zeros((4, 4)); path[0, 1] = path[1, 0] = path[1, 2] = path[2, 1] = path[2, 3] = path[3, 2] = 1
sym_network_check(path, "path P_4 (irregular, non-star)")

# Neumann series / Bonacich interpretation (numerical)
okN = True
for _ in range(200):
    m = int(rng.integers(3, 9)); G = [NL.complete, NL.cycle, NL.star][int(rng.integers(0, 3))](m)
    b = float(rng.uniform(1.05, 1.95)); Bv = rng.uniform(0.01, 0.15, m)
    d = G.sum(1) + 1; gam = Bv * b / (d - b); al = Bv * 15 / (1 - b / d)
    T = gam[:, None] * G
    if max(abs(np.linalg.eigvals(T))) >= 1:
        continue
    exact = np.linalg.solve(np.eye(m) - T, al)
    series, term = np.zeros(m), al.copy()
    for _k in range(3000):
        series += term; term = T @ term
    okN &= np.allclose(series, exact, atol=1e-9)
L.check("c* = sum_k (Gamma G)^k alpha (walk-weighted Bonacich form) when rho < 1", okN)

# ===========================================================================
L.head("3. Star S_n, heterogeneous weights, general n")
# ===========================================================================
BC, SigL, cC = sp.symbols("B_C Sigma_L c_C", positive=True)
Bj = sp.symbols("B_j", positive=True)
alpha_C, gamma_C = BC * e / th(n), BC * beta / (n - beta)
# leaves: c_j = alpha_j + gamma_j c_C ; sum over leaves: sum alpha_j = 2 e Sigma_L/(2-beta), sum gamma_j = beta Sigma_L/(2-beta)
sum_alpha_L = 2 * e * SigL / (2 - beta)
sum_gamma_L = beta * SigL / (2 - beta)
cC_star = simp(sp.solve(sp.Eq(cC, alpha_C + gamma_C * (sum_alpha_L + sum_gamma_L * cC)), cC)[0])
L.show("c_C*", cC_star)
DSh = (n - beta) * (2 - beta) - BC * beta**2 * SigL
L.check("c_C* = B_C e [n(2-beta) + 2 beta Sigma_L] / [(n-beta)(2-beta) - B_C beta^2 Sigma_L]",
        same(cC_star, BC * e * (n * (2 - beta) + 2 * beta * SigL) / DSh))
cj_star = simp(Bj / th(2) * (e + beta / 2 * cC_star))
L.show("c_j* (leaf j)", cj_star)
L.check("c_j* = B_j (2e + beta c_C*)/(2 - beta)", same(cj_star, Bj * (2 * e + beta * cC_star) / (2 - beta)))
L.check("equivalently c_j* = B_j e [2(n-beta) + n B_C beta] / [(n-beta)(2-beta) - B_C beta^2 Sigma_L]",
        same(cj_star, Bj * e * (2 * (n - beta) + n * BC * beta) / DSh))
L.say("  => the centre depends on the leaves only through Sigma_L = sum of leaf weights")
L.check("rho(Gamma G)^2 = gamma_C sum gamma_j < 1  <=>  D_S(B) > 0",
        same(1 - gamma_C * sum_gamma_L, DSh / ((n - beta) * (2 - beta))))
L.tex_eq("c_C* star het", r"c_C^\star", cC_star)
L.tex_eq("c_j* star het", r"c_j^\star", cj_star)

# explicit-graph check n = 3..6 (direct solve)
ok = True
for m in range(3, 7):
    Bs = sp.symbols(f"B0:{m}", positive=True)
    cs = sp.symbols(f"x0:{m}")
    eqs = [sp.Eq(cs[0], Bs[0] / th(m) * (e + beta / m * sum(cs[1:])))]
    eqs += [sp.Eq(cs[j], Bs[j] / th(2) * (e + beta / 2 * cs[0])) for j in range(1, m)]
    sol = sp.solve(eqs, cs, dict=True)[0]
    pt = {b: rat(0, sp.Rational(1, 6)) for b in Bs}; pt.update({beta: rat(1, 2), e: 15})
    gen_C = cC_star.subs({n: m, BC: Bs[0], SigL: sum(Bs[1:])})
    ok &= abs(sp.N((sol[cs[0]] - gen_C).subs(pt), 30)) < 1e-20
    for j in range(1, m):
        ok &= abs(sp.N((sol[cs[j]] - cj_star.subs({Bj: Bs[j]}).subs({n: m, BC: Bs[0], SigL: sum(Bs[1:])})).subs(pt), 30)) < 1e-20
L.check("general-n star formula = direct solution, n = 3..6", ok)
L.check("homogeneous limit (B_C = B, Sigma_L = (n-1)B) gives s03 centre formula",
        same(cC_star.subs({BC: sp.Symbol("B", positive=True), SigL: (n - 1) * sp.Symbol("B", positive=True)}),
             sp.Symbol("B", positive=True) * e * (n * (2 - beta) + 2 * (n - 1) * sp.Symbol("B", positive=True) * beta)
             / ((n - beta) * (2 - beta) - (n - 1) * sp.Symbol("B", positive=True) ** 2 * beta**2)))

# leaf vs centre
lead = simp(cj_star - cC_star)
L.say("  => leaf j out-contributes the centre  <=>  B_j (2e + beta c_C*) > (2 - beta) c_C*")
# common leaf weight B_P: threshold on the centre's weight
BP = sp.symbols("B_P", positive=True)
gapP = simp((cj_star - cC_star).subs({Bj: BP, SigL: (n - 1) * BP}))
thr = [simp(r) for r in sp.solve(sp.Eq(sp.numer(sp.together(gapP)), 0), BC)]
L.say("  roots in B_C of leaf = centre (common leaf weight):", [sp.sstr(r) for r in thr])
BCstar = [r for r in thr if r != 0][0]
L.show("threshold B_C*(B_P, n)", BCstar)
L.check("threshold B_C* = 2 B_P (n - beta)/[n(2 - beta) + (n-2) B_P beta]",
        same(BCstar, 2 * BP * (n - beta) / (n * (2 - beta) + (n - 2) * BP * beta)))
L.check("n = 3: B_C* = 2 B_P (3-beta)/(6 - 3 beta + B_P beta)  [thesis Prop. star_threshold]",
        same(BCstar.subs(n, 3), 2 * BP * (3 - beta) / (6 - 3 * beta + BP * beta)))


def sT():
    m = sp.Integer(random.randint(3, 10)); b = rat(1, 2)
    return {n: m, beta: b, BP: rat(0, sp.Rational(1, 5))}


L.check("B_C* > B_P: the centre needs a strictly higher weight to match a leaf",
        positive_on(simp(BCstar - BP), sT))
L.check("d B_C*/d n > 0: the hub's handicap grows with the number of leaves", positive_on(simp(sp.diff(BCstar, n)), sT))
L.tex_eq("threshold", r"B_C^{\ast}", BCstar)
# two leaves ordered by own weight
Bk = sp.symbols("B_k", positive=True)
L.check("two leaves j,k: c_j* - c_k* = (B_j - B_k)(2e + beta c_C*)/(2-beta)  => ordered by B",
        same(cj_star - cj_star.subs(Bj, Bk), (Bj - Bk) * (2 * e + beta * cC_star) / (2 - beta)))

# ===========================================================================
L.head("4. Cycle C_n, heterogeneous weights, general n")
# ===========================================================================
L.say("  General n: c* = (I - Gamma A_cyc)^{-1} alpha with alpha_i = 3 B_i e/(3-beta), gamma_i = B_i beta/(3-beta).")
L.say("  Sufficient for rho < 1: 2 max_i gamma_i < 1  <=>  max_i B_i < (3-beta)/(2 beta).")
# ordering by B on C_3 (= K_3) holds; on C_4 it can fail
G4 = NL.cycle(4)
Bv = np.array([0.06, 0.05, 0.14, 0.05])     # agent 0 vs agent 1 vs ...
c4 = NL.nash_closed(G4, Bv, 1.5, 15.0)
L.say(f"  C_4 example B = {Bv.tolist()}: c* = {np.round(c4, 4).tolist()}")
cex = None
for _ in range(20000):
    Bv = rng.uniform(0.01, 0.15, 4)
    c4 = NL.nash_closed(G4, Bv, 1.5, 15.0)
    for i in range(4):
        for j in range(4):
            if Bv[i] > Bv[j] and c4[i] < c4[j] - 1e-9 and c4.max() < 15:
                cex = (Bv.copy(), c4.copy(), i, j); break
        if cex: break
    if cex: break
if cex:
    L.say(f"  counterexample on C_4: B = {np.round(cex[0], 4).tolist()}, c* = {np.round(cex[1], 4).tolist()} "
          f"(agent {cex[2]} has higher B but lower c than agent {cex[3]})")
L.check("on C_n (n >= 4) contributions are NOT ordered by B_i alone (neighbours' weights matter)", cex is not None)
okc3 = True
for _ in range(3000):
    Bv = rng.uniform(0.01, 0.2, 3); c3 = NL.nash_closed(NL.cycle(3), Bv, 1.5, 15.0)
    o = np.argsort(Bv)
    okc3 &= bool(np.all(np.diff(c3[o]) >= -1e-12))
L.check("on C_3 = K_3 contributions ARE ordered by B_i (3000 draws)", okc3)

# ===========================================================================
L.head("5. Mixed regimes: active-set characterisation and uniqueness")
# ===========================================================================
okA = okU = True
worst = 0.0
for _ in range(1500):
    m = int(rng.integers(2, 8)); kind = int(rng.integers(0, 3))
    G = [NL.complete, NL.cycle, NL.star][kind](m if (kind != 1 or m >= 3) else 3)
    m = G.shape[0]
    d = G.sum(1) + 1
    b = float(rng.uniform(1.02, min(d.min(), 3) - 0.02))
    Bv = rng.uniform(0.01, 0.6, m)
    lo = NL.nash_iterate(G, Bv, b, 15.0, start=0.0); hi = NL.nash_iterate(G, Bv, b, 15.0, start=15.0)
    K = lo >= 15.0 - 1e-9
    c_set = NL.closed_on_set(G, Bv, b, 15.0, K)          # closed form on the equilibrium corner set
    worst = max(worst, np.max(abs(c_set - lo)), np.max(abs(lo - hi)))
L.say(f"  max deviation (active set vs least vs greatest equilibrium) = {worst:.2e}")
L.check("active-set closed form reproduces the equilibrium incl. corners; least = greatest (1500 draws)", worst < 1e-7)

# ===========================================================================
L.head("6. Experimental illustration (e = 15, n = 3, beta = 3/2)")
# ===========================================================================
for Bv in ([0.10, 0.10, 0.10], [0.05, 0.10, 0.15], [0.15, 0.05, 0.10], [0.20, 0.05, 0.05]):
    Bv = np.array(Bv)
    cr = NL.nash_closed(NL.complete(3), Bv, 1.5, 15.0); cs_ = NL.nash_closed(NL.star(3), Bv, 1.5, 15.0)
    L.say(f"  B = {Bv.tolist()}: regular {np.round(cr, 3).tolist()} | star (centre first) {np.round(cs_, 3).tolist()}")
thr3 = sp.N(BCstar.subs({n: 3, beta: sp.Rational(3, 2), BP: sp.Rational(1, 20)}), 6)
L.say(f"  threshold at B_P = 0.05: B_C* = {thr3}")
L.check("experiment, B_P = 0.05: centre overtakes leaves only above B_C* = 2/21 = 0.0952",
        abs(thr3 - 2 / 21) < 1e-6)

L.finish()
