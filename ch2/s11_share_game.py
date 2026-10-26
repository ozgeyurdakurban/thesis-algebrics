"""
Step 3d -- The network game in giving shares (Section 2.4.2 of the thesis).

For an arbitrary share vector pi (no CES or CD structure assumed), the interior best response is
    c_i = alpha~_i + gamma~_i S_i,  alpha~_i = pi_i e / theta_i,  gamma~_i = pi_i beta/(d_i - beta),
    theta_i = 1 - beta/d_i.
Derived with SymPy:
  1. K_n: best response in the aggregate (phi_i), aggregate solution, symmetric solution, ceiling
     theta/Lambda = (n-beta)/(n+beta(n-1)) = odds threshold theta/beta, determinant identity
     det(I - Gamma~ G) = det(I + Gamma~) (1 - (beta/n) Phi);
  2. C_n: symmetric solution and ceiling;
  3. S_n: centre/leaf solution for general shares, reduced equation, spectrum, leaf-vs-centre odds
     condition theta_L/theta_C and the threshold pi_C*, corner condition odds(pi_C) < theta_C/beta;
  4. any network: dc*/dpi_j = M_ij c_j*/pi_j on a 4-node example;
  5. the sign of the bracket W_j/theta_j = c_j/pi_j used in the comparative statics.
"""
import itertools

import sympy as sp

from common import Log, beta, e, n, same, simp

L = Log(__file__)

# ===========================================================================
L.head("1. Complete network K_n with arbitrary shares")
# ===========================================================================
pi_i, C, ci = sp.symbols("pi_i C c_i", positive=True)
theta = (n - beta) / n
sol = sp.solve(sp.Eq(ci, pi_i / theta * (e + beta / n * (C - ci))), ci)[0]
phi = n * pi_i / (n - (1 - pi_i) * beta)
L.check("Step 1: c_i = phi_i (e + beta C/n), phi_i = n pi_i/(n-(1-pi_i)beta)", same(sol, phi * (e + beta * C / n)))
L.check("phi_i < 1 iff pi_i (n - beta) < n - beta  (identically true for pi_i < 1)",
        sp.simplify((1 - phi) * (n - (1 - pi_i) * beta) - (n - beta) * (1 - pi_i)) == 0)
Phi = sp.symbols("Phi", positive=True)
Cs = sp.solve(sp.Eq(C, Phi * (e + beta * C / n)), C)[0]
mu = 1 - beta / n * Phi
L.check("Step 2: C* = e Phi/mu, e + beta C*/n = e/mu", same(Cs, e * Phi / mu) and same(e + beta * Cs / n, e / mu))
# determinant lemma: gamma/(1+gamma) = (beta/n) phi
gam = pi_i * beta / (n - beta)
L.check("gamma~/(1+gamma~) = (beta/n) phi_i", same(gam / (1 + gam), beta / n * phi))
# explicit determinant for n = 3 with generic shares
p1, p2, p3 = sp.symbols("p1 p2 p3", positive=True)
shares3 = [p1, p2, p3]
g = [pk * beta / (3 - beta) for pk in shares3]
G3 = sp.ones(3, 3) - sp.eye(3)
Gam = sp.diag(*g)
detv = sp.simplify((sp.eye(3) - Gam * G3).det())
Phi3 = sum(3 * pk / (3 - (1 - pk) * beta) for pk in shares3)
rhs = sp.simplify((sp.eye(3) + Gam).det() * (1 - beta / 3 * Phi3))
L.check("det(I - Gamma~ G) = det(I + Gamma~)(1 - (beta/n) Phi) for n = 3 with three different shares", same(detv, rhs))
# symmetric
pi = sp.symbols("pi", positive=True)
phi_s = n * pi / (n - (1 - pi) * beta)
c_sym = sp.simplify(e * phi_s / (1 - beta * phi_s))
L.check("Step 5: c_K = pi n e/(n - beta - pi beta (n-1))", same(c_sym, pi * n * e / (n - beta - pi * beta * (n - 1))))
ceil = sp.solve(sp.Eq(c_sym, e), pi)[0]
Lam = 1 + (n - 1) * beta / n
L.check("ceiling pi = (n-beta)/(n + beta(n-1)) = theta/Lambda", same(ceil, theta / Lam) and same(ceil, (n - beta) / (n + beta * (n - 1))))
L.check("Lambda - theta = beta, so pi < theta/Lambda iff odds(pi) < theta/beta", same(Lam - theta, beta))
od = sp.solve(sp.Eq(pi / (1 - pi), theta / beta), pi)[0]
L.check("odds(pi) = theta/beta  <=>  pi = theta/(theta+beta) = theta/Lambda", same(od, theta / Lam))
L.check("d phi/d pi = n (n-beta)/(n - (1-pi) beta)^2 > 0", same(sp.diff(phi, pi_i), n * (n - beta) / (n - (1 - pi_i) * beta) ** 2))

# ===========================================================================
L.head("2. Cycle C_n with arbitrary shares")
# ===========================================================================
c3 = sp.symbols("c3", positive=True)
sol_c = sp.solve(sp.Eq(c3, pi * 3 * e / (3 - beta) + pi * beta / (3 - beta) * 2 * c3), c3)[0]
L.check("C_n symmetric: c = 3 pi e/(3 - beta - 2 pi beta)", same(sol_c, 3 * pi * e / (3 - beta - 2 * pi * beta)))
L.check("C_n symmetric equals K_3 formula", same(sol_c, c_sym.subs(n, 3)))
L.check("C_n ceiling pi = (3-beta)/(3+2 beta) = theta/Lambda (d=3)",
        same(sp.solve(sp.Eq(sol_c, e), pi)[0], (3 - beta) / (3 + 2 * beta)))

# ===========================================================================
L.head("3. Star S_n with arbitrary shares")
# ===========================================================================
piC, piL, cC, cL = sp.symbols("pi_C pi_L c_C c_L", positive=True)
Sig = (n - 1) * piL
aC, gC = piC * n * e / (n - beta), piC * beta / (n - beta)
aL, gL = piL * 2 * e / (2 - beta), piL * beta / (2 - beta)
solst = sp.solve([sp.Eq(cC, aC + gC * (n - 1) * cL), sp.Eq(cL, aL + gL * cC)], [cC, cL], dict=True)[0]
DS = (n - beta) * (2 - beta) - piC * beta ** 2 * Sig
cC_f = piC * e * (n * (2 - beta) + 2 * beta * Sig) / DS
cL_f = piL * e * (2 * (n - beta) + n * piC * beta) / DS
L.check("Steps 1-2: c_C = pi_C e [n(2-beta) + 2 beta Sigma]/D~", same(solst[cC], cC_f))
L.check("Step 2: c_L = pi_L (2e + beta c_C)/(2-beta) = pi_L e [2(n-beta) + n pi_C beta]/D~",
        same(solst[cL], cL_f) and same(cL_f, piL * (2 * e + beta * cC_f) / (2 - beta)))
L.check("rho(Gamma~ G)^2 = gamma_C (n-1) gamma_L = pi_C beta^2 Sigma/((n-beta)(2-beta)); 1 - rho^2 = D~/((n-beta)(2-beta))",
        same(1 - gC * (n - 1) * gL, DS / ((n - beta) * (2 - beta))))
# eigenvalues of the bipartite matrix for n = 4 (3 leaves), generic shares
m = 4
sh = sp.symbols("s0 s1 s2 s3", positive=True)
Gs = sp.zeros(m, m)
for k in range(1, m):
    Gs[0, k] = 1; Gs[k, 0] = 1
dd = [m] + [2] * (m - 1)
Gm = sp.diag(*[sh[k] * beta / (dd[k] - beta) for k in range(m)])
lam_ = sp.symbols("lam")
cp = sp.factor((lam_ * sp.eye(m) - Gm * Gs).det())
expected = lam_ ** (m - 2) * (lam_ ** 2 - Gm[0, 0] * sum(Gm[k, k] for k in range(1, m)))
L.check("non-zero eigenvalues of Gamma~ G on S_4: lambda^2 = gamma_C sum_j gamma_j (heterogeneous shares)", same(cp, expected))
# leaf vs centre
diff_num = piL * (2 * (n - beta) + n * piC * beta) - piC * (n * (2 - beta) + 2 * (n - 1) * piL * beta)
oC, oL = piC / (1 - piC), piL / (1 - piL)
alt = piC * piL * (2 * (n - beta) / oC - n * (2 - beta) / oL)
L.check("sign(c_L - c_C): numerator = pi_C pi_L [2(n-beta)/odds(pi_C) - n(2-beta)/odds(pi_L)]", same(diff_num, alt))
L.check("leaf > centre iff odds(pi_L)/odds(pi_C) > n(2-beta)/(2(n-beta)) = theta_L/theta_C",
        same(n * (2 - beta) / (2 * (n - beta)), ((2 - beta) / 2) / ((n - beta) / n)))
k_ = (2 * (n - beta)) / (n * (2 - beta))
piCs = sp.solve(sp.Eq(piC / (1 - piC), k_ * piL / (1 - piL)), piC)[0]
L.check("threshold pi_C* = 2 pi_L (n-beta)/[n(2-beta) + (n-2) pi_L beta]",
        same(piCs, 2 * piL * (n - beta) / (n * (2 - beta) + (n - 2) * piL * beta)))
L.check("theta_L < theta_C for n > 2", sp.simplify(((n - beta) / n) - (2 - beta) / 2 - beta * (n - 2) / (2 * n)) == 0)
L.check("corner: c_C(leaves at e) = pi_C e (n + beta(n-1))/(n-beta); < e iff odds(pi_C) < theta_C/beta",
        same(piC * n * e * (1 + beta * (n - 1) / n) / (n - beta), piC * e * (n + beta * (n - 1)) / (n - beta))
        and same(sp.solve(sp.Eq(piC * (n + beta * (n - 1)) / (n - beta), 1), piC)[0], ((n - beta) / n) / (((n - beta) / n) + beta)))

# ===========================================================================
L.head("4. General network: dc*_i/dpi_j = M_ij c_j*/pi_j (4-node path with generic shares)")
# ===========================================================================
adj = [(0, 1), (1, 2), (2, 3)]
m = 4
Gp = sp.zeros(m, m)
for a, b in adj:
    Gp[a, b] = Gp[b, a] = 1
deg = [sum(Gp[i, :]) + 1 for i in range(m)]
pis = sp.symbols("q0:4", positive=True)
al = sp.Matrix([pis[i] * e / (1 - beta / deg[i]) for i in range(m)])
Gmm = sp.diag(*[pis[i] * beta / (deg[i] - beta) for i in range(m)])
M = (sp.eye(m) - Gmm * Gp).inv()
cst = sp.simplify(M * al)
ok = True
for i, j in itertools.product(range(m), range(m)):
    lhs = sp.diff(cst[i], pis[j])
    rhs = M[i, j] * cst[j] / pis[j]
    ok &= same(lhs, rhs, sampler=None) or sp.simplify(lhs - rhs) == 0
L.check("dc_i/dpi_j = M_ij c_j/pi_j for all i, j on the path P_4", ok)
Wj = lambda j: e + beta / deg[j] * sum(Gp[j, k] * cst[k] for k in range(m))
L.check("bracket e/theta_j + beta S_j/(d_j - beta) = W_j/theta_j = c_j/pi_j (all j)",
        all(sp.simplify(Wj(j) / (1 - beta / deg[j]) - cst[j] / pis[j]) == 0 for j in range(m)))

L.finish()
