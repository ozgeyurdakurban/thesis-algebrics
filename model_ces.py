"""
================================================================================
 Model -- CES benchmark (no-network, N agents)   (Section: The CES Benchmark)
================================================================================
CES is the benchmark, solved only in the no-network case (d_i = n, a single group
sharing a global public good). Sets up the model objects and verifies:
  - the CES ratio rule  (Y_i/X_i)^(rho-1) = delta theta / ((1-delta) * Lambda),
    Lambda = 1 + (n-1)beta/n  the (constant) impact multiplier,  Y_i = k_i X_i;
  - the Cobb--Douglas limit:  k_i -> Lambda * kappa_i as rho -> 0, with the
    behaviorally relevant object k_i/Lambda -> kappa_i = B_i/(A_i theta).
This is model setup (the no-network CES equilibrium is solved in the equilibrium
section).

Run:  python model_ces.py
================================================================================
"""
import sympy as sp

def is_zero(expr):
    """Identically zero? powsimp first, then numeric fallback (handles symbolic
    powers like X^(rho-1) that simplify() leaves unreduced)."""
    if sp.simplify(sp.powsimp(sp.expand_power_base(expr, force=True), force=True)) == 0:
        return True
    pts = [{'e':10,'beta':2,'n':4,'rho':-1,'delta':sp.Rational(3,10),
            'c_i':2,'C_mi':3,'k':sp.Rational(1,2)},
           {'e':8,'beta':sp.Rational(3,2),'n':5,'rho':sp.Rational(1,2),
            'delta':sp.Rational(1,2),'c_i':1,'C_mi':2,'k':sp.Rational(3,5)}]
    for pt in pts:
        sub = {s: pt[s.name] for s in expr.free_symbols if s.name in pt}
        if complex(sp.N(expr.subs(sub))).__abs__() > 1e-9:
            return False
    return True

e, beta, n, rho, delta = sp.symbols('e beta n rho delta', positive=True)
c_i, C_mi = sp.symbols('c_i C_mi', positive=True)

theta  = 1 - beta/n
Lam    = 1 + (n-1)*beta/n               # impact multiplier, no-network (constant)
X = e - c_i + (beta/n)*(c_i + C_mi)      # = e - theta c_i + (beta/n) C_{-i}
Y = Lam * c_i                            # moral component, no-network form
Z = delta*X**rho + (1-delta)*Y**rho      # inner CES index ; U = Z^(1/rho)

print("=== CES benchmark (no network, d_i = n) ===")
print("theta              =", theta)
print("Lambda (=dY/dc_i)  =", sp.simplify(Lam), "  (1 + (n-1)beta/n)")
print("X_i =", sp.simplify(X), " ;  Y_i = Lambda c_i")

# Interior stationarity dZ/dc_i = 0  (same condition for max of U=Z^(1/rho), any rho)
foc = sp.diff(Z, c_i)
# Clean form: delta X^(rho-1)(-theta) + (1-delta) Y^(rho-1) Lambda = 0
ratio_rule = sp.Eq((Y/X)**(rho-1), delta*theta/((1-delta)*Lam))
# verify the FOC reduces to the ratio rule
foc_clean = delta*X**(rho-1)*(-theta) + (1-delta)*Y**(rho-1)*Lam
print("\nFOC dZ/dc_i = rho * [delta X^(rho-1)(-theta)+(1-delta)Y^(rho-1)Lambda]? ->",
      is_zero(foc - rho*foc_clean))

# k_i from the ratio rule:  Y = k X
k = sp.symbols('k', positive=True)
k_def = (delta*theta/((1-delta)*Lam))**(1/(rho-1))
print("k_i = (delta theta /((1-delta) Lambda))^(1/(rho-1))")

# Cobb--Douglas limit rho -> 0
kappa = (1-delta)/(delta*theta)          # = B/(A theta) with A=delta, B=1-delta
lim_k = sp.limit(k_def, rho, 0)
print("lim_{rho->0} k_i =", sp.simplify(lim_k))
print("equals Lambda * kappa  (kappa = B/(A theta), A=delta,B=1-delta)? ->",
      sp.simplify(lim_k - Lam*kappa) == 0)
print("behaviorally relevant k_i/Lambda -> kappa ? ->",
      sp.simplify(lim_k/Lam - kappa) == 0)

checks = {
    "FOC reduces to ratio rule":   is_zero(foc - rho*foc_clean),
    "lim k = Lambda*kappa":        sp.simplify(lim_k - Lam*kappa) == 0,
    "k/Lambda -> kappa":           sp.simplify(lim_k/Lam - kappa) == 0,
}
print()
for kk, vv in checks.items():
    print(f"  [{'PASS' if vv else 'FAIL'}] {kk}")
assert all(checks.values())
print("All CES benchmark model checks passed.")
