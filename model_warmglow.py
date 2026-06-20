"""
================================================================================
 Model 5 -- Warm-glow benchmark with baseline T  (earlier-version model)
================================================================================
Additive-baseline warm-glow moral term Y^WG = T + c_i (NOT Andreoni's general
model). Sets it up and shows the key contrast with the impact-written model:
the baseline T survives in the FOC (marginal moral utility B/(T+c)), whereas the
impact-written model has B/c_i (the T=0 case). Same material payoff X_i.

Run:  python model_warmglow.py
================================================================================
"""
import sympy as sp

e, beta, n, A, B, T = sp.symbols('e beta n A B T', positive=True)
c_i, C_mi = sp.symbols('c_i C_mi', positive=True)

theta = 1 - beta/n
X  = e - c_i + (beta/n)*(c_i + C_mi)        # same material payoff

# warm-glow (baseline-T) utility
Ywg = T + c_i
Uwg = X**A * Ywg**B
foc_wg = sp.diff(A*sp.log(X) + B*sp.log(Ywg), c_i)

# impact-written utility (T=0, moral term = moral coef * c_i; coef constant)
m = sp.symbols('m', positive=True)
foc_ib = sp.diff(A*sp.log(X) + B*sp.log(m*c_i), c_i)

print("=== Warm-glow benchmark with baseline T ===")
print("Y^WG = T + c_i ,   U^WG = X^A (T+c_i)^B")
print("warm-glow FOC d(logU)/dc =", sp.simplify(foc_wg), "   (marginal moral utility B/(T+c_i))")
print("impact   FOC d(logU)/dc =", sp.simplify(foc_ib), "   (marginal moral utility B/c_i)")

checks = {
    "T survives in warm-glow FOC":      sp.simplify(sp.diff(foc_wg, T)) != 0,
    "impact FOC independent of m":      sp.simplify(sp.diff(foc_ib, m)) == 0,
    "warm-glow at T=0 equals impact FOC": sp.simplify(foc_wg.subs(T, 0) - foc_ib.subs(m, 1)) == 0,
}
for k, v in checks.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")
assert all(checks.values())
print("All warm-glow benchmark checks passed.")
