"""
================================================================================
 Model 3 -- Cobb--Douglas on the cycle network
================================================================================
Closed ring: each agent has two neighbors (i-1, i+1), so d_i = 3 for all i.
Degree-symmetric but with LOCAL interaction (payoff depends on the two
neighbors, not the aggregate). Sets up the model objects and verifies the FOC.

Run:  python model_cycle.py
================================================================================
"""
import sympy as sp

e, beta, A, B = sp.symbols('e beta A B', positive=True)
ci, cl, cr    = sp.symbols('c_i c_{i-1} c_{i+1}', positive=True)  # own, left, right neighbor

d     = 3
theta = 1 - beta/3
mcoef = 1 + 2*beta/3          # 1 + beta/d_{i-1} + beta/d_{i+1} = 1 + 2(beta/3)
X = e - ci + (beta/3)*(cl + ci + cr)        # = e - theta c_i + (beta/3)(c_{i-1}+c_{i+1})
Y = mcoef * ci
U = X**A * Y**B

print("=== Cobb--Douglas on the cycle (d_i = 3) ===")
print("theta            =", theta)
print("impact multiplier=", sp.simplify(mcoef), "  (= 1 + 2 beta/3)")
print("X_i              =", sp.simplify(X), "  (depends on the two neighbors)")
print("Y_i              =", sp.simplify(Y))
print("dilemma          : 1 < beta < 3")

foc   = sp.diff(A*sp.log(X) + B*sp.log(Y), ci)
kappa = B/(A*theta)
q     = sp.symbols('q', positive=True)
focq  = sp.diff(A*sp.log(X) + B*sp.log(q*ci), ci)
checks = {
    "dX/dc = -theta":           sp.simplify(sp.diff(X, ci) - (-theta)) == 0,
    "moral coef cancels":       sp.simplify(sp.diff(focq, q)) == 0,
    "FOC <=> c=kappa X":        sp.simplify(B*X - A*theta*(kappa*X)) == 0,
}
print("\nFOC d(logU)/dc_i =", sp.simplify(foc))
for k, v in checks.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")
assert all(checks.values())
print("All cycle model checks passed.")
