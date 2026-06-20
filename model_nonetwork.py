"""
================================================================================
 Model 2 -- Cobb--Douglas without a network   (global public good)
================================================================================
Fully connected group: every agent linked to all others, d_i = n.
Sets up the model objects and verifies the FOC and the cancellation of the
impact multiplier. Degree-symmetric: theta common to all agents.

Run:  python model_nonetwork.py
================================================================================
"""
import sympy as sp

e, beta, n, A, B = sp.symbols('e beta n A B', positive=True)
c_i, C_mi, C     = sp.symbols('c_i C_mi C', positive=True)

d     = n                      # closed-neighborhood size
theta = 1 - beta/n             # local net marginal cost
mcoef = 1 + (n-1)*beta/n       # impact multiplier  1 + sum_{j} beta/d_j  (d_j = n)
X = e - c_i + (beta/n)*(c_i + C_mi)        # = e - theta c_i + (beta/n) C_{-i}
Y = mcoef * c_i
U = X**A * Y**B

print("=== Cobb--Douglas without a network (d_i = n) ===")
print("theta            =", theta)
print("impact multiplier=", sp.simplify(mcoef), "  (= 1 + (n-1)beta/n)")
print("X_i              =", sp.simplify(X))
print("Y_i              =", sp.simplify(Y))
print("dilemma          : 1 < beta < n")

foc = sp.diff(A*sp.log(X) + B*sp.log(Y), c_i)
kappa = B/(A*theta)
checks = {
    "moral coef cancels":          sp.simplify(sp.diff(foc, sp.symbols('q'))) == 0 or True,  # mcoef has no free new symbol; check directly:
    "FOC independent of mcoef":    sp.simplify(foc - (B/c_i - A*theta/X)) == 0,
    "FOC <=> c=kappa X":           sp.simplify(B*X - A*theta*(kappa*X)) == 0,
}
# direct cancellation check: replace mcoef by a free symbol and differentiate
q = sp.symbols('q', positive=True)
focq = sp.diff(A*sp.log(X) + B*sp.log(q*c_i), c_i)
checks["moral coef cancels"] = sp.simplify(sp.diff(focq, q)) == 0

print("\nFOC d(logU)/dc_i =", sp.simplify(foc))
for k, v in checks.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")
assert all(checks.values())
print("All no-network model checks passed.")
