"""
================================================================================
 Model 4 -- Cobb--Douglas on the star network
================================================================================
One center (agent 1, d_1 = n) linked to n-1 leaves (d_j = 2), leaves linked
only to the center. Maximally degree-asymmetric: theta differs by position, so
positions differ even under a common prosocial weight. Sets up center and leaf
objects and verifies each FOC.

Run:  python model_star.py
================================================================================
"""
import sympy as sp

e, beta, n, A, B = sp.symbols('e beta n A B', positive=True)
c1, cj, C, Speri = sp.symbols('c_1 c_j C S_peri', positive=True)  # center, a leaf, total, sum of leaves

# --- center (d_1 = n) ---
theta1 = 1 - beta/n
mcoef1 = 1 + (n-1)*beta/2            # 1 + sum_{leaves} beta/d_leaf = 1 + (n-1)(beta/2)
X1 = e - c1 + (beta/n)*(c1 + Speri)  # = e - theta1 c1 + (beta/n) S_peri  ; C = c1 + S_peri
Y1 = mcoef1 * c1
U1 = X1**A * Y1**B

# --- leaf j (d_j = 2) ---
thetaL = 1 - beta/2
mcoefL = 1 + beta/n                  # 1 + beta/d_center = 1 + beta/n
Xj = e - cj + (beta/2)*(c1 + cj)     # = e - thetaL c_j + (beta/2) c_1
Yj = mcoefL * cj
Uj = Xj**A * Yj**B

print("=== Cobb--Douglas on the star ===")
print("center: theta_1   =", theta1, " ; moral coef =", sp.simplify(mcoef1), " (1+(n-1)beta/2)")
print("leaf  : theta_L   =", thetaL, " ; moral coef =", sp.simplify(mcoefL), " (1+beta/n)")
print("center X_1        =", sp.simplify(X1))
print("leaf   X_j        =", sp.simplify(Xj))
print("dilemma           : 1 < beta < 2  (binds at the leaves, d_j = 2)")
print("theta_1 != theta_L  =>  kappa differs by position even with common (A,B)")

q = sp.symbols('q', positive=True)
foc1 = sp.diff(A*sp.log(X1) + B*sp.log(q*c1), c1)
focj = sp.diff(A*sp.log(Xj) + B*sp.log(q*cj), cj)
k1, kL = B/(A*theta1), B/(A*thetaL)
checks = {
    "center moral coef cancels": sp.simplify(sp.diff(foc1, q)) == 0,
    "leaf   moral coef cancels": sp.simplify(sp.diff(focj, q)) == 0,
    "center FOC <=> c1=k1 X1":   sp.simplify(B*X1 - A*theta1*(k1*X1)) == 0,
    "leaf   FOC <=> cj=kL Xj":   sp.simplify(B*Xj - A*thetaL*(kL*Xj)) == 0,
    "theta_1 != theta_L":        sp.simplify(theta1 - thetaL) != 0,
}
for k, v in checks.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")
assert all(checks.values())
print("All star model checks passed.")
