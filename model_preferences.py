"""
================================================================================
 Model 1 -- Environment and preferences   (Section: Environment and preferences)
================================================================================
Builds the common preference objects symbolically and verifies the two model-
level facts used everywhere:
  (a) the impact multiplier (constant in c_i) cancels from the Cobb--Douglas FOC,
      so the FOC reads  c_i = kappa_i X_i ,  kappa_i = B_i/(A_i theta_i);
  (b) the normalization A_i+B_i=1 is WLOG: behavior depends on (A,B) only through
      the ratio B/A, since kappa is homogeneous of degree zero.
This is model setup (not an equilibrium solve).

Run:  python model_preferences.py
================================================================================
"""
import sympy as sp

e, beta, n = sp.symbols('e beta n', positive=True)
A, B, m    = sp.symbols('A B m', positive=True)   # m : impact multiplier, constant in c_i
c_i, C_mi  = sp.symbols('c_i C_mi', positive=True)
t          = sp.symbols('t', positive=True)

theta = 1 - beta/n                                # representative (fully connected) theta
X = e - c_i + (beta/n)*(c_i + C_mi)               # material payoff  = e - theta c_i + (beta/n) C_{-i}
Y = m * c_i                                       # moral component  = (impact multiplier) * c_i
U = X**A * Y**B                                   # Cobb--Douglas utility

print("X_i           =", sp.simplify(X), "   (dX/dc_i =", sp.diff(X, c_i), "= -theta)")
print("Y_i           = m * c_i   with m the (position-determined) impact multiplier")
print("U_i           = X^A Y^B")

logU = A*sp.log(X) + B*sp.log(Y)
foc  = sp.diff(logU, c_i)
print("\nd(log U)/dc_i =", sp.simplify(foc))

# (a) impact multiplier cancels
moral_cancels = sp.simplify(sp.diff(foc, m)) == 0
print("impact multiplier m cancels from FOC?  d(FOC)/dm = 0  ->", moral_cancels)

# FOC  <=>  c_i = kappa X_i ,  kappa = B/(A theta)
kappa = B/(A*theta)
foc_is_kappa = sp.simplify(foc.subs(m, 1) - (B/c_i - A*theta/X)) == 0 and \
               sp.simplify(B*X - A*theta*(kappa*X)) == 0
print("FOC <=> c_i = kappa X_i,  kappa = B/(A theta)?       ->", foc_is_kappa)

# (b) normalization scale-invariance
kappa_invariant = sp.simplify((t*B)/((t*A)*theta) - kappa) == 0
print("kappa invariant under (A,B)->(tA,tB)?               ->", kappa_invariant)

assert moral_cancels and foc_is_kappa and kappa_invariant
print("\nAll model-preference checks passed.")
