#!/usr/bin/env python3
"""
p2_exact.py -- exact evaluation of

    p_2 = P( two independent uniform random triangles in the unit square meet )
        = 191/300 ,

together with the by-products

    E[alpha^2] = 133/432 , E[alpha^3] = 61/288 , E[alpha^4] = 281/1800 ,
    E[h_4] = 133/36 , E[h_5] = 305/72 , E[h_6] = 281/60 ,
    E[A_3] = 11/144 , E[A_4] = 11/72  , E[A_5] = 79/360 ,

where alpha is the area cut off by the line through two independent uniform
points of the square, h_N is the number of convex-hull vertices of N uniform
points and A_N the area of their convex hull.

METHOD (Theorems 4.11 and 4.13 of the paper)
--------------------------------------------
Separating-line identity.  For six i.i.d. absolutely continuous points, the
number of pairs (a,b) such that the line through P_a and Q_b weakly separates
conv{P} from conv{Q} equals 2 if the two triangles are disjoint and 0 otherwise.
Taking expectations and using exchangeability,

        1 - p_2 = 9 * E[ alpha^2 (1-alpha)^2 ].

Blaschke-Petkantschin.  For P_1,P_2 i.i.d. uniform in a convex body K of area 1,

        E[ g(line(P_1,P_2)) ] = (1/3) \int_0^pi \int_R ch(theta,p)^3 g dp dtheta,

ch = chord length.  For K = [0,1]^2 the 8-fold symmetry reduces theta to
(0,pi/4); with c = cos theta >= s = sin theta the chord and the area fraction
are piecewise polynomial in p, and the substitution t = tan theta turns the
theta-integrand into a polynomial on [0,1].

Two of the six evaluations below are known independently and serve as checks:
E[alpha] = 1/2 (symmetry) and E[alpha^2] = 133/432, which is equivalent to
Sylvester's value 25/36 for the probability that 4 uniform points in a square
are in convex position.

Usage:  python3 p2_exact.py
"""
import sympy as sp

p, q, C, S, t = sp.symbols('p q C S t', positive=True)


def moment(m, name):
    """(4/3) * int_0^{pi/4} int_0^{c+s} ch^3 * m(alpha) dp dtheta, exactly."""
    # piece 1: p in [0,S]   -- corner triangle at the origin
    ch1, al1 = p / (C * S), p ** 2 / (2 * C * S)
    I1 = sp.integrate(sp.expand(ch1 ** 3 * m(al1)), (p, 0, S))
    # piece 2: p in [S,C]   -- trapezoid
    ch2, al2 = 1 / C, (2 * p - S) / (2 * C)
    I2 = sp.integrate(sp.expand(ch2 ** 3 * m(al2)), (p, S, C))
    # piece 3: p in [C,C+S] -- opposite corner, q = C+S-p
    ch3, al3 = q / (C * S), 1 - q ** 2 / (2 * C * S)
    I3 = sp.integrate(sp.expand(ch3 ** 3 * m(al3)), (q, 0, S))

    J = sp.simplify(I1 + I2 + I3)
    Jt = sp.simplify(J.subs(S, C * t).subs(C, 1 / sp.sqrt(1 + t ** 2)))
    integrand = sp.simplify(sp.expand(sp.Rational(4, 3) * Jt / (1 + t ** 2)))
    val = sp.nsimplify(sp.simplify(sp.expand(sp.integrate(integrand, (t, 0, 1)))))
    print(f"  {name:<22} t-integrand = {integrand}")
    print(f"  {name:<22} value       = {val}   ({sp.N(val, 20)})")
    return val


def main():
    print("=== consistency checks ===")
    one = moment(lambda a: sp.Integer(1), "normalisation")
    assert one == 1, "Blaschke-Petkantschin normalisation failed"
    m1 = moment(lambda a: a, "E[alpha]")
    assert m1 == sp.Rational(1, 2), "symmetry of alpha failed"
    m2 = moment(lambda a: a ** 2, "E[alpha^2]")
    assert m2 == sp.Rational(133, 432), "Sylvester check failed"

    print("\n=== new moments ===")
    m3 = moment(lambda a: a ** 3, "E[alpha^3]")
    m4 = moment(lambda a: a ** 4, "E[alpha^4]")
    mX = moment(lambda a: a ** 2 * (1 - a) ** 2, "E[alpha^2(1-alpha)^2]")
    assert sp.simplify(m2 - 2 * m3 + m4 - mX) == 0, "moment expansion inconsistent"

    p2 = sp.simplify(1 - 9 * mX)
    print("\n=== results ===")
    print(f"  p_2                 = {p2}  =  {sp.N(p2, 20)}")
    print(f"  (2/pi for contrast) = {sp.N(2 / sp.pi, 20)}   difference {sp.N(p2 - 2/sp.pi, 3)}")
    assert p2 == sp.Rational(191, 300)

    h = {4: 12 * m2, 5: 20 * m3, 6: 30 * m4}
    for N in (4, 5, 6):
        print(f"  E[h_{N}]              = {sp.simplify(h[N])}   ({sp.N(h[N], 12)})")
    for N in (3, 4, 5):
        A = sp.simplify(1 - h[N + 1] / (N + 1))
        print(f"  E[A_{N}]              = {A}   ({sp.N(A, 12)})")

    print(f"\n  leading coefficient of f(n,2):  p_2/6^2 = {sp.Rational(191,300)/36}")


if __name__ == "__main__":
    main()
