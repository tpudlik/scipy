"""
An implementation of the confluent hypergeometric function.

"""

from __future__ import division
import numpy as np
from numpy import pi
from numpy.lib.scimath import sqrt
from . import gamma, rgamma, jv, gammaln, poch

import warnings

tol = 1.0e-15
BIGZ = 30

# Coefficients of the polynomial g appearing in the hyperasymptotic
# expansion of Paris (2013).
PARIS_G = np.vstack((np.array([0,0,0,0,0,0,0,0, -1, 2/3]),
                     np.array([0,0,0,0,0,0, -90, 270, -225, 46])/15,
                     np.array([0,0,0,0,-756, 5040, -11760, 11340, -3969,
                               230])/70,
                     np.array([0,0, -3240, 37800, -170100, 370440, -397530,
                               183330, -17781, -3226])/350,
                     np.array([-1069200, 19245600, -141134400, 541870560,
                               -1160830440, 1353607200, -743046480,
                               88280280, 43924815, -4032746])/231000))


def new_hyp1f1(a, b, z):
    """
    An implementation of the confluent hypergeometric function based on _[pop].

    References
    ----------
    ..[dlmf] Nist, DLMF
    ..[gst] Gil, Segura, Temme, 
            *Numerical Methods for Special Functions*,
            SIAM, 2007.
    ..[luke] Yudell L. Luke,
            *Algorithms for the Computation of Mathematical Functions",
            Academic Press, 1977
    ..[muller] Keith Muller,
            *Computing the confluent hypergeometric function, M(a, b, x)*,
            Numerische Mathematik 90(1) 179 (2001),
            http://link.springer.com/10.1007/s002110100285
    ..[pop] Pearson, Olver, Porter,
            *Numerical Methods for the Computation of the Confluent
            and Gauss Hypergeometric Functions*,
            http://arxiv.org/abs/1407.7786.

    """
    if b <= 0 and b == int(b):
        # Poles are located at 0, -1, -2, ...
        return np.nan + 1J*np.nan
    kummer = False
    if (a < 0 or b < 0) and not (a < 0 and b < 0):
        # Use Kummer's relation (3.19) in _[pop].
        a, z = b - a, -z
        kummer = True

    if a >= 0 and b >= 0 and z >= 0:
        res = hyp1f1_IA(a, b, z)
    elif a >= 0 and b >= 0 and z < 0:
        res = hyp1f1_IB(a, b, z)
    elif a < 0 and b < 0:
        res = hyp1f1_III(a, b, z)
    else:
        raise Exception("Shouldn't be able to get here!")

    if kummer:
        res *= np.exp(-z)
    return res


def hyp1f1_IA(a, b, z):
    """Compute hyp1f1 in the case where a, b >= 0 and z >= 0."""
    if np.abs(z) >= BIGZ:
        res = asymptotic_series(a, b, z)
    else:
        res = taylor_series(a, b, z)
    return res


def hyp1f1_IB(a, b, z):
    """Compute hyp1f1 in the case where a, b >= 0 and z < 0."""
    if a > b + 2:
        N = int(a - b)
        if np.abs(z) >= BIGZ:
            w0 = asymptotic_series(a, b + N + 1, z)
            w1 = asymptotic_series(a, b + N, z)
        else:
            w0 = taylor_series(a, b + N + 1, z)
            w1 = taylor_series(a, b + N, z)
        res = b_backward_recurrence(a, b + N, z, w0, w1, N)
    else:
        if np.abs(z) >= BIGZ:
            res = asymptotic_series(a, b, z)
        else:
            res = taylor_series(a, b, z)
    return res


def hyp1f1_III(a, b, z):
    """Compute hyp1f1 in the case where a, b < 0."""
    # Handle the special case where a is a negative integer, in which
    # case hyp1f1 is a polynomial
    if a == int(a):
        m = int(-a)
        term = 1
        res = 1
        for i in range(m):
            term *= z*(a + i)/((i + 1)*(b + i))
            res += term
    # Use the (++) recurrence to get to case IA or IB.
    else:
        N = int(max(-a, -b) + 1)
        if z >= 0:
            w0 = hyp1f1_IA(a + N + 1, b + N + 1, z)
            w1 = hyp1f1_IA(a + N, b + N, z)
        else:
            w0 = hyp1f1_IB(a + N + 1, b + N + 1, z)
            w1 = hyp1f1_IB(a + N, b + N, z)
        res = ab_backward_recurrence(a + N, b + N, z, w0, w1, N)
    return res


def a_forward_recurrence(a, b, z, w0, w1, N):
    """Use the recurrence relation (DLMF 13.3.1)

    (b-a)*1F1(a-1,b,z) + (2a-b+z)1F1(a,b,z) - a*1F1(a+1,b,z) = 0

    to compute 1F1(a+n,b,z) given w0 = 1F1(a-1,b,z) and w1 = 1F1(a,b,z).

    WARNING: 1F1 is the dominant solution of this recurrence relation
    *if* Re(z) > 0. In other words, don't use it if Re(z) is small or
    negative. Also, note that when Re(z) <= 0, 1F1 is *not* the
    minimal solution, so you can't use Miller/Olver's algorithm to
    recover stability. See [1] Chapter 4 for information on
    dominant/minimal solutions, and in particular Example 4.9 for
    1F1.

    """
    for i in xrange(N):
        tmp = w1
        w1 = ((b - a)*w0 + (2*a - b + z)*w1)/a
        w0 = tmp
        a += 1
    return w1


def b_backward_recurrence(a, b, z, w0, w1, N):
    """Use recurrence relation (3.14) from _[pop] to compute hyp1f1(a, b -
    N, z) given w0 = hyp1f1(a, b + 1, z) and w1 = hyp1f1(a, b, z).

    The minimal solution is gamma(b - a)*hyp1f1(a, b, z)/gamma(b), so
    it's safe to naively use the recurrence relation.

    """
    for i in range(N):
        tmp = w1
        w1 = -((z*(b - a))*w0 + b*(1 - b - z)*w1) / (b*(b - 1))
        w0 = tmp
        b -= 1
    return w1


def b_forward_recurrence(a, b, z, w0, N, tol):
    """Use the recurrence relation (3.14) from _[pop] to compute hyp1f1(a,
    b + N, z) given w0 = hyp1f1(a, b, z).

    The minimal solution is gamma(b - a)*hyp1f1(a, b, z)/gamma(b),
    so we use Olver's algorithm. Here we follow the notation from the
    DLMF 3.6.

    """
    # TODO: use the log of gamma to prevent blowup
    w0 *= gamma(b - a)*rgamma(b)
    p, e = [0, 1], [w0]
    curmin, n = 1e100, 1
    # Forward substitution
    while True:
        an, bn, cn = 1, -(1 - b - n - z)/z, (b - a + n - 1)/z
        p.append((bn*p[-1] - cn*p[-2])/an)
        e.append(cn*e[-1]/an)
        testmin = abs(e[-1]/(p[-2]*p[-1]))
        if n <= N:
            if testmin < curmin:
                curmin = testmin
        else:
            if testmin <= tol*curmin:
                break
        n += 1
    # Back substitution
    wn = 0
    for i in range(n, N, -1):
        wn = (p[i-1]*wn + e[i-1])/p[i]
    return rgamma(b + N - a)*gamma(b + N)*wn


def ab_backward_recurrence(a, b, z, w0, w1, N):
    """Use recurrence relation (3.14) from _[pop] to compute hyp1f1(a - N,
    b - N, z) given w0 = hyp1f1(a + 1, b + 1, z) and w1 = hyp1f1(a, b, z).

    The minimal solution is hyp1f1(a, b, z)/gamma(b), so it's safe to
    naively use the recurrence relation.

    """
    for i in range(N):
        tmp = w1
        w1 = (a*z*w0 + b*(b - z - 1)*w1) / (b*(b - 1))
        w0 = tmp
        a -= 1
        b -= 1
    return w1


def ab_forward_recurrence(a, b, z, w0, N, tol):
    """Use the recurrence relation (3.14) from _[pop] to compute hyp1f1(a
    + N, b + N, z) given w0 = hyp1f1(a, b, z).

    The minimal solution is hyp1f1(a, b, z)/gamma(b), so we use
    Olver's algorithm. Here we follow the notation from the DLMF 3.6.

    """
    w0 *= rgamma(b)
    p, e = [0, 1], [w0]
    curmin, n = 1e100, 1
    # Forward substitution
    while True:
        an, bn, cn = 1, -(b - z - 1 + n)/((a + n)*z), -1/((a + n)*z)
        p.append((bn*p[-1] - cn*p[-2])/an)
        e.append(cn*e[-1]/an)
        testmin = abs(e[-1]/(p[-2]*p[-1]))
        if n <= N:
            if testmin < curmin:
                curmin = testmin
        else:
            if testmin <= tol*curmin:
                break
        n += 1
    # Back substitution
    wn = 0
    for i in range(n, N, -1):
        wn = (p[i-1]*wn + e[i-1])/p[i]
    return gamma(b + N)*wn


def taylor_series(a, b, z, maxiters=500, tol=tol):
    """
    Compute hyp1f1 by evaluating the Taylor series directly.

    """
    Ao = 1
    So = Ao
    i = 0
    while i <= maxiters:
        An = Ao*(a + i)*z / ((b + i)*(i + 1))
        Sn = So + An
        if Sn != 0 and So != 0 and np.abs(An/Sn) < tol and np.abs(Ao/So) < tol:
            break
        else:
            So = Sn
            Ao = An
            i += 1
    if i > maxiters:
        warnings.warn("Number of evaluations exceeded maxiters on "
                      "a = {}, b = {}, z = {}.".format(a, b, z))
    return Sn


def single_fraction(a, b, z, maxiters=500, tol=tol):
    """Compute 1F1 by expanding the Taylor series as a single fraction and
    performing one division at the end. See section 3.3 of _[pop] for
    details.

    """
    # zeroth iteration
    alpha, beta, gamma = 0, 1, 1
    zetam = 1
    # first iteration
    i = 1
    alpha = (alpha + beta)*i*(b + i - 1)
    beta = beta*(a + i - 1)*z
    gamma = gamma*i*(b + i - 1)
    zetan = (alpha + beta) / gamma
    i = 2
    while i <= maxiters:
        alpha = (alpha + beta)*i*(b + i - 1)
        beta = beta*(a + i - 1)*z
        gamma = gamma*i*(b + i - 1)
        tmp = zetan
        zetan = (alpha + beta) / gamma
        zetao = zetam
        zetam = tmp
        if np.abs((zetan - zetam) / zetam) < tol and \
           np.abs((zetam - zetao) / zetao) < tol:
            break
        i += 1
    if i > maxiters:
        warnings.warn("Number of evaluations exceeded maxiters on "
                      "a = {}, b = {}, z = {}.".format(a, b, z))
    return zetan


def asymptotic_series(a, b, z, maxiters=500, tol=tol):
    """Compute hyp1f1 using an asymptotic series. This uses DLMF 13.7.2
    and DLMF 13.2.4. Note that the series is divergent (as one would
    expect); this can be seen by the ratio test.

    """
    if np.imag(z) == 0 and np.real(z) < 0:
        return paris_series(a, b, z, maxiters)

    phi = np.angle(z)
    if np.imag(z) == 0:
        expfac = np.cos(pi*a)
    elif phi > -0.5*pi and phi < 1.5*pi:
        expfac = np.exp(1J*pi*a)
    elif phi > -1.5*pi and phi <= -0.5*pi:
        expfac = np.exp(-1J*pi*a)
    else:
        raise Exception("Shouldn't be able to get here!")

    if np.real(a) and np.real(b) and np.real(z) and (a < 0 or b < 0):
        # gammaln will not give correct results for real negative
        # arguments, so the args must be cast to complex.
        c1 = np.real(np.exp(gammaln(b+0j) - gammaln(a+0j) + z))*z**(a - b)
    else:
        c1 = np.exp(gammaln(b) - gammaln(a) + z)*z**(a - b)
    if np.real(a) and np.real(b) and np.real(z) and (b - a < 0 or b < 0):
        c2 = np.real(np.exp(gammaln(b + 0j) - gammaln(b - a + 0j))*z**(-a))
    else:
        c2 = np.exp(gammaln(b) - gammaln(b - a))*z**(-a)

    # S1 is the first sum; the ith term is
    # (1 - a)_i * (b - a)_i * z^(-s) / i!
    # S2 is the second sum; the ith term is
    # (a)_i * (a - b + 1)_i * (-z)^(-s) / i!
    largest_term = 0
    previous_term = np.inf
    A1 = 1
    S1 = A1
    A2 = 1
    S2 = A2
    for i in range(1, maxiters + 1):
        A1 = A1*(i - a)*(b - a + i - 1) / (z*i)
        A2 = -A2*(a + i - 1)*(a - b + i) / (z*i)
        current_term = np.abs(c1*A1 + c2*A2)
        if current_term > largest_term:
            # Sometimes, the terms of the series increase initially.  We want
            # to keep summing until we're past the first maximum.
            largest_term = current_term
        elif current_term > previous_term:
            # We've passed the smallest term of the series: adding more will
            # only harm precision
            break
        current_sum = c1*S1 + c2*S2
        if c1*(S1 + A1) + c2*(S2 + A2) == current_sum or not np.isfinite(current_sum):
            break
        S1 += A1
        S2 += A2
        previous_term = current_term

    return c1*S1 + c2*S2


def paris_series(a, b, z, maxiters=200):
    """The exponentially improved asymptotic expansion along the negative real
    axis developed by Paris (2013).

    The exponentially improved expansion does not appear to improve the
    performance of the series in general, perhaps as a result of an
    implementation bug, so it has been commented out.

    """
    x = -z
    if np.real(a) and np.real(b) and (b < 0 or b - a < 0):
        c = np.exp(gammaln(b + 0j) - gammaln(b - a + 0j) - a*np.log(x))
        c = np.real(c)
    else:
        c = np.exp(gammaln(b) - gammaln(b - a) - a*np.log(x))

    theta = a - b
    if np.isreal(theta) and theta == np.floor(theta):
        if theta >= 0:
            # The hypergeometric function is a polynomial in n, so there are
            # no exponentially small corrections.
            if np.real(a) and np.real(b) and (b < 0 or a < 0):
                c = np.exp(-x + gammaln(a + 0j) - gammaln(b + 0j))
                c = np.real(c)
            else:
                c = np.exp(-x + gammaln(a) - gammaln(b))
            c = c * x**theta * (-1)**theta

            A1 = 1
            S1 = 1
            n = int(theta)
            for i in xrange(n + 1):
                A1 = A1*((1 - a + i)*(n - i)/(x*(i + 1)))
                S1 += A1

            return c*S1
        else:
            # We use the same prefactor c as in the general case, but sum
            # a fixed number of terms, not up to the smallest one.
            A1 = 1
            S1 = 1
            n = int(-theta)
            for i in xrange(n):
                A1 = A1*((a + i)*(1 + theta + i)/((i + 1)*x))
                S1 += A1

            return c*S1  # + paris_exponential_series(a, b, z, i, maxiters)

    A1 = 1
    S1 = 1
    previous_term = np.inf
    largest_term = 0
    for i in xrange(maxiters):
        A1 = A1*((a + i)*(1 + theta + i)/((i + 1)*x))
        current_term = np.abs(A1)
        if current_term > largest_term:
            largest_term = current_term
        elif current_term > previous_term:
            break
        S1 += A1
        previous_term = current_term

    return c*S1  # + paris_exponential_series(a, b, z, i, maxiters)


def paris_exponential_series(a, b, z, i, maxiters):
    """The exponentially small addition to the asymptotic expansion on the
    negative real axis.

    The argument `i` is the truncation index for the original series.

    This function does not reproduce all of the significant figures of Table 2
    in Paris (2013), suggesting a small implementation bug.

    Possibly fewer than all 5 terms should be summed for optimal peformance.
    (There are of course infinitely many terms, but the first 5 are the only
    ones for which the polynomial coefficients were included in the paper.)

    """
    M = 5
    theta = a - b
    x = -z
    if i < maxiters:
        # The optimal truncation term has been determined.
        v = a + i + theta
    else:
        # We don't know how many terms are optimal exactly (it's more than the
        # number of terms summed), so we'll use an approximation.
        v = x

    A = np.arange(M)
    A = poch(1 - a, A)*poch(b - a, A)/gamma(A + 1)

    first_sum = np.sum((-1)**np.arange(M) * A * x**(-np.arange(M)))
    second_sum = 0
    for idx in xrange(M):
        B = sum((-2)**k*poch(0.5, k)*A[idx-k]*np.polyval(PARIS_G[k,:],v - x - (idx-k))*6**(-2*k)
                for k in xrange(idx + 1))
        second_sum += (-1)**idx * B * x**(-idx)

    if np.real(a) and np.real(b) and (b < 0 or a < 0):
        c = np.exp(gammaln(b + 0j) - gammaln(a + 0j) - x + theta*np.log(x))
        c = np.real(c)
    else:
        c = np.exp(gammaln(b) - gammaln(a) - x + theta*np.log(x))

    return c*(np.cos(np.pi*theta)*first_sum
              - 2*np.sin(np.pi*theta)/np.sqrt(2*np.pi*x)*second_sum)


def bessel_series(a, b, z, maxiters=500, tol=tol):
    """Compute hyp1f1 using a series of Bessel functions; see (3.20) in
    _[pop].

    """
    Do, Dm, Dn = 1, 0, b/2
    r = sqrt(z*(2*b - 4*a))
    w = z**2 / r**(b + 1)
    # Ao is the first coefficient, and An is the second.
    Ao = 0
    # The first summand comes from the zeroth term.
    So = Do*jv(b - 1, r) / r**(b - 1) + Ao
    An = Dn*w*jv(b + 1, r)
    Sn = So + An
    i = 3
    while i <= maxiters:
        if np.abs(Ao/So) < tol and np.abs(An/Sn) < tol:
            break
        tmp = Dn
        Dn = ((i - 2 + b)*Dm + (2*a - b)*Do) / i
        Do = Dm
        Dm = tmp
        w *= z/r
        Ao = An
        An = Dn*w*jv(b - 1 + i, r)
        So = Sn
        Sn += An
        i += 1
    if i > maxiters:
        warnings.warn("Number of evaluations exceeded maxiters on "
                      "a = {}, b = {}, z = {}.".format(a, b, z))
    return gamma(b)*np.exp(z/2)*2**(b - 1)*Sn

def rational_approximation(a, b, z, maxiters=500, tol=tol):
    """Compute hyp1f1 using the rational approximation of _[luke], as described
    in _[muller].

    This method only works for real a, b, and z.

    """
    if z < 0:
        return np.exp(z)*rational_approximation(b - a, b, -z)

    d = b - a
    Q0 = 1
    Q1 = 1 + z*(d + 1)/(2*b)
    Q2 = 1 + z*(d + 2)/(2*(b + 1)) + z**2*(d + 1)*(d + 2)/(12*b*(b + 1))
    P0 = 1
    P1 = Q1 - z*d/b
    P2 = Q2 - (z*d/b)*(1 + z*(d + 2)/(2*(b + 1))) + z**2*d*(d + 1)/(2*b*(b + 1))

    def f1(i):
        return (i - d - 2)/(2*(2*i - 3)*(i + b - 1))

    def f2(i):
        num = (i + d)*(i + d - 1)
        den = 4*(2*i - 1)*(2*i - 3)*(i + b - 2)*(i + b - 1)
        return num/den

    def f3(i):
        num = (i + d - 2)*(i + d - 1)*(i - d - 2)
        den = 8*(2*i - 3)**2*(2*i - 5)*(i + b - 3)*(i + b - 2)*(i + b - 1)
        return -num/den

    def f4(i):
        num = (i + d - 1)*(i - b - 1)
        den = 2*(2*i - 3)*(i + b - 2)*(i + b - 1)
        return -num/den

    for i in xrange(3, maxiters):
        Pi = (1 + f1(i)*z)*P2 + (f4(i) + f2(i)*z)*z*P1 + f3(i)*z**3*P0
        Qi = (1 + f1(i)*z)*Q2 + (f4(i) + f2(i)*z)*z*Q1 + f3(i)*z**3*Q0
        Mi = Pi/Qi
        
        if not np.isfinite(Mi) or (P1 != 0 and np.abs(Q2*Mi/(P2) - 1) < 10**(-15)):
            return Mi*np.exp(z)

        P0 = P1
        P1 = P2
        P2 = Pi
        Q0 = Q1
        Q1 = Q2
        Q2 = Qi

    warnings.warn("Number of evaluations exceeded maxiters at "
                  "a = {}, b = {}, z = {}.".format(a, b, z))
    return Mi*np.exp(z)

