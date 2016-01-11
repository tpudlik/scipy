"""
An implementation of the confluent hypergeometric function based on [3].

Sources
[1] Gil, Segura, Temme, "Numerical Methods for Special Functions"
[2] Nist, DLMF
[3] Pearson, Olver, and Porter "Computation of Hypergeometric Functions",
"Numerical Methods for the Computation of the Confluent and Gauss Hypergeometric Functions",
http://arxiv.org/abs/1407.7786.

"""

from __future__ import division
import numpy as np
from numpy import pi
from numpy.lib.scimath import sqrt
from . import gamma, rgamma, jv

import warnings

tol = 1.0e-15
BIGPARAM = 4 # should be > 2
BIGZ = 30


def new_hyp1f1(a, b, z):
    """Compute the confluent hypergeometric function 1F1."""
    if b <= 0 and b == int(b):
        # Poles are located at 0, -1, -2, ...
        return np.inf
    elif b < -BIGPARAM:
        n = int(np.abs(b) - BIGPARAM) + 2
        b = b + n
        if a < -BIGPARAM:
            raise NotImplementedError
        elif a > BIGPARAM:
            raise NotImplementedError
        else:
            w0 = hyp1f1_small_parameters(a, b + 1, z)
            w1 = hyp1f1_small_parameters(a, b, z)
            return b_backwards_recursion(a, b, z, w0, w1, n)
    elif b > BIGPARAM:
        raise NotImplementedError
    elif a < -BIGPARAM:
        raise NotImplementedError
    elif a > BIGPARAM:
        raise NotImplementedError
    else:
        return hyp1f1_small_parameters(a, b, z)


def hyp1f1_small_parameters(a, b, z):
    """Compute 1F1 when |a|, |b| < BIGPARAM.

    WARNING: this function does not check for poles.

    """
    assert np.abs(a) <= BIGPARAM and np.abs(b) <= BIGPARAM
    if np.abs(z) < BIGZ:
        return taylor_series(a, b, z)
    else:
        return asymptotic_series(a, b, z)


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
    """Use recurrence relation (3.14) from [3] to compute hyp1f1(a, b -
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
    """Use the recurrence relation (3.14) from [3] to compute
    hyp1f1(a, b + N, z) given w0 = hyp1f1(a, b, z).

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
    """Use recurrence relation (3.14) from [3] to compute hyp1f1(a - N, b
    - N, z) given w0 = hyp1f1(a + 1, b + 1, z) and w1 = hyp1f1(a, b, z).

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
    """Use the recurrence relation (3.14) from [3] to compute
    hyp1f1(a + N, b + N, z) given w0 = hyp1f1(a, b, z).

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
        if np.abs(An/Sn) < tol and np.abs(Ao/So) < tol:
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
    """Compute 1F1 by expanding the Taylor series as a single fraction
    and performing one division at the end. See section 3.3 of [3] for
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
    """Compute 1F1 using an asymptotic series. This uses DLMF 13.7.2 and
    DLMF 13.2.4. Note that the series is divergent (as one would
    expect); this can be seen by the ratio test.

    """
    # S1 is the first sum; the ith term is
    # (1 - a)_i * (b - a)_i * z^(-s) / i!
    # S2 is the second sum; the ith term is
    # (a)_i * (a - b + 1)_i * (-z)^(-s) / i!
    A1 = 1
    S1 = A1
    A2 = 1
    S2 = A2
    # Is 8 terms optimal? Not sure.
    for i in xrange(1, 9):
        A1 = A1*(i - a)*(b - a + i - 1) / (z*i)
        S1 += A1
        A2 = -A2*(a + i - 1)*(a - b + i) / (z*i)
        S2 += A2

    x = np.real(z)
    y = np.imag(z)
    if np.allclose(x, 0) and y > 0:
        phi = 0.5*pi
    elif np.allclose(x, 0) and y < 0:
        phi = -0.5*pi
    else:
        phi = np.arctan(y/x)

    if np.allclose(phi, 0):
        expfac = np.cos(pi*a)
    elif phi > -0.5*pi and phi < 1.5*pi:
        expfac = np.exp(1J*pi*a)
    elif phi > -1.5*pi and phi <= -0.5*pi:
        expfac = np.exp(-1J*pi*a)
    else:
        raise Exception("Shouldn't be able to get here!")

    c1 = np.exp(z)*z**(a - b)*rgamma(a)
    c2 = expfac*z**(-a)*rgamma(b - a)
    return gamma(b)*(c1*S1 + c2*S2)


def bessel_series(a, b, z, maxiters=500, tol=tol):
    """Compute 1F1 using a series of Bessel functions; see (3.20) in
    [3].

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
