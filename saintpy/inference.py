import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

# log_factorial table and maximum index threshold (must be initialized beforehand)
# Example:
log_factorial_max = 255  # or whatever the C++ constant is
log_factorial_table = [math.lgamma(i + 1) for i in range(log_factorial_max + 1)]

def llik_MRF_gamma_0(model, x, _grad_ignored=None):
    beta0_ = 0.0
    beta1_ = x[0]
    loglik = 0.0
    MRFtrue  = math.exp(beta1_)
    MRFfalse = math.exp(beta0_)
    for i in range(model.nprey):
        for j in range(model.nbait):
            y = model.test_mat1[i][j]
            if np.all(np.asarray(y) == 0):
                continue
            k = model.Z[i][j]                   # bool array length m
            m = model.n_rep_vec[j]
            prod = 1.0
            for rep in range(m):
                prod *= (MRFtrue if k[rep] else MRFfalse) / (MRFtrue + MRFfalse)
            loglik += math.log(prod) / m
    return loglik

# --- 2) NLOpt wrapper behavior: maximize f over bounds; fallback if no gain ---
# We'll use a simple golden-section *maximizer* to mirror "maximize" semantics.
def _maximize_1d(f, lo, hi, tol=1e-4, maxiter=100):
    phi = (1 + 5**0.5) / 2
    invphi = 1 / phi
    invphi2 = 1 - invphi  # = 1/phi^2
    a, b = float(lo), float(hi)
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc = f(c)
    fd = f(d)
    it = 0
    while (b - a) > tol and it < maxiter:
        it += 1
        if fc >= fd:   # keep [a,d]
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = f(c)
        else:          # keep [c,b]
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = f(d)
    xbest = c if fc >= fd else d
    fbest = fc if fc >= fd else fd
    return xbest, fbest

def wrt_MRF_gamma_0(model):
    # mirrors:
    #   oldbeta1 = beta1
    #   oldf = llik_MRF_gamma_0([beta1])
    #   maximize over [-15, 15] with tol=1e-4, maxeval=1e2
    oldbeta1 = model.beta1
    oldf = llik_MRF_gamma_0(model, [oldbeta1], None)

    def f(b1):
        return llik_MRF_gamma_0(model, [b1], None)

    xopt, maxf = _maximize_1d(f, -15.0, 15.0, tol=1e-4, maxiter=100)
    model.beta1 = float(xopt)
    if maxf < oldf:    # fallback if no improvement
        model.beta1 = oldbeta1

# --- 3) ICMS driver (γ=0 path), exactly like your C++ icms() ---
def icms(model, with_gamma=False, max_iter=15):
    newllik = model.llikelihood()
    model.gamma = 0.0
    for _ in range(max_iter):
        oldllik = newllik
        icm_Z(model)                   # <—— this one
        if with_gamma:
            model.wrt_MRF()
        else:
            wrt_MRF_gamma_0(model)
        newllik = model.llikelihood()
        if newllik >= oldllik and (math.exp(newllik - oldllik) - 1) < 1e-3:
            break
    return model


def wrt_MRF_gamma_0(model):
    oldbeta1 = model.beta1
    oldf = llik_MRF_gamma_0(model, [oldbeta1])   # pass model explicitly
    def f(b1): return llik_MRF_gamma_0(model, [b1])
    xopt, maxf = _maximize_1d(f, -15, 15, tol=1e-4, maxiter=100)
    model.beta1 = xopt if maxf >= oldf else oldbeta1


def icm_Z(model):
    """Exact port of Model_data::icm_Z() — one iteration over all Z bits."""
    nprey, nbait = model.nprey, model.nbait
    pre_calc_loglik = [[None for _ in range(nbait)] for _ in range(nprey)]

    for i in range(nprey):
        for j in range(nbait):
            y = model.test_mat1[i][j]
            m = model.n_rep_vec[j]
            pre_calc_loglik[i][j] = np.zeros((m, 2))
            for rep in range(m):
                pre_calc_loglik[i][j][rep, 0] = model.GP_log_pmf1(
                    min(y[rep], (model.eta[i] + model.d[i])),
                    (model.eta[i] + model.d[i]),
                    model.lambda2_true[i]
                )
                pre_calc_loglik[i][j][rep, 1] = model.GP_log_pmf1(
                    max(y[rep], model.eta[i]),
                    model.eta[i],
                    model.lambda2_false[i]
                )

    # --- single pass updating every replicate bit ---
    for i in range(nprey):
        for j in range(nbait):
            m = model.n_rep_vec[j]
            for rep in range(m):
                first = loglikelihood_Z(model, i, j, rep, pre_calc_loglik)
                model.Z[i][j][rep] = not model.Z[i][j][rep]     # flip
                second = loglikelihood_Z(model, i, j, rep, pre_calc_loglik)
                if first > second:                               # revert if worse
                    model.Z[i][j][rep] = not model.Z[i][j][rep]


def loglikelihood_Z(model, i, j, rep, pre_calc_loglik):
    """Faithful port of Model_data::loglikelihood_Z()."""
    k = model.Z[i][j]
    gsum = 0.0
    if model.gamma != 0.0:
        for l in model.p2p_mapping[i]:
            gsum += float(np.mean(model.Z[l][j]))
    logMRFtrue  = model.beta1 + model.gamma * gsum
    logMRFfalse = model.beta0
    pcl_true, pcl_false = pre_calc_loglik[i][j][rep]
    return logMRFtrue + pcl_true if k[rep] else logMRFfalse + pcl_false


def log_factorial(n: int) -> float:
    """Equivalent of inline double log_factorial(unsigned short n)."""
    if n > log_factorial_max:
        return math.lgamma(n + 1)
    else:
        return log_factorial_table[n]


def GP_log_pmf(x: int, lambda1: float, lambda2: float) -> float:
    """Generalized Poisson log PMF, matching C++ logic."""
    if x == 0:
        return -lambda1
    tmp = lambda1 + x * lambda2
    return math.log(lambda1) + (x - 1) * math.log(tmp) - tmp - log_factorial(x)


def _lfact(n: int) -> float:
    return math.lgamma(n + 1)


def GP_log_pmf1(x: int, mean: float, lam2: float) -> float:
    if mean <= 0.0 or x < 0:
        return -1e12
    lam1 = mean * (1.0 - lam2)
    t = lam1 + lam2 * x
    if lam1 <= 0.0 or t <= 0.0:
        return -1e12
    if x == 0:
        return -lam1
    return math.log(lam1) + (x - 1) * math.log(t) - t - _lfact(x)


def var1_vec(v):
    v = np.asarray(v, dtype=float)
    n = v.size
    if n <= 1:
        return 0.0
    m = v.mean()
    return np.sum((v - m) * (v - m)) / (n - 1.0)



