import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Deque, Dict, Set, Tuple
from collections import deque, defaultdict
from saintpy.inference import var1_vec, GP_log_pmf1

class ModelData:
    # Holds exactly the fields used by calculateScore()
    def __init__(self,
                 nprey, nbait,
                 test_mat_DATA, test_mat, test_mat1, test_SS,
                 n_rep_vec, p2p_mapping, ctrl_mean, n_ctrl_ip, apply_MRF,
                 ctrl_mat_DATA,
                 Z, beta0, beta1, gamma,
                 eta, d, lambda2_true, lambda2_false,
                 opts):
        self.nprey = nprey
        self.nbait = nbait
        self.test_mat_DATA = test_mat_DATA
        self.test_mat = test_mat
        self.test_mat1 = test_mat1
        self.test_SS = test_SS
        self.n_rep_vec = n_rep_vec
        self.p2p_mapping = p2p_mapping
        self.ctrl_mean = ctrl_mean
        self.n_ctrl_ip = n_ctrl_ip
        self.apply_MRF = apply_MRF
        self.ctrl_mat_DATA = ctrl_mat_DATA
        self.Z = Z
        self.beta0 = beta0
        self.beta1 = beta1
        self.gamma = gamma
        self.eta = eta
        self.d = d
        self.lambda2_true = lambda2_true
        self.lambda2_false = lambda2_false
        self.opts = opts

    # hook to the GP function you already defined
    def GP_log_pmf1(self, k, mean, lambda2):
        return GP_log_pmf1(k, mean, lambda2)
    
    def llikelihood(self):
        loglik = 0.0
        for i in range(self.nprey):
            for j in range(self.nbait):
                y = self.test_mat1[i][j]           # iterable of counts (len = m)
                if np.all(np.asarray(y) == 0):
                    continue
                k = self.Z[i][j]                   # boolean array (len = m)
                m = self.n_rep_vec[j]

                # gsum over prey neighbors if gamma != 0
                gsum = 0.0
                if self.gamma != 0.0:
                    for l in self.p2p_mapping[i]:
                        gsum += float(np.mean(self.Z[l][j]))

                MRFtrue  = math.exp(self.beta1 + self.gamma * gsum)
                MRFfalse = math.exp(self.beta0)

                prod = 1.0
                for rep in range(m):
                    # log pmfs as in C++
                    ll_true  = GP_log_pmf1(
                        min(y[rep], (self.eta[i] + self.d[i])),
                        (self.eta[i] + self.d[i]),
                        self.lambda2_true[i]
                    )
                    ll_false = GP_log_pmf1(
                        max(y[rep], self.eta[i]),
                        self.eta[i],
                        self.lambda2_false[i]
                    )
                    # choose branch based on k[rep]
                    num = (MRFtrue if k[rep] else MRFfalse)
                    ll  = (ll_true if k[rep] else ll_false)
                    prod *= (num / (MRFtrue + MRFfalse)) * math.exp(ll)

                loglik += math.log(prod) / m
        return loglik


# --------- faithful statModel port ---------------------------------------
def statModel(p2p_mapping, ubait, test_mat_DATA, ctrl_mat_DATA, ip_idx_to_bait_no, nprey, nbait, opts):
    """
    Python port of the C++ statModel() body you pasted, with identical logic.
    Parameters
    ----------
    p2p_mapping : list[list[int]]
    ubait : list[str]
    test_mat_DATA : np.ndarray shape (nprey, n_test_ip)
    ctrl_mat_DATA : np.ndarray shape (nprey, n_ctrl_ip)
    ip_idx_to_bait_no : list[int] length = n_test_ip, values in [0..nbait-1]
    nprey, nbait : int
    opts : Options with fields L, f, R
    """
    test_mat_DATA = np.asarray(test_mat_DATA, dtype=int)
    ctrl_mat_DATA = np.asarray(ctrl_mat_DATA, dtype=int)

    # bait index -> list of ip indices
    bait_no_to_ip_idxes = [[] for _ in range(len(ubait))]
    for i in range(len(ubait)):
        for j in range(len(ip_idx_to_bait_no)):
            if ip_idx_to_bait_no[j] == i:
                bait_no_to_ip_idxes[i].append(j)

    # n_rep_vec (unsigned char in C++; here plain int)
    n_rep_vec = [len(bait_no_to_ip_idxes[j]) for j in range(nbait)]

    # number of control IPs (columns)
    n_ctrl_ip = ctrl_mat_DATA.shape[1]
    L_used = min(n_ctrl_ip, opts.L)

    eta = np.zeros(nprey, dtype=float)
    d   = np.zeros(nprey, dtype=float)

    for i in range(nprey):
        ctrl = np.asarray(ctrl_mat_DATA[i, :], dtype=float)
        ctrl_sorted = np.sort(ctrl)[::-1]
        topL = ctrl_sorted[:L_used]

        tmp = float(np.mean(topL)) if topL.size > 0 else 0.0
        if tmp < 0.1:
            tmp = 0.1
        eta[i] = tmp
        d[i] = 4.0 * eta[i]  # SAINT default: μ=5η

    # build test_mat (original counts grouped by bait replicates)
    # and test_mat1 (on a copy test_mat_DATA1 identical to test_mat_DATA here)
    test_mat = [[None for _ in range(len(ubait))] for _ in range(nprey)]
    for i in range(nprey):
        for j in range(len(ubait)):
            reps = bait_no_to_ip_idxes[j]
            v = np.empty(len(reps), dtype=int)
            for k, ip_idx in enumerate(reps):
                v[k] = int(test_mat_DATA[i, ip_idx])
            test_mat[i][j] = v

    test_mat_DATA1 = np.array(test_mat_DATA, copy=True)
    test_mat1 = [[None for _ in range(len(ubait))] for _ in range(nprey)]
    for i in range(nprey):
        for j in range(len(ubait)):
            reps = bait_no_to_ip_idxes[j]
            v = np.empty(len(reps), dtype=int)
            for k, ip_idx in enumerate(reps):
                v[k] = int(test_mat_DATA1[i, ip_idx])
            test_mat1[i][j] = v

    # write back test_mat1 into test_mat_DATA1 (identical here)
    for i in range(nprey):
        for j in range(len(ubait)):
            reps = bait_no_to_ip_idxes[j]
            for k, ip_idx in enumerate(reps):
                test_mat_DATA1[i, ip_idx] = int(test_mat1[i][j][k])

    # sufficient statistics (sum across replicates per bait)
    test_SS1 = np.zeros((nprey, len(ubait)), dtype=int)
    for i in range(nprey):
        for j in range(len(ubait)):
            test_SS1[i, j] = int(np.sum(test_mat1[i][j]))

    # initialize Z shape (nprey, nbait) of bool arrays sized by n_rep_vec[j]
    Z = [[None for _ in range(nbait)] for _ in range(nprey)]
    for i in range(nprey):
        for j in range(nbait):
            Z[i][j] = np.zeros(n_rep_vec[j], dtype=bool)

    for i in range(nprey):
        for j in range(nbait):
            for rep in range(len(bait_no_to_ip_idxes[j])):
                Z[i][j][rep] = (test_mat1[i][j][rep] > (2.0 * eta[i]))

    # control means per prey (mean across all control IPs)
    ctrl_mean = [0.0] * nprey
    for i in range(ctrl_mat_DATA.shape[0]):
        ctrl_mean[i] = float(np.sum(ctrl_mat_DATA[i, :])) / float(ctrl_mat_DATA.shape[1])

    # apply_MRF (valarray<bool> in C++; here list[bool])
    apply_MRF = [False] * nprey
    for i in range(nprey):
        s = 0
        for bait_idx in range(nbait):
            s += int(np.sum(Z[i][bait_idx]))
        if s < test_mat_DATA.shape[1] * opts.f:
            apply_MRF[i] = True

    # lambda2_false per prey using variance of top-L controls
    lambda2_false = [0.1] * nprey
    for i in range(nprey):
        ctrl = np.array(ctrl_mat_DATA[i, :], dtype=int)
        ctrl_sorted = np.sort(ctrl)[::-1]
        topL = ctrl_sorted[:L_used]
        variance = var1_vec(topL)  # sample variance
        if variance > eta[i]:
            lambda2_false[i] = 1.0 - math.sqrt(eta[i] / variance)

    # lambda2_true default 0.1 per C++
    lambda2_true = [0.1] * nprey

    beta0 = 0.0
    beta1 = 0.0
    gamma = 0.0

    return ModelData(
        nprey=nprey, nbait=nbait,
        test_mat_DATA=test_mat_DATA,     # original test matrix
        test_mat=test_mat,               # grouped replicates (list of np.array)
        test_mat1=test_mat1,             # transformed (same here)
        test_SS=test_SS1,                # sums across reps
        n_rep_vec=n_rep_vec,
        p2p_mapping=p2p_mapping,
        ctrl_mean=ctrl_mean,
        n_ctrl_ip=ctrl_mat_DATA.shape[1],
        ctrl_mat_DATA=ctrl_mat_DATA,
        apply_MRF=apply_MRF,
        Z=Z,
        beta0=beta0, beta1=beta1, gamma=gamma,
        eta=eta, d=d,
        lambda2_true=lambda2_true,
        lambda2_false=lambda2_false,
        opts=opts
    )


def calculateScore(model):
    """
    Trustful port of Model_data::calculateScore().
    Returns (average_score, maximum_score, min_log_odds_score),
    where average_score is 'avgp' in the original code.
    """
    nprey = model.nprey
    nbait = model.nbait

    average_score      = [[0.0 for _ in range(nbait)] for _ in range(nprey)]
    min_log_odds_score = [[0.0 for _ in range(nbait)] for _ in range(nprey)]
    maximum_score      = [[0.0 for _ in range(nbait)] for _ in range(nprey)]

    for i in range(nprey):
        for j in range(nbait):
            y = model.test_mat1[i][j]  # 1-D iterable of length m
            if np.all(np.asarray(y) == 0):  # emulate all_of(..., iszero)
                continue

            m = model.n_rep_vec[j]

            # gsum over p2p_mapping[i]
            gsum = 0.0
            for l in model.p2p_mapping[i]:
                # Z(l, j).mean() in C++; here Z[l][j] is a 1-D bool array
                z_lj = model.Z[l][j]
                gsum += float(np.mean(z_lj))

            MRFtrue  = math.exp(model.beta1 + model.gamma * gsum)
            MRFfalse = math.exp(model.beta0)

            # collect per-replicate scores
            tmp_scores = [0.0] * m
            tmp_odds_scores = [0.0] * m

            for rep in range(m):
                yrep = y[rep]

                # tmp_mean logic identical to C++
                if yrep >= 5:
                    tmp_mean = min(model.eta[i] + model.d[i], 5.0 * model.eta[i])
                else:
                    tmp_mean = (model.eta[i] + model.d[i])

                # ----- compute in log-space to avoid underflow -----
                # log(MRFtrue) and log(MRFfalse)
                log_true  = model.beta1 + model.gamma * gsum
                log_false = (
                    model.beta0
                    + model.GP_log_pmf1(max(yrep, model.eta[i]), model.eta[i], model.lambda2_false[i])
                    - model.GP_log_pmf1(min(yrep, tmp_mean),      tmp_mean,      model.lambda2_true[i])
                )

                # log-odds (same quantity as C++: log(unnorm_true) - log(unnorm_false))
                tmp_odds_scores[rep] = log_true - log_false

                # probability = unnorm_true / (unnorm_true + unnorm_false)
                # = 1 / (1 + exp(log_false - log_true))  (stable logistic)
                if (yrep <= 1) or (yrep <= model.ctrl_mean[i]):
                    tmp_scores[rep] = 0.0
                else:
                    delta = log_false - log_true
                    if   delta > 50:   # extreme favoring "false"
                        tmp_scores[rep] = 0.0
                    elif delta < -50:  # extreme favoring "true"
                        tmp_scores[rep] = 1.0
                    else:
                        tmp_scores[rep] = 1.0 / (1.0 + math.exp(delta))

            # sort descending like std::sort(..., greater<>())
            tmp_scores.sort(reverse=True)
            tmp_odds_scores.sort(reverse=True)

            max_rep = model.opts.R
            if m > max_rep:
                average_score[i][j] = sum(tmp_scores[:max_rep]) / float(max_rep)
            else:
                average_score[i][j] = sum(tmp_scores) / float(m)

            # back() after descending sort is the minimum log-odds
            min_log_odds_score[i][j] = tmp_odds_scores[-1]
            maximum_score[i][j]      = tmp_scores[0] if m > 0 else 0.0

            # if max count is 1, set avg and max to zero
            if np.max(y) == 1:
                average_score[i][j] = 0.0
                maximum_score[i][j] = 0.0

    return average_score, maximum_score, min_log_odds_score

