# Diagnose the factor approximation: separate "is the factor MODEL good" (exact Gauss-Hermite
# integral vs oracle) from "is the 2nd-order EXPANSION of it good" (2nd-order vs GH). If GH tracks
# the oracle but 2nd-order doesn't, the loadings are simply too strong for a parabola and we need a
# few quadrature nodes instead. Also shows how many GH nodes are needed.
import os, sys
os.environ.setdefault('SESSION_SECRET_KEY', 'test-only-session-secret')
sys.path.insert(0, 'testing_files')

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from benchmark_helpers import client, _build_session_request
from backend.state.session import get_session
from backend.math.algorithm_agents import _normal_ppf, _normal_pdf, _normal_cdf
from backend.math.algorithm_helpers import (compute_win_probability, calculate_correction_terms,
                                            calculate_tipping_points, calculate_pair_bracket_matrix)

# R once (project to PSD)
mc = get_session(client.post('/sessions', json=_build_session_request('Head to Head: Most Categories')).json()['session_id']).agent
R_raw = np.asarray(mc.rho[0], float)
ev, evec = np.linalg.eigh((R_raw + R_raw.T) / 2)
R = (evec * np.clip(ev, 1e-6, None)) @ evec.T
d = np.sqrt(np.diag(R)); R = R / np.outer(d, d)
n = R.shape[0]; I = np.eye(n); thr = n // 2 + 1
L = np.linalg.cholesky(R)

fev, fvec = np.linalg.eigh(R)
lam = np.clip(fvec[:, -1] * np.sqrt(max(float(fev[-1]), 0.0)), -0.98, 0.98)
E = (R - I) - np.outer(lam, lam); np.fill_diagonal(E, 0.0)
print('leading eigenvalue', round(float(fev[-1]), 3), ' loadings', np.round(lam, 2))
print('||E||_F', round(float(np.sqrt((E**2).sum())), 3), ' ||R-I||_F', round(float(np.sqrt(((R-I)**2).sum())), 3))
root = np.sqrt(1 - lam**2)

def residual_fo(rate):
    z = _normal_ppf(rate)
    return float(calculate_correction_terms(rate.reshape(1, n, 1), E, _normal_pdf(z).reshape(1, n, 1))[0][0, 0])

def factor_2nd(rate):
    z = _normal_ppf(rate); a = z / root; b = lam / root; beta = b * _normal_pdf(a)
    p0 = _normal_cdf(a).reshape(1, n, 1)
    g0 = float(compute_win_probability(p0)[0, 0])
    Vc = calculate_tipping_points(p0)[0, :, 0]
    B = calculate_pair_bracket_matrix(p0)[0, :, :, 0].copy(); np.fill_diagonal(B, 0.0)
    return g0 + 0.5 * float(beta @ B @ beta) - 0.5 * float(np.sum(Vc * a * b * beta)) + residual_fo(rate)

def factor_probit(rate):
    # Approximate the sigmoid g(u) ~ Phi(alpha + kappa u) by matching value and slope at u=0, then
    # integrate exactly: E_U[Phi(alpha+kappa U)] = Phi(alpha/sqrt(1+kappa^2)). ~2 DP evals, analytic.
    z = _normal_ppf(rate); a = z / root; b = lam / root; beta = b * _normal_pdf(a)
    p0 = _normal_cdf(a).reshape(1, n, 1)
    g0 = float(compute_win_probability(p0)[0, 0])
    g0 = min(max(g0, 1e-6), 1 - 1e-6)
    gp0 = float(np.sum(calculate_tipping_points(p0)[0, :, 0] * beta))   # g'(0) = sum_c V_c beta_c
    alpha = _normal_ppf(g0)
    kappa = gp0 / _normal_pdf(alpha)
    return float(_normal_cdf(alpha / np.sqrt(1 + kappa**2))) + residual_fo(rate)

def factor_gh(rate, K):
    z = _normal_ppf(rate); nodes, w = hermegauss(K); w = w / np.sqrt(2 * np.pi)
    total = 0.0
    for u, wi in zip(nodes, w):
        p = _normal_cdf((z + lam * u) / root).reshape(1, n, 1)
        total += wi * float(compute_win_probability(p)[0, 0])
    return total + residual_fo(rate)

def g_of_u(rate, u):
    z = _normal_ppf(rate)
    p = _normal_cdf((z + lam * u) / root).reshape(1, n, 1)
    return float(compute_win_probability(p)[0, 0])

def verify(rate, label):
    # numerical g'(0), g''(0) vs analytic, plus the term breakdown that shows the blow-up
    h = 1e-3
    g0n = g_of_u(rate, 0.0)
    gpp_num = (g_of_u(rate, h) - 2*g0n + g_of_u(rate, -h)) / h**2
    z = _normal_ppf(rate); a = z/root; b = lam/root; beta = b*_normal_pdf(a)
    p0 = _normal_cdf(a).reshape(1, n, 1)
    Vc = calculate_tipping_points(p0)[0, :, 0]
    B = calculate_pair_bracket_matrix(p0)[0, :, :, 0].copy(); np.fill_diagonal(B, 0.0)
    term_pairs = 0.5 * float(beta @ B @ beta); term_self = -0.5*float(np.sum(Vc*a*b*beta))
    gpp_analytic = 2*term_pairs + 2*term_self
    print(f"\n[{label}] g(0)={g0n*100:.1f}  g''(0) analytic={gpp_analytic:.3f} numeric={gpp_num:.3f}"
          f"  (match={abs(gpp_analytic-gpp_num)<1e-2})")
    print(f"   2nd-order terms: g0={g0n*100:.1f}  +term_pairs={term_pairs*100:+.1f}  "
          f"+term_self={term_self*100:+.1f}  -> {(g0n+term_pairs+term_self)*100:.1f}pp "
          f"(a parabola over U~N(0,1) is meaningless when |1/2 g''| >> spread)")

request = _build_session_request('Head to Head: Each Category'); request['data_source']['season'] = '2025-26'
agent = get_session(client.post('/sessions', json=request).json()['session_id']).agent
rates = agent._default_result['Rates']
top = [p for p in agent.default_h_scores.index if p in rates.index][:12]
rng = np.random.default_rng(1)

verify(np.clip(rates.loc[top[0]].to_numpy(float), 1e-4, 1-1e-4), top[0].split(' (')[0])

print(f"\n{'player':22s} {'oracle':>7} {'1st':>7} {'probit':>7} {'GH5':>7} {'GH9':>7} {'GH15':>7}")
errs = {k: [] for k in ['1st','probit','GH5','GH9','GH15']}
for player in top:
    rate = np.clip(rates.loc[player].to_numpy(float), 1e-4, 1-1e-4)
    z = _normal_ppf(rate); probs = rate.reshape(1, n, 1); pdf = _normal_pdf(z).reshape(1, n, 1)
    first = float(compute_win_probability(probs)[0, 0]) + float(calculate_correction_terms(probs, R - I, pdf)[0][0, 0])
    draws = z[:, None] + L @ rng.standard_normal((n, 500000))
    oracle = float(((draws > 0).sum(0) >= thr).mean())
    vals = {'1st': first, 'probit': factor_probit(rate), 'GH5': factor_gh(rate, 5),
            'GH9': factor_gh(rate, 9), 'GH15': factor_gh(rate, 15)}
    for k in errs: errs[k].append(abs(vals[k] - oracle))
    print(f"{player.split(' (')[0]:22s} {oracle*100:7.1f} {first*100:7.1f} {vals['probit']*100:7.1f} "
          f"{vals['GH5']*100:7.1f} {vals['GH9']*100:7.1f} {vals['GH15']*100:7.1f}")
print('\nmean |err| (pp): ' + '  '.join(f'{k} {100*np.mean(v):.2f}' for k, v in errs.items()))
