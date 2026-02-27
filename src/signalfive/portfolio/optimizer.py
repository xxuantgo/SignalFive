# -*- coding: utf-8 -*-
"""
组合优化模块
============
提供两类权重生成器：
  1) 最小方差权重
  2) 风险平价权重
  3) CVaR 最小化权重（条件风险价值）
  4) CVaR / 风险平价混合权重
均满足：
  - 无空仓约束：w >= 0
  - 单资产权重上限：max_weight
  - 权重和为 1
  - 至少持有 min_holdings 只资产
"""
import numpy as np
import pandas as pd
from scipy import optimize, stats

from signalfive.config import (
    SHRINKAGE_FACTOR,
    MAX_SINGLE_WEIGHT,
    MIN_HOLDINGS,
    CVAR_ALPHA,
    CVAR_TURNOVER_LAMBDA,
    HYBRID_BETA,
)


# ---------------------------------------------------------------------------
# 协方差矩阵处理
# ---------------------------------------------------------------------------

def shrink_cov(cov: pd.DataFrame, shrinkage: float = None) -> pd.DataFrame:
    """对协方差矩阵做简单收缩（向对角线收缩），提升数值稳定性。"""
    shrinkage = SHRINKAGE_FACTOR if shrinkage is None else shrinkage
    cov_np = cov.to_numpy()
    diag = np.diag(np.diag(cov_np))
    shrunk = shrinkage * diag + (1 - shrinkage) * cov_np
    return pd.DataFrame(shrunk, index=cov.index, columns=cov.columns)


# ---------------------------------------------------------------------------
# 约束与后处理
# ---------------------------------------------------------------------------

def _post_process_weights(w: np.ndarray,
                          assets: list,
                          max_weight: float,
                          min_holdings: int) -> pd.Series:
    """
    - 去负数，按上限截断
    - 确保至少持有 min_holdings 只资产
    - 归一化到和为 1
    """
    n = len(w)
    if n == 0:
        return pd.Series(dtype=float)
    if max_weight * n < 1 - 1e-12:
        raise ValueError(f"不可行约束: n={n}, max_weight={max_weight}, 无法满足 sum(w)=1")

    w = np.maximum(w, 0.0)
    if w.sum() <= 0:
        w = np.ones_like(w)

    # 若持仓数不足，先在原权重最大的 min_holdings 只资产上分配初始权重
    if (w > 1e-10).sum() < min_holdings:
        k = min(min_holdings, n)
        idx = np.argsort(w)[-k:]
        base = np.zeros_like(w)
        base[idx] = 1.0 / k
        w = base

    w = _project_to_capped_simplex(w, max_weight=max_weight)

    # 数值稳健性检查
    if np.max(w) > max_weight + 1e-8:
        raise ValueError("权重后处理失败：存在超过 max_weight 的资产权重")
    if abs(w.sum() - 1.0) > 1e-8:
        raise ValueError("权重后处理失败：权重和不为 1")
    if (w > 1e-10).sum() < min_holdings:
        raise ValueError("权重后处理失败：持仓数不足 min_holdings")

    return pd.Series(w, index=assets)


def _project_to_capped_simplex(w: np.ndarray,
                               max_weight: float,
                               tol: float = 1e-12,
                               max_iter: int = 100) -> np.ndarray:
    """
    投影到约束集合：
      - w_i >= 0
      - w_i <= max_weight
      - sum(w) = 1
    """
    w = np.maximum(w.astype(float), 0.0)
    if w.sum() <= tol:
        w = np.ones_like(w, dtype=float)
    w = w / w.sum()

    for _ in range(max_iter):
        over = w > (max_weight + tol)
        if not over.any():
            break

        w[over] = max_weight
        free = ~over
        remain = 1.0 - w[over].sum()
        if remain <= 0:
            # 极端数值场景，退化为均匀后继续迭代
            w = np.ones_like(w, dtype=float) / len(w)
            continue

        free_sum = w[free].sum()
        if free_sum <= tol:
            w[free] = remain / max(1, free.sum())
        else:
            w[free] = w[free] / free_sum * remain

    # 最终数值修正
    w = np.clip(w, 0.0, max_weight)
    shortfall = 1.0 - w.sum()
    if abs(shortfall) > 1e-10:
        free = w < (max_weight - tol)
        if free.any():
            free_sum = w[free].sum()
            if free_sum <= tol:
                w[free] += shortfall / free.sum()
            else:
                w[free] += shortfall * (w[free] / free_sum)
        else:
            w = w / w.sum()
            w = np.minimum(w, max_weight)
            w = w / w.sum()

    return w


# ---------------------------------------------------------------------------
# 最小方差
# ---------------------------------------------------------------------------

def min_variance_weights(cov: pd.DataFrame,
                         max_weight: float = None,
                         min_holdings: int = None) -> pd.Series:
    """解最小方差优化：min w^T C w, s.t. sum w = 1, 0<=w<=max_weight"""
    max_weight = MAX_SINGLE_WEIGHT if max_weight is None else max_weight
    min_holdings = MIN_HOLDINGS if min_holdings is None else min_holdings

    n = len(cov)
    assets = list(cov.index)
    cov_np = cov.to_numpy()

    def obj(w):
        return w @ cov_np @ w

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
    bounds = [(0.0, max_weight) for _ in range(n)]
    x0 = np.ones(n) / n

    res = optimize.minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons)
    if not res.success:
        # 失败时退化为等权并截断上限
        w = x0
    else:
        w = res.x

    return _post_process_weights(w, assets, max_weight, min_holdings)


# ---------------------------------------------------------------------------
# 风险平价
# ---------------------------------------------------------------------------

def risk_parity_weights(cov: pd.DataFrame,
                        max_weight: float = None,
                        min_holdings: int = None) -> pd.Series:
    """
    风险平价：最小化各资产风险贡献的相对离散程度。
    使用对数壁障 + 相对偏差目标，确保数值稳定。
    """
    max_weight = MAX_SINGLE_WEIGHT if max_weight is None else max_weight
    min_holdings = MIN_HOLDINGS if min_holdings is None else min_holdings

    n = len(cov)
    assets = list(cov.index)
    cov_np = cov.to_numpy()

    def obj(w):
        # 风险贡献 RC_i = w_i * (Σ w)_i
        cw = cov_np @ w
        rc = w * cw
        avg_rc = rc.mean()
        if avg_rc < 1e-20:
            return 0.0
        # 使用相对偏差（scale-invariant），避免极小绝对值导致的数值问题
        return ((rc / avg_rc - 1.0) ** 2).sum()

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
    bounds = [(0.0, max_weight) for _ in range(n)]
    x0 = np.ones(n) / n

    res = optimize.minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons)
    if not res.success:
        w = x0
    else:
        w = res.x

    return _post_process_weights(w, assets, max_weight, min_holdings)


# ---------------------------------------------------------------------------
# CVaR 最小化
# ---------------------------------------------------------------------------

def _normalize_cvar_method(method: str | None) -> str:
    method = "empirical" if method is None else str(method).strip().lower()
    alias = {
        "lp": "empirical",
        "historical": "empirical",
        "gaussian": "parametric",
        "cornish-fisher": "cornish_fisher",
        "cf": "cornish_fisher",
    }
    method = alias.get(method, method)
    valid = {"empirical", "parametric", "cornish_fisher"}
    if method not in valid:
        raise ValueError(f"未知 cvar_method={method}，可选: {sorted(valid)}")
    return method


def _moment_based_cvar_loss(port_ret: np.ndarray, alpha: float, method: str) -> float:
    """
    基于矩估计的 CVaR 近似损失（以 loss 口径最小化）：
      loss = -E[r] + ES_tail
    method:
      - parametric: 高斯假设
      - cornish_fisher: 对分位点做偏度/峰度修正
    """
    x = np.asarray(port_ret, dtype=float)
    if x.size < 3:
        return float("inf")

    mu = float(np.nanmean(x))
    sigma = float(np.nanstd(x, ddof=1))
    sigma = max(sigma, 1e-8)
    tail_prob = max(1.0 - float(alpha), 1e-8)

    z = float(stats.norm.ppf(alpha))
    if method == "cornish_fisher":
        centered = x - mu
        m2 = float(np.mean(centered ** 2))
        if m2 <= 1e-16:
            skew = 0.0
            ex_kurt = 0.0
        else:
            m3 = float(np.mean(centered ** 3))
            m4 = float(np.mean(centered ** 4))
            skew = m3 / (m2 ** 1.5)
            ex_kurt = m4 / (m2 ** 2) - 3.0
        z = (
            z
            + (z ** 2 - 1.0) * skew / 6.0
            + (z ** 3 - 3.0 * z) * ex_kurt / 24.0
            - (2.0 * z ** 3 - 5.0 * z) * (skew ** 2) / 36.0
        )
        z = float(np.clip(z, -5.0, 8.0))

    tail_term = float(stats.norm.pdf(z)) / tail_prob
    return -mu + sigma * tail_term


def _cvar_weights_empirical(ret_use: pd.DataFrame,
                            alpha: float,
                            max_weight: float,
                            min_holdings: int,
                            prev_weights: pd.Series | np.ndarray | None,
                            turnover_lambda: float | None) -> pd.Series:
    """原始经验 CVaR（历史场景 LP）。"""
    assets = list(ret_use.columns)
    t_obs, n = ret_use.shape
    if n == 0 or t_obs == 0:
        return pd.Series(dtype=float)
    if max_weight * n < 1 - 1e-12:
        raise ValueError(f"不可行约束: n={n}, max_weight={max_weight}, 无法满足 sum(w)=1")

    r = ret_use.to_numpy()
    use_turnover_penalty = (turnover_lambda is not None) and (turnover_lambda > 0) and (prev_weights is not None)
    if use_turnover_penalty:
        if isinstance(prev_weights, pd.Series):
            prev_series = prev_weights
        else:
            prev_series = pd.Series(prev_weights)
        prev = prev_series.reindex(assets).fillna(0.0).to_numpy(dtype=float)
        prev = np.clip(prev, 0.0, None)
    else:
        prev = None

    # 变量顺序（无换手惩罚）: [w_1...w_n, z, u_1...u_T]
    # 变量顺序（含换手惩罚）: [w_1...w_n, z, u_1...u_T, d+_1...d+_n, d-_1...d-_n]
    n_var = n + 1 + t_obs + (2 * n if use_turnover_penalty else 0)
    c = np.zeros(n_var)
    c[n] = 1.0
    u_start = n + 1
    u_end = u_start + t_obs
    c[u_start:u_end] = 1.0 / ((1.0 - alpha) * t_obs)
    if use_turnover_penalty:
        d_plus_start = n + 1 + t_obs
        d_minus_start = d_plus_start + n
        c[d_plus_start:d_plus_start + n] = turnover_lambda
        c[d_minus_start:d_minus_start + n] = turnover_lambda

    # 约束: -r_t @ w - z - u_t <= 0
    a_ub = np.zeros((t_obs, n_var))
    a_ub[:, :n] = -r
    a_ub[:, n] = -1.0
    a_ub[:, u_start:u_end] = -np.eye(t_obs)
    b_ub = np.zeros(t_obs)

    # 等式约束:
    # 1) sum(w)=1
    # 2) 含换手惩罚时，w_i - prev_i = d+_i - d-_i
    if use_turnover_penalty:
        a_eq = np.zeros((1 + n, n_var))
        a_eq[0, :n] = 1.0
        b_eq = np.zeros(1 + n)
        b_eq[0] = 1.0
        for i in range(n):
            a_eq[1 + i, i] = 1.0
            a_eq[1 + i, d_plus_start + i] = -1.0
            a_eq[1 + i, d_minus_start + i] = 1.0
            b_eq[1 + i] = prev[i]
    else:
        a_eq = np.zeros((1, n_var))
        a_eq[0, :n] = 1.0
        b_eq = np.array([1.0])

    bounds = [(0.0, max_weight)] * n + [(None, None)] + [(0.0, None)] * t_obs
    if use_turnover_penalty:
        bounds += [(0.0, None)] * (2 * n)
    res = optimize.linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not res.success or res.x is None:
        w = np.ones(n) / n
    else:
        w = res.x[:n]
    return _post_process_weights(w, assets, max_weight, min_holdings)


def _cvar_weights_moment_based(ret_use: pd.DataFrame,
                               alpha: float,
                               max_weight: float,
                               min_holdings: int,
                               prev_weights: pd.Series | np.ndarray | None,
                               turnover_lambda: float | None,
                               method: str) -> pd.Series:
    """参数化 / Cornish-Fisher CVaR（矩估计 + 非线性优化）。"""
    assets = list(ret_use.columns)
    t_obs, n = ret_use.shape
    if n == 0 or t_obs == 0:
        return pd.Series(dtype=float)
    if max_weight * n < 1 - 1e-12:
        raise ValueError(f"不可行约束: n={n}, max_weight={max_weight}, 无法满足 sum(w)=1")

    r = ret_use.to_numpy()
    use_turnover_penalty = (turnover_lambda is not None) and (turnover_lambda > 0) and (prev_weights is not None)
    if use_turnover_penalty:
        if isinstance(prev_weights, pd.Series):
            prev_series = prev_weights
        else:
            prev_series = pd.Series(prev_weights)
        prev = prev_series.reindex(assets).fillna(0.0).to_numpy(dtype=float)
        prev = np.clip(prev, 0.0, None)
    else:
        prev = None

    def obj(w: np.ndarray) -> float:
        port_ret = r @ w
        loss = _moment_based_cvar_loss(port_ret=port_ret, alpha=alpha, method=method)
        if use_turnover_penalty:
            loss += float(turnover_lambda) * float(np.abs(w - prev).sum())
        return float(loss)

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, max_weight)] * n
    if use_turnover_penalty and prev is not None and prev.sum() > 1e-12:
        x0 = _project_to_capped_simplex(prev, max_weight=max_weight)
    else:
        x0 = np.ones(n) / n

    res = optimize.minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons)
    if not res.success or res.x is None:
        w = x0
    else:
        w = res.x
    return _post_process_weights(w, assets, max_weight, min_holdings)


def cvar_weights(ret_window: pd.DataFrame,
                 alpha: float = None,
                 max_weight: float = None,
                 min_holdings: int = None,
                 prev_weights: pd.Series = None,
                 turnover_lambda: float = None,
                 cvar_method: str = "empirical") -> pd.Series:
    """
    统一 CVaR 权重入口：
      - empirical: 历史场景 CVaR（Rockafellar-Uryasev 线性规划）
      - parametric: 高斯参数化 CVaR
      - cornish_fisher: 偏度/峰度修正的参数化 CVaR
    """
    alpha = CVAR_ALPHA if alpha is None else alpha
    max_weight = MAX_SINGLE_WEIGHT if max_weight is None else max_weight
    min_holdings = MIN_HOLDINGS if min_holdings is None else min_holdings
    turnover_lambda = CVAR_TURNOVER_LAMBDA if turnover_lambda is None else turnover_lambda
    cvar_method = _normalize_cvar_method(cvar_method)

    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha 必须在 (0,1) 区间内，当前为 {alpha}")

    ret_use = ret_window.dropna(how="all", axis=1).dropna(how="any", axis=0)
    if cvar_method == "empirical":
        return _cvar_weights_empirical(
            ret_use=ret_use,
            alpha=alpha,
            max_weight=max_weight,
            min_holdings=min_holdings,
            prev_weights=prev_weights,
            turnover_lambda=turnover_lambda,
        )
    return _cvar_weights_moment_based(
        ret_use=ret_use,
        alpha=alpha,
        max_weight=max_weight,
        min_holdings=min_holdings,
        prev_weights=prev_weights,
        turnover_lambda=turnover_lambda,
        method=cvar_method,
    )


def hybrid_cvar_rp_weights(ret_window: pd.DataFrame,
                           beta: float = None,
                           alpha: float = None,
                           max_weight: float = None,
                           min_holdings: int = None,
                           shrinkage: float = None,
                           prev_weights: pd.Series = None,
                           turnover_lambda: float = None,
                           cvar_method: str = "empirical") -> pd.Series:
    """
    混合优化器：CVaR + 风险平价（凸组合）。
      w_final = (1 - beta) * w_cvar + beta * w_rp
      beta=0 -> 纯 CVaR；beta=1 -> 纯风险平价
    """
    beta = HYBRID_BETA if beta is None else beta
    alpha = CVAR_ALPHA if alpha is None else alpha
    max_weight = MAX_SINGLE_WEIGHT if max_weight is None else max_weight
    min_holdings = MIN_HOLDINGS if min_holdings is None else min_holdings
    shrinkage = SHRINKAGE_FACTOR if shrinkage is None else shrinkage
    turnover_lambda = CVAR_TURNOVER_LAMBDA if turnover_lambda is None else turnover_lambda

    beta = float(np.clip(beta, 0.0, 1.0))

    ret_use = ret_window.dropna(how="all", axis=1).dropna(how="any", axis=0)
    assets = list(ret_use.columns)
    if len(assets) == 0:
        return pd.Series(dtype=float)

    w_cvar = cvar_weights(
        ret_use,
        alpha=alpha,
        max_weight=max_weight,
        min_holdings=min_holdings,
        prev_weights=prev_weights,
        turnover_lambda=turnover_lambda,
        cvar_method=cvar_method,
    ).reindex(assets).fillna(0.0)

    cov = compute_cov_from_returns(ret_use, shrinkage=shrinkage)
    w_rp = risk_parity_weights(
        cov,
        max_weight=max_weight,
        min_holdings=min_holdings,
    ).reindex(assets).fillna(0.0)

    w_mix = (1.0 - beta) * w_cvar.to_numpy() + beta * w_rp.to_numpy()
    return _post_process_weights(w_mix, assets, max_weight=max_weight, min_holdings=min_holdings)


# ---------------------------------------------------------------------------
# 辅助：根据历史收益矩阵计算协方差
# ---------------------------------------------------------------------------

def compute_cov_from_returns(ret_window: pd.DataFrame,
                             shrinkage: float = None) -> pd.DataFrame:
    """从收益窗口计算协方差并收缩"""
    cov = ret_window.cov()
    return shrink_cov(cov, shrinkage)


__all__ = [
    "compute_cov_from_returns",
    "min_variance_weights",
    "risk_parity_weights",
    "cvar_weights",
    "hybrid_cvar_rp_weights",
]
