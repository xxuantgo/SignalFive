# -*- coding: utf-8 -*-
"""
因子测试模块
============
职责：
  1. 计算周度 Rank IC（因为策略按周调仓，因子预测力应在周度频率上评估）
  2. 计算 IC 均值、标准差、ICIR、p 值、胜率
  3. 提供滚动 IC 序列用于合成权重
  4. 输出单因子测试报告

关键设计：
  - Rank IC = Spearman(因子截面排序, 未来收益截面排序)
  - 前瞻收益使用 T+5 日收益（≈一周持有期），与调仓频率匹配
  - 每天都算截面 IC（而非仅在调仓日），但测试结果按周度频率汇报
"""
import numpy as np
import pandas as pd
from scipy import stats
from signalfive.config import (
    FORWARD_RETURN_PERIODS, IC_ROLLING_WINDOW, IC_MIN_OBS,
)


def compute_forward_returns(close_matrix: pd.DataFrame,
                             periods: list = None) -> dict:
    """
    计算未来 N 日收益率矩阵

    Parameters
    ----------
    close_matrix : pd.DataFrame
        收盘价矩阵 (date × sec)
    periods : list[int]
        前瞻天数列表, 默认 [5]

    Returns
    -------
    dict[int, pd.DataFrame]
        {天数: 未来收益矩阵(date × sec)} 的字典
        第 t 行 = close[t+n] / close[t] - 1
    """
    periods = periods or FORWARD_RETURN_PERIODS
    fwd_rets = {}
    for n in periods:
        fwd = close_matrix.shift(-n) / close_matrix - 1
        fwd_rets[n] = fwd
    return fwd_rets


def calc_daily_rank_ic(factor_matrix: pd.DataFrame,
                        fwd_return_matrix: pd.DataFrame,
                        min_obs: int = None) -> pd.Series:
    """
    计算日度截面 Rank IC（Spearman 相关系数）

    每个交易日 t：
      IC_t = Spearman_corr(factor_t 的截面排序, fwd_ret_t 的截面排序)

    Parameters
    ----------
    factor_matrix : pd.DataFrame
        因子值矩阵 (date × sec)
    fwd_return_matrix : pd.DataFrame
        前瞻收益矩阵 (date × sec)
    min_obs : int
        最少有效观测数，少于此数的日期返回 NaN

    Returns
    -------
    pd.Series
        index=date, values=当日 Rank IC
    """
    min_obs = min_obs or IC_MIN_OBS

    # 对齐日期和标的
    common_dates = factor_matrix.index.intersection(fwd_return_matrix.index)
    common_secs = factor_matrix.columns.intersection(fwd_return_matrix.columns)
    factor = factor_matrix.loc[common_dates, common_secs]
    fwd_ret = fwd_return_matrix.loc[common_dates, common_secs]

    ic_series = pd.Series(index=common_dates, dtype=float, name="rank_ic")

    for dt in common_dates:
        f = factor.loc[dt].dropna()
        r = fwd_ret.loc[dt].dropna()
        common = f.index.intersection(r.index)
        if len(common) < min_obs:
            ic_series[dt] = np.nan
            continue
        corr, _ = stats.spearmanr(f[common], r[common])
        ic_series[dt] = corr

    return ic_series


def calc_daily_rank_ic_fast(factor_matrix: pd.DataFrame,
                             fwd_return_matrix: pd.DataFrame,
                             min_obs: int = None) -> pd.Series:
    """    
    向量化版本的日度截面 Rank IC，性能远优于逐日循环。

    原理：对因子和收益分别做截面 rank，然后逐行计算 Pearson 相关
    （rank 后的 Pearson = Spearman）。

    Parameters
    ----------
    factor_matrix: pd.DataFrame
        因子值矩阵 (日期×证券)
    fwd_return_matrix: pd.DataFrame
        前瞻收益矩阵 (日期×证券)
    min_obs: int
        最少有效观测数，默认使用全局配置中的IC_MIN_OBS（默认值为10）
    """
    min_obs = min_obs or IC_MIN_OBS

    common_dates = factor_matrix.index.intersection(fwd_return_matrix.index)
    common_secs = factor_matrix.columns.intersection(fwd_return_matrix.columns)
    factor = factor_matrix.loc[common_dates, common_secs]
    fwd_ret = fwd_return_matrix.loc[common_dates, common_secs]

    # 截面 rank
    f_rank = factor.rank(axis=1)
    r_rank = fwd_ret.rank(axis=1)

    # 有效观测数
    valid_count = factor.notna() & fwd_ret.notna()
    n = valid_count.sum(axis=1)

    # 将 NaN 设为 0（不影响相关系数计算，因为我们后续用 mask）
    f_rank_filled = f_rank.fillna(0)
    r_rank_filled = r_rank.fillna(0)

    # 均值去中心化
    mask = valid_count.astype(float).replace(0, np.nan)
    f_mean = (f_rank_filled * mask).sum(axis=1) / n
    r_mean = (r_rank_filled * mask).sum(axis=1) / n

    f_dem = f_rank_filled.sub(f_mean, axis=0) * mask
    r_dem = r_rank_filled.sub(r_mean, axis=0) * mask

    # Pearson(rank) = Spearman
    cov = (f_dem * r_dem).sum(axis=1)
    f_std = (f_dem ** 2).sum(axis=1).pow(0.5)
    r_std = (r_dem ** 2).sum(axis=1).pow(0.5)

    ic = cov / (f_std * r_std)
    ic[n < min_obs] = np.nan

    ic.name = "rank_ic"
    return ic


def calc_ic_summary(ic_series: pd.Series) -> dict:
    """
    从 IC 时间序列计算摘要统计量

    Returns
    -------
    dict with keys:
      - mean_ic: IC 均值
      - std_ic: IC 标准差
      - icir: IC 信息比率 = mean_ic / std_ic
      - p_value: 双侧 t 检验 p 值（H0: mean_ic=0）
      - win_rate: IC > 0 的比例
      - count: 有效观测数
      - sig_05: 是否在 5% 水平显著 (p < 0.05)
      - sig_01: 是否在 1% 水平显著 (p < 0.01)
    """
    ic = ic_series.dropna()
    n = len(ic)
    if n == 0:
        return {k: np.nan for k in
                ["mean_ic", "std_ic", "icir", "p_value", "win_rate", "count",
                 "sig_05", "sig_01"]}

    mean_ic = ic.mean()
    std_ic = ic.std()
    icir = mean_ic / std_ic if std_ic > 0 else np.nan
    t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 else np.nan
    # 双侧 p 值：H0 为 IC 均值=0
    if np.isnan(t_stat) or n <= 1:
        p_value = np.nan
    else:
        p_value = 2 * stats.t.sf(np.abs(t_stat), df=n - 1)
    win_rate = (ic > 0).mean()

    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "icir": icir,
        "p_value": p_value,
        "win_rate": win_rate,
        "count": n,
        "sig_05": p_value < 0.05 if not np.isnan(p_value) else False,
        "sig_01": p_value < 0.01 if not np.isnan(p_value) else False,
    }


def test_all_factors(factor_matrices: dict,
                      close_matrix: pd.DataFrame,
                      periods: list = None) -> tuple:
    """
    对所有截面因子进行单因子测试

    Parameters
    ----------
    factor_matrices : dict[str, pd.DataFrame]
        {因子名: 因子值矩阵(date × sec)}
    close_matrix : pd.DataFrame
        收盘价矩阵 (date × sec)
    periods : list[int]
        前瞻天数列表

    Returns
    -------
    summary_df : pd.DataFrame
        因子测试摘要表 (因子 × 统计量)
    ic_series_dict : dict[str, pd.Series]
        {因子名: IC 时间序列}，用于后续合成权重
    """
    periods = periods or FORWARD_RETURN_PERIODS
    fwd_rets = compute_forward_returns(close_matrix, periods)

    rows = []
    ic_series_dict = {}

    for name, factor_mat in sorted(factor_matrices.items()):
        for n, fwd_ret in fwd_rets.items():
            ic = calc_daily_rank_ic_fast(factor_mat, fwd_ret)
            summary = calc_ic_summary(ic)
            summary["factor"] = name
            summary["period"] = f"ret_{n}d"
            rows.append(summary)

            # 存储 IC 序列（仅保留主要周期，用于合成）
            if n == periods[0]:
                ic_series_dict[name] = ic

    summary_df = pd.DataFrame(rows)
    col_order = ["factor", "period", "mean_ic", "std_ic", "icir",
                 "p_value", "win_rate", "count", "sig_05", "sig_01"]
    summary_df = summary_df[[c for c in col_order if c in summary_df.columns]]

    return summary_df, ic_series_dict


def select_effective_factors(summary_df: pd.DataFrame,
                              min_abs_ic: float = 0.02,
                              max_p: float = 0.05) -> list:
    """
    筛选有效因子：|mean_IC| > min_abs_ic 且 p_value <= max_p

    Returns
    -------
    list[str]
        有效因子名列表
    """
    # 取主要周期的结果
    primary = summary_df[summary_df["period"] == f"ret_{FORWARD_RETURN_PERIODS[0]}d"]
    mask = (primary["mean_ic"].abs() > min_abs_ic) & (primary["p_value"] <= max_p)
    effective = primary.loc[mask, "factor"].tolist()
    return effective


def select_effective_factors_from_ic(ic_series_dict: dict,
                                     min_abs_ic: float = 0.02,
                                     max_p: float = 0.05,
                                     cutoff: str = None) -> tuple[list, pd.DataFrame]:
    """
    直接从 IC 序列筛选有效因子，可按 cutoff 限定样本区间。
    筛选条件：|mean_IC| > min_abs_ic 且 p_value <= max_p。
    """
    rows = []
    cutoff_ts = pd.Timestamp(cutoff) if cutoff else None
    for factor_name, ic in sorted(ic_series_dict.items()):
        ic_use = ic.loc[:cutoff_ts] if cutoff_ts is not None else ic
        summary = calc_ic_summary(ic_use)
        summary["factor"] = factor_name
        summary["period"] = f"ret_{FORWARD_RETURN_PERIODS[0]}d"
        rows.append(summary)

    summary_df = pd.DataFrame(rows)
    col_order = ["factor", "period", "mean_ic", "std_ic", "icir",
                 "p_value", "win_rate", "count", "sig_05", "sig_01"]
    summary_df = summary_df[[c for c in col_order if c in summary_df.columns]]

    mask = (summary_df["mean_ic"].abs() > min_abs_ic) & (summary_df["p_value"] <= max_p)
    effective = summary_df.loc[mask, "factor"].tolist()
    return effective, summary_df


if __name__ == "__main__":
    from data_loader import load_all
    from factor_calc import compute_factors, prepare_factor_matrices

    data = load_all()
    panel_wide, macro_df = compute_factors(data["aligned"])
    processed = prepare_factor_matrices(panel_wide, method="rank")

    summary_df, ic_series_dict = test_all_factors(processed, data["close_matrix"])
    print("\n单因子测试结果:")
    print(summary_df.to_string(index=False))

    effective = select_effective_factors(summary_df)
    print(f"\n有效因子 ({len(effective)} 个): {effective}")
