# -*- coding: utf-8 -*-
"""
因子合成模块
============
职责：
  1. 根据各因子的历史 IC/ICIR 加权合成综合因子
  2. 支持多种合成方式：等权、IC 加权、ICIR 加权
  3. 确定因子方向（IC 为负的因子需要反转）
  4. 输出合成因子序列 (date × sec)，供后续选股使用

关键设计：
  - 滚动窗口合成：每天用过去 N 天的 IC 序列计算权重，避免前视偏差
  - 因子方向自动判定：IC 均值为负 → 原始值取反
  - 合成后再做截面排序，输出 [0, 1] 的排序分数
"""
import numpy as np
import pandas as pd
from signalfive.config import (
    COMBINE_METHOD, COMBINE_ROLLING_WINDOW, FORWARD_RETURN_PERIODS,
    BACKTEST_START,
    COMBINE_ROBUST_CORR_PENALTY, COMBINE_ROBUST_TURNOVER_SMOOTH,
)

def determine_factor_directions(ic_series_dict: dict,
                                  lookback: int = None,
                                  cutoff: str = None) -> dict:
    """
    判定每个因子的方向：IC 均值 > 0 → 正向(+1)，否则反向(-1)

    Parameters
    ----------
    ic_series_dict : dict[str, pd.Series]
        {因子名: IC 时间序列}（应已做前视偏差位移）
    lookback : int
        如果指定，只用最近 lookback 天的 IC 判定方向
    cutoff : str
        截止日期字符串；只用此日期之前的 IC 判定方向，
        避免使用回测区间内的未来信息。

    Returns
    -------
    dict[str, int]
        {因子名: +1 或 -1}
    """
    directions = {}
    for name, ic in ic_series_dict.items():
        ic_use = ic.copy()
        if cutoff:
            ic_use = ic_use.loc[:pd.Timestamp(cutoff)]
        if lookback:
            ic_use = ic_use.iloc[-lookback:]
        mean_ic = ic_use.dropna().mean()
        directions[name] = 1 if np.isnan(mean_ic) or mean_ic >= 0 else -1
    return directions


def _calc_quality_score_from_prestart(ic_df: pd.DataFrame, cutoff: str | None) -> pd.Series:
    """
    基于回测前样本估计因子静态质量分：
      quality = |ICIR| * (0.5 + 0.5 * tanh(|t_stat|/2))
    """
    ic_use = ic_df.copy()
    if cutoff:
        ic_use = ic_use.loc[:pd.Timestamp(cutoff)]

    mean_ic = ic_use.mean(axis=0)
    std_ic = ic_use.std(axis=0).replace(0, np.nan)
    n_obs = ic_use.count(axis=0).replace(0, np.nan)

    abs_icir = (mean_ic / std_ic).abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    t_stat = (mean_ic / (std_ic / np.sqrt(n_obs))).abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sig = np.tanh(t_stat / 2.0)
    quality = abs_icir * (0.5 + 0.5 * sig)
    return quality.fillna(0.0)


def _apply_corr_penalty(ic_df: pd.DataFrame,
                        quality: pd.Series,
                        cutoff: str | None,
                        corr_penalty_lambda: float) -> pd.Series:
    """
    用回测前样本相关性惩罚冗余因子：
      q_adj_i = q_i / (1 + lambda * sum_j |corr_ij| * q_j)
    """
    ic_use = ic_df.copy()
    if cutoff:
        ic_use = ic_use.loc[:pd.Timestamp(cutoff)]

    corr_abs = ic_use.corr().abs().fillna(0.0)
    if corr_abs.empty:
        return quality

    np.fill_diagonal(corr_abs.values, 0.0)
    q = quality.reindex(corr_abs.columns).fillna(0.0)
    redundancy = corr_abs.dot(q)
    q_adj = q / (1.0 + float(corr_penalty_lambda) * redundancy)
    return q_adj.reindex(quality.index).fillna(0.0)


def _normalize_rowwise(raw_weights: pd.DataFrame) -> pd.DataFrame:
    row_sum = raw_weights.sum(axis=1).replace(0, np.nan)
    weights = raw_weights.div(row_sum, axis=0)
    # 行内全 NaN 时回退等权，避免后续合成出现整行缺失
    nan_rows = weights.isna().all(axis=1)
    if nan_rows.any():
        n_cols = weights.shape[1]
        weights.loc[nan_rows, :] = 1.0 / n_cols if n_cols > 0 else np.nan
    return weights.fillna(0.0)


def calc_rolling_ic_weights(ic_series_dict: dict,
                              method: str = None,
                              window: int = None,
                              cutoff: str = None,
                              corr_penalty_lambda: float = None,
                              turnover_smooth: float = None) -> pd.DataFrame:
    """
    计算滚动因子权重

    Parameters
    ----------
    ic_series_dict : dict[str, pd.Series]
        {因子名: IC 时间序列}
    method : str
        "equal" : 等权
        "ic"    : 按滚动 IC 均值加权
        "icir"  : 按滚动 ICIR (IC均值/IC标准差) 加权
        "icir_robust": 在滚动 ICIR 上叠加显著性质量分、相关性惩罚与时间平滑
    window : int
        滚动窗口（交易日）

    Returns
    -------
    pd.DataFrame
        index=date, columns=因子名, values=归一化后的权重
    """
    method = method or COMBINE_METHOD
    window = window or COMBINE_ROLLING_WINDOW
    corr_penalty_lambda = (
        COMBINE_ROBUST_CORR_PENALTY if corr_penalty_lambda is None else float(corr_penalty_lambda)
    )
    turnover_smooth = (
        COMBINE_ROBUST_TURNOVER_SMOOTH if turnover_smooth is None else float(turnover_smooth)
    )

    # 拼接所有因子的 IC 序列为 DataFrame
    ic_df = pd.DataFrame(ic_series_dict)
    ic_df = ic_df.sort_index()

    if method == "equal":
        # 等权：不随时间变化
        weights = pd.DataFrame(
            1.0 / len(ic_df.columns),
            index=ic_df.index,
            columns=ic_df.columns,
        )
        return weights

    elif method == "ic":
        # 滚动 IC 均值，取绝对值作为权重
        rolling_mean = ic_df.rolling(window, min_periods=20).mean()
        raw_weights = rolling_mean.abs()

    elif method == "icir":
        # 滚动 ICIR = IC均值 / IC标准差，取绝对值作为权重
        rolling_mean = ic_df.rolling(window, min_periods=20).mean()
        rolling_std = ic_df.rolling(window, min_periods=20).std()
        raw_weights = (rolling_mean / rolling_std.replace(0, np.nan)).abs()

    elif method == "icir_robust":
        # 动态层：滚动 ICIR
        rolling_mean = ic_df.rolling(window, min_periods=20).mean()
        rolling_std = ic_df.rolling(window, min_periods=20).std()
        dynamic_score = (rolling_mean / rolling_std.replace(0, np.nan)).abs()

        # 静态层：回测前样本质量分 + 冗余惩罚
        quality = _calc_quality_score_from_prestart(ic_df, cutoff=cutoff)
        quality = _apply_corr_penalty(
            ic_df,
            quality=quality,
            cutoff=cutoff,
            corr_penalty_lambda=corr_penalty_lambda,
        )
        raw_weights = dynamic_score.mul(quality, axis=1)

    else:
        raise ValueError(f"未知的合成方法: {method}")

    weights = _normalize_rowwise(raw_weights)

    # 时间平滑：仅使用历史权重，抑制权重抖动
    if method == "icir_robust":
        alpha = float(np.clip(turnover_smooth, 1e-3, 1.0))
        weights = weights.ewm(alpha=alpha, adjust=False).mean()
        weights = _normalize_rowwise(weights)

    return weights


def combine_factors(factor_matrices: dict,
                     ic_series_dict: dict,
                     effective_factors: list = None,
                     method: str = None,
                     window: int = None,
                     corr_penalty_lambda: float = None,
                     turnover_smooth: float = None) -> pd.DataFrame:
    """
    合成综合因子

    流程：
      1. 筛选有效因子
      2. 判定因子方向
      3. 计算滚动权重
      4. 加权求和得到合成因子
      5. 输出截面排序分数 [0, 1]

    Parameters
    ----------
    factor_matrices : dict[str, pd.DataFrame]
        {因子名: 因子值矩阵 (date × sec)}，已经过截面标准化
    ic_series_dict : dict[str, pd.Series]
        {因子名: IC 时间序列}
    effective_factors : list[str]
        有效因子列表；若为 None 则使用全部
    method : str
        合成方法
    window : int
        滚动窗口

    Returns
    -------
    pd.DataFrame
        合成因子矩阵 (date × sec)，值为截面排序分数 [0, 1]
    """
    method = method or COMBINE_METHOD

    # 筛选有效因子
    if effective_factors:
        use_factors = [f for f in effective_factors if f in factor_matrices and f in ic_series_dict]
    else:
        use_factors = [f for f in factor_matrices if f in ic_series_dict]

    if not use_factors:
        raise ValueError("没有可用的因子进行合成！")

    print(f"合成因子使用 {len(use_factors)} 个因子: {use_factors}")
    print(f"合成方法: {method}, 滚动窗口: {window or COMBINE_ROLLING_WINDOW}")
    if (method or COMBINE_METHOD) == "icir_robust":
        cpl = COMBINE_ROBUST_CORR_PENALTY if corr_penalty_lambda is None else corr_penalty_lambda
        tsm = COMBINE_ROBUST_TURNOVER_SMOOTH if turnover_smooth is None else turnover_smooth
        print(f"鲁棒参数: corr_penalty_lambda={cpl}, turnover_smooth={tsm}")

    # ── 关键：修正前视偏差 ──
    # IC[t] 使用了 T+fwd 日的收益，直到 t+fwd 才可知。
    # 将 IC 序列右移 fwd_shift 个位置，使得在时点 T 仅使用已实现的 IC。
    fwd_shift = FORWARD_RETURN_PERIODS[0]  # 5
    shifted_ic = {
        f: ic_series_dict[f].shift(fwd_shift) for f in use_factors
    }
    # 截止到回测起点前一日，避免使用回测起始日当日收盘后才能观察到的信息
    train_cutoff = pd.Timestamp(BACKTEST_START) - pd.Timedelta(days=1)

    # 1. 因子方向（仅使用回测起点前的 IC，杜绝未来信息）
    directions = determine_factor_directions(
        shifted_ic, cutoff=str(train_cutoff.date())
    )
    dir_info = {f: ("+" if d > 0 else "-") for f, d in directions.items()}
    print(f"因子方向: {dir_info}")

    # 2. 计算滚动权重
    weights = calc_rolling_ic_weights(
        shifted_ic,
        method=method, window=window,
        cutoff=str(train_cutoff.date()),
        corr_penalty_lambda=corr_penalty_lambda,
        turnover_smooth=turnover_smooth,
    )

    # 3. 对齐日期范围
    all_dates = sorted(set.intersection(*[set(factor_matrices[f].index) for f in use_factors]))
    all_dates = [d for d in all_dates if d in weights.index]

    # 获取所有 ETF 列
    all_secs = sorted(set.intersection(*[set(factor_matrices[f].columns) for f in use_factors]))

    # 4. 加权合成
    composite = pd.DataFrame(0.0, index=all_dates, columns=all_secs)

    for factor_name in use_factors:
        mat = factor_matrices[factor_name].loc[all_dates, all_secs]
        direction = directions[factor_name]
        w = weights.loc[all_dates, factor_name].fillna(0.0)

        # composite += direction * w * factor_value
        composite += mat.mul(direction).mul(w, axis=0)

    # 5. 截面排序分数
    composite_rank = composite.rank(axis=1, pct=True)

    print(f"合成因子: {composite_rank.shape[0]} 天 × {composite_rank.shape[1]} 只 ETF")
    print(f"有效日期: {composite_rank.index.min()} ~ {composite_rank.index.max()}")

    return composite_rank


def export_composite_factor(composite: pd.DataFrame,
                              output_path: str = None) -> pd.DataFrame:
    """
    导出合成因子序列为长表 CSV

    Parameters
    ----------
    composite : pd.DataFrame
        合成因子矩阵 (date × sec)
    output_path : str
        输出路径

    Returns
    -------
    pd.DataFrame
        长表格式 (date, sec, composite_score)
    """
    from signalfive.config import OUTPUT_DIR
    output_path = output_path or str(OUTPUT_DIR / "合成因子序列.csv")

    long = composite.stack().reset_index()
    long.columns = ["date", "sec", "composite_score"]
    long = long.sort_values(["date", "sec"]).reset_index(drop=True)
    long.to_csv(output_path, index=False)
    print(f"合成因子已保存: {output_path} ({len(long)} 行)")
    return long


if __name__ == "__main__":
    from data_loader import load_all
    from factors.calc import compute_factors, prepare_factor_matrices
    from factors.testing import test_all_factors, select_effective_factors

    data = load_all()
    panel_wide, macro_df = compute_factors(data["aligned"])
    processed = prepare_factor_matrices(panel_wide, method="rank")
    summary_df, ic_series_dict = test_all_factors(processed, data["close_matrix"])
    effective = select_effective_factors(summary_df)

    composite = combine_factors(processed, ic_series_dict, effective)
    export_composite_factor(composite)
