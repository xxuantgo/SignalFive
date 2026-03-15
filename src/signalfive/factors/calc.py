# -*- coding: utf-8 -*-
"""
因子计算模块
============
职责：
  1. 复用 signalfive.factors.library 的因子计算逻辑
  2. 将因子从长表 (date, sec, factor_value) 转为宽表矩阵 (date × sec)
  3. 对因子做截面标准化（Z-Score），消除量纲差异
  4. 宏观因子单独输出（不参与截面排序，用于 Regime 分析）

关键设计：
  - 因子值使用截面排序百分位（rank_pct），而非原始值。
    原因：28 只 ETF 截面太薄，原始值的分布差异大，rank 更稳健。
  - 标准化在截面（同一天跨 ETF）维度进行，不在时序维度。
"""
import numpy as np
import pandas as pd
from signalfive.factors.library import compute_all_factors, load_product_pool

from signalfive.config import PANEL_FACTORS, MACRO_FACTORS, AUX_FACTORS, DATA_DIR


def compute_factors(aligned: pd.DataFrame) -> tuple:
    """
    计算全部因子，返回 (panel_factors_wide, macro_factors)

    Parameters
    ----------
    aligned : pd.DataFrame
        对齐后的宽表 (date, sec, open, high, low, close, volume, amount, 宏观列...)

    Returns
    -------
    panel_wide : dict[str, pd.DataFrame]
        量价/截面因子, {因子名: DataFrame(index=date, columns=sec)} 的字典
        每个 DataFrame 存储该因子在每天每只 ETF 上的值
    macro_df : pd.DataFrame
        宏观因子, index=date, columns=[F01, F02, ..., F06]
    """
    print("正在计算全部因子...")
    product_pool = load_product_pool(DATA_DIR / "附件1 28只非债券ETF产品池.xlsx")
    factors_panel, factors_macro = compute_all_factors(
        aligned_data=aligned, product_pool=product_pool
    )

    # ---- 截面因子：转为 {因子名: 矩阵} 的字典 ----
    # factors_panel 包含 date, sec 列 + 各因子列
    all_factor_cols = PANEL_FACTORS + AUX_FACTORS
    available_cols = [c for c in all_factor_cols if c in factors_panel.columns]
    missing_cols = [c for c in all_factor_cols if c not in factors_panel.columns]
    if missing_cols:
        print(f"  警告：以下因子未计算出: {missing_cols}")

    panel_wide = {}
    for col in available_cols:
        mat = factors_panel.pivot_table(index="date", columns="sec", values=col)
        mat = mat.sort_index()
        panel_wide[col] = mat

    print(f"  截面因子: {len(panel_wide)} 个, "
          f"日期范围 {factors_panel['date'].min()} ~ {factors_panel['date'].max()}")

    # ---- 宏观因子 ----
    macro_df = factors_macro.copy()
    if not isinstance(macro_df.index, pd.DatetimeIndex):
        if "date" in macro_df.columns:
            macro_df = macro_df.set_index("date")
        macro_df.index = pd.to_datetime(macro_df.index)
    macro_df = macro_df.sort_index()
    available_macro = [c for c in MACRO_FACTORS if c in macro_df.columns]
    macro_df = macro_df[available_macro]
    print(f"  宏观因子: {list(macro_df.columns)}")

    return panel_wide, macro_df


def cross_section_rank(factor_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    截面排序百分位：每天对所有 ETF 排序，输出 [0, 1] 之间的排名百分位。
    好处：消除因子原始值分布差异，在 28 只 ETF 的薄截面上更稳健。

    Parameters
    ----------
    factor_matrix : pd.DataFrame
        index=date, columns=sec, values=因子原始值

    Returns
    -------
    pd.DataFrame
        同形状，值为截面排序百分位 (0=最小, 1=最大)
    """
    return factor_matrix.rank(axis=1, pct=True)


def cross_section_zscore(factor_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    截面 Z-Score 标准化：每天对所有 ETF 做 (x - mean) / std。
    """
    mean = factor_matrix.mean(axis=1)
    std = factor_matrix.std(axis=1).replace(0, 1)
    return factor_matrix.sub(mean, axis=0).div(std, axis=0)


def winsorize_cross_section(factor_matrix: pd.DataFrame,
                             lower_pct: float = 0.025,
                             upper_pct: float = 0.975) -> pd.DataFrame:
    """
    截面去极值 (Winsorize)：每天对因子值裁剪到 [lower_pct, upper_pct] 分位数。
    """
    if factor_matrix.empty or factor_matrix.shape[1] == 0:
        return factor_matrix
    lower = factor_matrix.quantile(lower_pct, axis=1)
    upper = factor_matrix.quantile(upper_pct, axis=1)
    return factor_matrix.clip(lower=lower, upper=upper, axis=0)


def prepare_factor_matrices(panel_wide: dict,
                             method: str = "rank") -> dict:
    """
    对所有截面因子进行统一预处理：
      1. 去极值 (Winsorize)
      2. 标准化 (rank 或 zscore)

    Parameters
    ----------
    panel_wide : dict[str, pd.DataFrame]
        原始因子矩阵字典
    method : str
        标准化方法: "rank"(默认, 推荐) 或 "zscore"

    Returns
    -------
    dict[str, pd.DataFrame]
        预处理后的因子矩阵字典（仅包含 PANEL_FACTORS，不含辅助因子）
    """
    processed = {}
    standardize_fn = cross_section_rank if method == "rank" else cross_section_zscore

    for name, mat in panel_wide.items():
        # 辅助因子不参与合成
        if name in AUX_FACTORS:
            continue
        # 去极值 → 标准化
        cleaned = winsorize_cross_section(mat)
        processed[name] = standardize_fn(cleaned)

    return processed


if __name__ == "__main__":
    from data_loader import load_all
    data = load_all()
    panel_wide, macro_df = compute_factors(data["aligned"])
    processed = prepare_factor_matrices(panel_wide, method="rank")
    print(f"\n预处理后因子数量: {len(processed)}")
    for name, mat in list(processed.items())[:3]:
        print(f"  {name}: shape={mat.shape}, "
              f"non-NaN ratio={mat.notna().sum().sum() / mat.size:.2%}")
