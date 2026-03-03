# -*- coding: utf-8 -*-
"""
严格时间推进 OOS 评估（分段重调参 + 净值拼接）
=================================================

定位与目标：
  - 本脚本用于“严格样本外评估”，不替代 run_main.py 的固定参数基线回测。
  - 核心任务是：对每个外层 OOS 时间段独立调参、独立回测，再拼接成完整 OOS 净值。
  - 默认优化器为 hybrid_cvar_rp，与项目主方案保持一致。

为什么要用本脚本：
  - 解决“先全样本调参再回测”带来的前视偏差风险。
  - 将参数选择严格限制在当时可见历史内，评估更接近真实投研落地流程。

核心原则（防泄漏约束）：
  1) 外层测试段（outer folds）按时间顺序推进；
  2) 每个 outer 段只使用该段开始日前的数据做调参（tune_end = outer_start 前一交易日）；
  3) 选出的参数只用于当前 outer 段，不跨段共享；
  4) 各段 OOS 净值仅做首尾拼接，不回填、不重估历史参数。

模型调参（本脚本重点）：
  - 调参引擎：Optuna + TPE（每个 outer 段独立建 study）。
  - 调参数据：当前 outer 的 inner folds（WFO），必要时回退为单折验证。
  - 搜索参数：top_n / cvar_alpha / cov_window / turnover_lambda / hybrid_beta / cvar_method。
  - 目标函数：以 inner folds 的 Sharpe 为核心，
    同时惩罚波动（std）、差尾部（worst fold）和换手（turnover），强调稳健性。
  - 冷启动机制：可配置前 N 个 outer 段不跑 Bayes，改用锚定参数（anchor_no_bayes）。

端到端流程概览：
  Step 0) 加载数据；
  Step 1) 准备信号（复用缓存或 fresh 重算：合成因子 + 宏观仓位）；
  Step 2) 构建 outer OOS 分段；
  Step 3) 对每个 outer 段执行“内层调参 -> 外层回测”；
  Step 4) 导出 trial/选参/分段绩效；
  Step 5) 拼接各 outer 段净值并输出总绩效摘要与可选图表。

主要输出文件：
  - STRICT_OOS_trials.csv：每个 outer 的所有 trial 明细；
  - STRICT_OOS_selected_params.csv：每个 outer 最终采用参数；
  - STRICT_OOS_segment_performance.csv：每个 outer 段独立绩效；
  - 严格OOS拼接_净值序列.csv / 严格OOS拼接_绩效汇总.csv / 严格OOS拼接_摘要.json。
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from signalfive.backtest.engine import (
    build_optimized_schedule,
    calc_performance,
    export_backtest_plots,
    extract_nav,
    run_backtests,
)
from signalfive.config import (
    BACKTEST_START,
    COMBINE_METHOD,
    CVAR_METHOD,
    DEFAULT_PORTFOLIO_PARAMS,
    DEFAULT_EFFECTIVE_FACTORS,
    MAX_SINGLE_WEIGHT,
    MIN_HOLDINGS,
    OPTIMIZER_METHOD,
    OUTPUT_DIR,
    REGIME_MAX_STEP,
    REGIME_MODE,
    REGIME_RELAX_GAMMA,
    REGIME_STRESS_THRESHOLD,
)
from signalfive.data_loader.loader import load_all
from signalfive.factors.calc import compute_factors, prepare_factor_matrices
from signalfive.factors.combine import combine_factors, export_composite_factor
from signalfive.factors.testing import test_all_factors
from signalfive.portfolio.regime import apply_position_scale, calc_position_scale


def _require_optuna():
    # 依赖检查：严格 OOS 的 Bayes 调参依赖 optuna，缺失时尽早报错。
    try:
        import optuna  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "缺少依赖 optuna，无法运行严格 OOS 调参。请先安装: `pip install optuna`。"
        ) from exc
    return optuna


def _first_trading_on_or_after(index: pd.DatetimeIndex, dt: pd.Timestamp) -> Optional[pd.Timestamp]:
    # 返回 >= dt 的首个交易日；若不存在则返回 None。
    pos = index.searchsorted(dt, side="left")
    if pos >= len(index):
        return None
    return index[pos]


def _last_trading_on_or_before(index: pd.DatetimeIndex, dt: pd.Timestamp) -> Optional[pd.Timestamp]:
    # 返回 <= dt 的最后一个交易日；若不存在则返回 None。
    pos = index.searchsorted(dt, side="right") - 1
    if pos < 0:
        return None
    return index[pos]


def _filter_schedule_by_date(
    schedule: Dict[pd.Timestamp, pd.Series],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Dict[pd.Timestamp, pd.Series]:
    # 裁剪调仓计划，仅保留 [start, end] 区间。
    return {dt: w for dt, w in schedule.items() if start <= dt <= end}


def _run_nav_for_period(
    close_matrix: pd.DataFrame,
    schedule: Dict[pd.Timestamp, pd.Series],
    start: pd.Timestamp,
    end: pd.Timestamp,
    strategy_name: str,
) -> Optional[pd.Series]:
    # 仅在指定时间段运行一次回测，用于 inner/outer 分段评估。
    # 返回 None 表示该段无有效回测结果（无调仓/无价格/回测异常等）。
    period_schedule = _filter_schedule_by_date(schedule, start=start, end=end)
    if not period_schedule:
        return None

    px = close_matrix.loc[(close_matrix.index >= start) & (close_matrix.index <= end)]
    if px.empty:
        return None

    try:
        res = run_backtests(px, {strategy_name: period_schedule})
        nav = extract_nav(res, start_date=str(start.date())).get(strategy_name)
    except Exception:  # noqa: BLE001
        return None

    if nav is None or len(nav) < 3:
        return None

    nav = nav.loc[nav.index <= end] # strict_outer2, Length: 243, dtype: float64(2022-01-04到2023-01-03)
    return nav if len(nav) >= 3 else None


def _average_turnover(schedule: Dict[pd.Timestamp, pd.Series]) -> float:
    # 平均换手率：0.5 * |w_t - w_{t-1}|_1 的均值。
    if not schedule:
        return float("nan")
    prev = None
    vals: List[float] = []
    for _, w in sorted(schedule.items(), key=lambda kv: kv[0]):
        cur = w.sort_index()
        if prev is not None:
            idx = prev.index.union(cur.index)
            turn = 0.5 * (
                cur.reindex(idx, fill_value=0.0) - prev.reindex(idx, fill_value=0.0)
            ).abs().sum()
            vals.append(float(turn))
        prev = cur
    return float(np.mean(vals)) if vals else 0.0


def _safe_float(v: object, default: float = float("nan")) -> float:
    # 安全转浮点：失败或非有限值时返回 default。
    try:
        x = float(v)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return default


def _resolve_period_months(*, years: int, months: int, name: str) -> int:
    # 将 “years/months” 统一折算为“月”：
    # - months>0 时优先使用 months
    # - 否则 years>0 时用 years*12
    # - 否则报参数错误
    y = int(years)
    m = int(months)
    if m > 0:
        return m
    if y > 0:
        return y * 12
    raise ValueError(f"{name} 参数非法，需满足 years>0 或 months>0")


def _build_outer_folds(
    index: pd.DatetimeIndex,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    test_months: int,
    step_months: int,
) -> List[Dict[str, object]]:
    """ 构建外层 OOS 分段：这些段用于最终拼接净值（真实样本外评估）。
    每个分段包含:
    - outer_fold_id
    - test_start/test_end
    - test_days
    """
    folds: List[Dict[str, object]] = []
    k = 0
    while True:
        seg_start_raw = test_start + pd.DateOffset(months=k * step_months)
        if seg_start_raw > test_end:
            break

        seg_start = _first_trading_on_or_after(index, pd.Timestamp(seg_start_raw))
        if seg_start is None or seg_start > test_end:
            break

        seg_end_raw = pd.Timestamp(seg_start_raw) + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
        seg_end_cap = min(pd.Timestamp(seg_end_raw), pd.Timestamp(test_end))
        seg_end = _last_trading_on_or_before(index, seg_end_cap)
        if seg_end is None or seg_end < seg_start:
            k += 1
            continue

        folds.append(
            {
                "outer_fold_id": len(folds) + 1,
                "test_start": pd.Timestamp(seg_start),
                "test_end": pd.Timestamp(seg_end),
                "test_days": int(((index >= seg_start) & (index <= seg_end)).sum()),
            }
        )
        k += 1
    return folds


def _build_wfo_folds(
    index: pd.DatetimeIndex,
    train_start_min: pd.Timestamp,
    first_test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    train_months: int,
    test_months: int,
    step_months: int,
    min_train_days: int,
) -> List[Dict[str, object]]:
    """ 构建内层 WFO folds：每个 outer 段开始前，用历史做调参验证。
    每个 inner fold:
    - 用过去 train_months 做训练窗口
    - 用接下来 test_months 做验证窗口
    - 按 step_months 滚动推进
    """
    folds: List[Dict[str, object]] = []
    k = 0
    while True:
        test_start_raw = first_test_start + pd.DateOffset(months=k * step_months)
        if test_start_raw > test_end:
            break

        test_start = _first_trading_on_or_after(index, pd.Timestamp(test_start_raw))
        if test_start is None or test_start > test_end:
            break

        test_end_raw = pd.Timestamp(test_start_raw) + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
        test_end_cap = min(pd.Timestamp(test_end_raw), pd.Timestamp(test_end))
        test_end_dt = _last_trading_on_or_before(index, test_end_cap)
        if test_end_dt is None or test_end_dt < test_start:
            k += 1
            continue

        train_end_dt = _last_trading_on_or_before(index, pd.Timestamp(test_start) - pd.Timedelta(days=1))
        if train_end_dt is None:
            k += 1
            continue

        train_start_raw = pd.Timestamp(train_end_dt) - pd.DateOffset(months=train_months) + pd.Timedelta(days=1)
        train_start_raw = max(pd.Timestamp(train_start_raw), pd.Timestamp(train_start_min))
        train_start_dt = _first_trading_on_or_after(index, pd.Timestamp(train_start_raw))
        if train_start_dt is None or train_start_dt > train_end_dt:
            k += 1
            continue

        train_days = int(((index >= train_start_dt) & (index <= train_end_dt)).sum())
        test_days = int(((index >= test_start) & (index <= test_end_dt)).sum())
        if train_days >= int(min_train_days) and test_days >= 3:
            folds.append(
                {
                    "inner_fold_id": len(folds) + 1,
                    "train_start": pd.Timestamp(train_start_dt),
                    "train_end": pd.Timestamp(train_end_dt),
                    "test_start": pd.Timestamp(test_start),
                    "test_end": pd.Timestamp(test_end_dt),
                    "train_days": int(train_days),
                    "test_days": int(test_days),
                }
            )
        k += 1
    return folds


def _build_inner_folds_with_fallback(
    index: pd.DatetimeIndex,
    train_start_min: pd.Timestamp,
    tune_end: pd.Timestamp,
    train_months: int,
    test_months: int,
    step_months: int,
    min_train_days: int,
    min_folds: int,
    fallback_valid_days: int,
) -> List[Dict[str, object]]:
    """构建内层调参折（inner folds），并在不足折时回退为单折。

    Parameters
    ----------
    index : pd.DatetimeIndex
        全量交易日索引。
    train_start_min : pd.Timestamp
        内层训练可用的最早交易日。
    tune_end : pd.Timestamp
        当前 outer 段调参截止日（外层测试开始前一天）。
    train_months : int
        内层每折训练窗口（月）。
    test_months : int
        内层每折验证窗口（月）。
    step_months : int
        内层滚动步长（月）。
    min_train_days : int
        每折训练段最少交易日。
    min_folds : int
        期望最少有效 inner 折数。
    fallback_valid_days : int
        当有效折不足时，回退方案的验证段长度（交易日）。

    Returns
    -------
    List[Dict[str, object]]
        inner fold 列表。每个元素包含:
        - inner_fold_id
        - train_start/train_end
        - test_start/test_end
        - train_days/test_days
        - fallback（仅回退方案为 True）

    Notes
    -----
    逻辑优先级：
    1) 先按标准 WFO 构建多折；
    2) 若有效折数 >= min_folds，直接返回；
    3) 否则尝试“历史训练 + 最近验证”的单折回退。
    """
    first_test_start_raw = pd.Timestamp(train_start_min) + pd.DateOffset(months=train_months)
    first_test_start = _first_trading_on_or_after(index, first_test_start_raw)
    folds: List[Dict[str, object]] = []
    if first_test_start is not None and first_test_start <= tune_end:
        folds = _build_wfo_folds(
            index=index,
            train_start_min=train_start_min,
            first_test_start=first_test_start,
            test_end=tune_end,
            train_months=train_months,
            test_months=test_months,
            step_months=step_months,
            min_train_days=min_train_days,
        )
    if len(folds) >= int(min_folds):
        return folds

    # 回退：构造一个“历史训练 + 最近验证”的单折，保证首段也可严格调参
    hist_idx = index[(index >= train_start_min) & (index <= tune_end)]
    if len(hist_idx) < int(min_train_days) + 20:
        return folds

    valid_days = min(max(60, int(fallback_valid_days)), len(hist_idx) - int(min_train_days))
    if valid_days <= 0:
        return folds

    valid_start = hist_idx[-valid_days]
    train_end = _last_trading_on_or_before(index, pd.Timestamp(valid_start) - pd.Timedelta(days=1))
    train_start = _first_trading_on_or_after(index, train_start_min)
    if train_end is None or train_start is None or train_start > train_end:
        return folds

    train_days = int(((index >= train_start) & (index <= train_end)).sum())
    test_days = int(((index >= valid_start) & (index <= tune_end)).sum())
    if train_days < int(min_train_days) or test_days < 3:
        return folds

    return [
        {
            "inner_fold_id": 1,
            "train_start": pd.Timestamp(train_start),
            "train_end": pd.Timestamp(train_end),
            "test_start": pd.Timestamp(valid_start),
            "test_end": pd.Timestamp(tune_end),
            "train_days": int(train_days),
            "test_days": int(test_days),
            "fallback": True,
        }
    ]


def _load_cached_signals(run_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """从历史运行目录加载缓存信号。

    Parameters
    ----------
    run_dir : Path
        需要复用的 `run_main`（或同结构运行）输出目录。

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        composite : pd.DataFrame
            合成因子矩阵，形状为 date x sec。
        position_scale : pd.Series
            宏观仓位系数序列，index 为 date。

    Raises
    ------
    FileNotFoundError
        缓存文件缺失时抛出。
    ValueError
        仓位文件缺少 `position_scale` 列时抛出。
    """
    comp_path = run_dir / "合成因子序列.csv"
    ps_path = run_dir / "宏观仓位系数.csv"
    if not comp_path.exists():
        raise FileNotFoundError(f"未找到缓存合成因子文件: {comp_path}")
    if not ps_path.exists():
        raise FileNotFoundError(f"未找到缓存仓位系数文件: {ps_path}")

    comp_long = pd.read_csv(comp_path, parse_dates=["date"])
    composite = comp_long.pivot(index="date", columns="sec", values="composite_score").sort_index()

    ps_df = pd.read_csv(ps_path, parse_dates=["date"])
    if "position_scale" not in ps_df.columns:
        raise ValueError(f"{ps_path} 缺少列 `position_scale`")
    position_scale = ps_df.set_index("date")["position_scale"].sort_index()
    return composite, position_scale


def _build_signals_fresh(run_dir: Path, close_matrix: pd.DataFrame, aligned: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """现场重算信号（不复用缓存）。

    Parameters
    ----------
    run_dir : Path
        当前 strict_oos 运行输出目录（用于落地中间信号文件）。
    close_matrix : pd.DataFrame
        收盘价矩阵（date x sec），用于 IC 测试。
    aligned : pd.DataFrame
        对齐后的原始输入数据（量价+宏观）。

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        composite : pd.DataFrame
            合成因子矩阵（date x sec）。
        position_scale : pd.Series
            宏观仓位系数（date -> scale）。

    Notes
    -----
    流程：
    1) 计算并预处理因子；
    2) 计算 IC 序列；
    3) 使用固定有效因子列表与固定合成方法生成 composite；
    4) 计算 Regime 仓位系数；
    5) 落盘保存 `合成因子序列.csv` 与 `宏观仓位系数.csv`。
    """
    panel_wide, macro_df = compute_factors(aligned)
    processed = prepare_factor_matrices(panel_wide, method="rank")
    _, ic_series_dict = test_all_factors(processed, close_matrix)

    effective = list(DEFAULT_EFFECTIVE_FACTORS)
    missing = [f for f in effective if f not in processed]
    if missing:
        raise ValueError(f"固定有效因子中存在未计算因子: {missing}")

    composite = combine_factors(
        processed,
        ic_series_dict,
        effective,
        method=COMBINE_METHOD,
    )
    export_composite_factor(composite, output_path=str(run_dir / "合成因子序列.csv"))

    if REGIME_MODE == "off":
        position_scale = pd.Series(1.0, index=macro_df.index, name="position_scale")
    else:
        stress_threshold_use = (
            float(REGIME_STRESS_THRESHOLD) if REGIME_STRESS_THRESHOLD is not None else None
        )
        max_step_use = float(REGIME_MAX_STEP) if float(REGIME_MAX_STEP) > 0 else None
        position_scale = calc_position_scale(
            macro_df,
            smooth_window=5,
            relax_gamma=float(REGIME_RELAX_GAMMA),
            stress_threshold=stress_threshold_use,
            max_daily_step=max_step_use,
            stress_factors=("F01", "F02", "F04"),
        )

    ps_df = position_scale.to_frame("position_scale")
    ps_df.index.name = "date"
    ps_df.to_csv(run_dir / "宏观仓位系数.csv")
    return composite.sort_index(), position_scale.sort_index()


def _build_parser() -> argparse.ArgumentParser:
    """定义命令行参数。

    参数分层：
    - 时间分段参数：outer/inner 的窗口和步长
    - 搜索空间参数：top_n/alpha/window/lambda/beta/cvar_method
    - 调参过程参数：n_trials/seed/timeout
    - 目标函数惩罚项：std/worst/turnover
    """
    parser = argparse.ArgumentParser(description="严格时间推进 OOS：分段重调参 + 净值拼接")
    parser.add_argument("--reuse-run-dir", type=str, default="", help="可选：复用 run_main 输出目录的信号缓存")

    parser.add_argument("--train-start-min", type=str, default="2019-11-01", help="最早可用训练起点")
    parser.add_argument("--outer-test-start", type=str, default=BACKTEST_START, help="外层OOS起点")
    parser.add_argument("--outer-test-end", type=str, default="2025-10-30", help="外层OOS终点")
    parser.add_argument("--outer-test-years", type=int, default=1, help="外层每段测试长度（年）")
    parser.add_argument("--outer-step-years", type=int, default=1, help="外层滚动步长（年）")
    parser.add_argument("--outer-test-months", type=int, default=0, help="外层每段测试长度（月，>0 时优先于 years）")
    parser.add_argument("--outer-step-months", type=int, default=0, help="外层滚动步长（月，>0 时优先于 years）")

    parser.add_argument("--inner-train-years", type=int, default=2, help="内层调参训练窗口（年）")
    parser.add_argument("--inner-test-years", type=int, default=1, help="内层调参验证窗口（年）")
    parser.add_argument("--inner-step-years", type=int, default=1, help="内层调参步长（年）")
    parser.add_argument("--inner-train-months", type=int, default=0, help="内层调参训练窗口（月，>0 时优先于 years）")
    parser.add_argument("--inner-test-months", type=int, default=0, help="内层调参验证窗口（月，>0 时优先于 years）")
    parser.add_argument("--inner-step-months", type=int, default=0, help="内层调参步长（月，>0 时优先于 years）")
    parser.add_argument("--inner-min-folds", type=int, default=2, help="内层最少有效验证折数")
    parser.add_argument("--inner-fallback-valid-days", type=int, default=120, help="内层不足折时的回退验证天数")
    parser.add_argument("--min-train-days", type=int, default=220, help="训练最少交易日")

    parser.add_argument("--top-n-low", type=int, default=3)
    parser.add_argument("--top-n-high", type=int, default=9)
    parser.add_argument("--top-n-step", type=int, default=1)
    parser.add_argument(
        "--no-bayes-first-n-outer",
        type=int,
        default=1,
        help="前 N 个 outer fold 禁用 Bayes（0 表示所有 outer fold 均启用 Bayes）",
    )
    parser.add_argument(
        "--first-fold-anchor-top-n",
        type=int,
        default=int(DEFAULT_PORTFOLIO_PARAMS["top_n"]),
        help="锚定 top_n（用于 no-bayes 的 outer 折）",
    )
    parser.add_argument(
        "--first-fold-anchor-alpha",
        type=float,
        default=float(DEFAULT_PORTFOLIO_PARAMS["cvar_alpha"]),
        help="锚定 cvar_alpha（用于 no-bayes 的 outer 折）",
    )
    parser.add_argument(
        "--first-fold-anchor-method",
        type=str,
        default=str(DEFAULT_PORTFOLIO_PARAMS["cvar_method"]),
        help="锚定 cvar_method（用于 no-bayes 的 outer 折）",
    )
    parser.add_argument(
        "--first-fold-anchor-window",
        type=int,
        default=int(DEFAULT_PORTFOLIO_PARAMS["cov_window"]),
        help="锚定 cov_window（用于 no-bayes 的 outer 折）",
    )
    parser.add_argument(
        "--first-fold-anchor-lambda",
        type=float,
        default=float(DEFAULT_PORTFOLIO_PARAMS["turnover_lambda"]),
        help="锚定 turnover_lambda（用于 no-bayes 的 outer 折）",
    )
    parser.add_argument(
        "--first-fold-anchor-beta",
        type=float,
        default=float(DEFAULT_PORTFOLIO_PARAMS["hybrid_beta"]),
        help="锚定 hybrid_beta（用于 no-bayes 的 outer 折）",
    )
    parser.add_argument("--alpha-low", type=float, default=0.92)
    parser.add_argument("--alpha-high", type=float, default=0.96)
    parser.add_argument("--window-low", type=int, default=60)
    parser.add_argument("--window-high", type=int, default=220)
    parser.add_argument("--window-step", type=int, default=5)
    parser.add_argument("--lambda-low", type=float, default=1e-3)
    parser.add_argument("--lambda-high", type=float, default=3e-2)
    parser.add_argument("--beta-low", type=float, default=0.10)
    parser.add_argument("--beta-high", type=float, default=0.50)
    parser.add_argument("--beta-step", type=float, default=0.05)
    parser.add_argument(
        "--cvar-methods",
        type=str,
        default="empirical,parametric,cornish_fisher",
        help="empirical,parametric,cornish_fisher",
    )

    parser.add_argument("--n-trials", type=int, default=60, help="每个外层段的 Bayes trial 数")
    parser.add_argument("--n-startup-trials", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=0)

    parser.add_argument("--obj-std-penalty", type=float, default=1.0)
    parser.add_argument("--obj-worst-penalty", type=float, default=1.0)
    parser.add_argument("--obj-sharpe-floor", type=float, default=0.0)
    parser.add_argument("--obj-turnover-penalty", type=float, default=0.2)

    parser.add_argument("--export-backtest-plots", action="store_true", help="导出拼接净值/回撤图")
    return parser


def main() -> None:
    """严格 OOS 主流程入口。

    总体流程：
    1) 校验参数并标准化
    2) 加载或重算信号
    3) 构建 outer 分段
    4) 对每个 outer 段：仅用历史做 inner 调参，再跑本段 OOS
    5) 拼接各段净值并导出汇总
    """
    args = _build_parser().parse_args()
    optuna = _require_optuna()

    # ---------------------------------------------------------------------
    # A. 参数合法性校验与标准化
    # ---------------------------------------------------------------------
    # 目的：尽早发现非法配置（fail fast），避免进入耗时回测后才报错。
    if args.top_n_low < MIN_HOLDINGS:
        raise ValueError(f"top_n_low 不能小于 MIN_HOLDINGS={MIN_HOLDINGS}")
    if args.top_n_high < args.top_n_low or args.top_n_step <= 0:
        raise ValueError("TopN 搜索区间非法")
    if args.first_fold_anchor_top_n < MIN_HOLDINGS:
        raise ValueError(f"first-fold-anchor-top-n 不能小于 MIN_HOLDINGS={MIN_HOLDINGS}")
    if not (0.0 < args.alpha_low < args.alpha_high < 1.0):
        raise ValueError("alpha 搜索区间必须满足 0<low<high<1")
    if not (0.0 < args.first_fold_anchor_alpha < 1.0):
        raise ValueError("first-fold-anchor-alpha 必须在 (0,1) 内")
    if args.window_low <= 0 or args.window_high < args.window_low or args.window_step <= 0:
        raise ValueError("window 搜索区间非法")
    if args.first_fold_anchor_window <= 0:
        raise ValueError("first-fold-anchor-window 必须 > 0")
    if args.lambda_low <= 0 or args.lambda_high <= args.lambda_low:
        raise ValueError("lambda 搜索区间非法")
    if args.first_fold_anchor_lambda < 0:
        raise ValueError("first-fold-anchor-lambda 不能为负")
    if not (0.0 <= args.beta_low <= args.beta_high <= 1.0) or args.beta_step <= 0:
        raise ValueError("beta 搜索区间非法")
    if not (0.0 <= args.first_fold_anchor_beta <= 1.0):
        raise ValueError("first-fold-anchor-beta 必须在 [0,1] 内")
    if args.n_trials <= 0 or args.n_startup_trials < 1:
        raise ValueError("n_trials / n_startup_trials 非法")
    if args.min_train_days <= 0:
        raise ValueError("min_train_days 必须 > 0")
    if int(args.no_bayes_first_n_outer) < 0:
        raise ValueError("no-bayes-first-n-outer 必须 >= 0")
    no_bayes_first_n_outer = int(args.no_bayes_first_n_outer) # 前 N 个 outer fold 禁用 Bayes
    # 将所有年/月窗口统一成“月”，后续分段推进统一用 DateOffset(months=...)。
    outer_test_months = _resolve_period_months( # 12
        years=int(args.outer_test_years),
        months=int(args.outer_test_months),
        name="outer-test",
    )
    outer_step_months = _resolve_period_months( # 12
        years=int(args.outer_step_years),
        months=int(args.outer_step_months),
        name="outer-step",
    )
    inner_train_months = _resolve_period_months( # 24
        years=int(args.inner_train_years),
        months=int(args.inner_train_months),
        name="inner-train",
    )
    inner_test_months = _resolve_period_months(
        years=int(args.inner_test_years),
        months=int(args.inner_test_months),
        name="inner-test",
    )
    inner_step_months = _resolve_period_months(
        years=int(args.inner_step_years),
        months=int(args.inner_step_months),
        name="inner-step",
    )

    # 将 cvar_method 输入做别名归一化，保证后续分支判断稳定。
    cvar_methods = [m.strip().lower() for m in str(args.cvar_methods).split(",") if m.strip()]
    alias = {
        "lp": "empirical",
        "historical": "empirical",
        "gaussian": "parametric",
        "cornish-fisher": "cornish_fisher",
        "cf": "cornish_fisher",
    }
    cvar_methods = [alias.get(m, m) for m in cvar_methods]
    valid_methods = {"empirical", "parametric", "cornish_fisher"}
    invalid_methods = sorted(set(cvar_methods) - valid_methods)
    if invalid_methods:
        raise ValueError(f"非法 cvar-methods: {invalid_methods}，可选: {sorted(valid_methods)}")
    if not cvar_methods:
        cvar_methods = [CVAR_METHOD]
    cvar_methods = sorted(set(cvar_methods))

    # 首段锚定参数（no-bayes 模式）也统一做方法名规范化。
    first_fold_anchor_method = alias.get(
        str(args.first_fold_anchor_method).strip().lower(),
        str(args.first_fold_anchor_method).strip().lower(),
    )
    if first_fold_anchor_method not in valid_methods:
        raise ValueError(
            f"非法 first-fold-anchor-method: {first_fold_anchor_method}，可选: {sorted(valid_methods)}"
        )
    # 首段锚定参数：
    # 当 outer_id <= no_bayes_first_n_outer 时，不跑 Bayes 搜索，直接使用这组参数。
    # 设计目的：在早期历史样本相对不足时，减少搜索不稳定性。
    first_fold_anchor_params = {
        "top_n": int(args.first_fold_anchor_top_n),
        "cvar_alpha": float(args.first_fold_anchor_alpha),
        "cvar_method": str(first_fold_anchor_method),
        "cov_window": int(args.first_fold_anchor_window),
        "turnover_lambda": float(args.first_fold_anchor_lambda),
        "hybrid_beta": float(args.first_fold_anchor_beta),
        "max_weight": float(MAX_SINGLE_WEIGHT),
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(OUTPUT_DIR) / f"strict_oos_stitch_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {run_dir}")

    # ---------------------------------------------------------------------
    # B. Step 0 - 加载基础数据
    # ---------------------------------------------------------------------
    print("=" * 70)
    print("Step 0: 加载数据")
    data = load_all()
    close_matrix = data["close_matrix"].sort_index()

    # ---------------------------------------------------------------------
    # C. Step 1 - 准备信号（合成因子 + 宏观仓位系数）
    # ---------------------------------------------------------------------
    reuse_run_dir = Path(args.reuse_run_dir).expanduser() if args.reuse_run_dir else None
    if reuse_run_dir is not None:
        print("=" * 70)
        print(f"Step 1: 复用信号缓存: {reuse_run_dir}")
        composite, position_scale = _load_cached_signals(reuse_run_dir)
    else:
        print("=" * 70)
        print("Step 1: fresh 重新计算信号（固定因子+固定合成）")
        composite, position_scale = _build_signals_fresh(
            run_dir=run_dir,
            close_matrix=close_matrix,
            aligned=data["aligned"],
        )

    composite = composite.sort_index() # 合成因子矩阵           
    position_scale = position_scale.sort_index() #  宏观仓位系数序列，index 为 date。

    # ---------------------------------------------------------------------
    # D. 统一时间边界到交易日，并构建外层 OOS 分段
    # ---------------------------------------------------------------------
    # 用户输入是自然日；回测需要映射到真实交易日索引。
    train_start_min = _first_trading_on_or_after(close_matrix.index, pd.Timestamp(args.train_start_min))
    test_start = _first_trading_on_or_after(close_matrix.index, pd.Timestamp(args.outer_test_start))
    test_end_raw = pd.Timestamp(args.outer_test_end) if args.outer_test_end else pd.Timestamp(close_matrix.index.max())
    test_end = _last_trading_on_or_before(close_matrix.index, test_end_raw)
    if train_start_min is None or test_start is None or test_end is None:
        raise ValueError("train/test 日期超出交易日范围")
    if not (train_start_min < test_start <= test_end):
        raise ValueError("日期关系非法，需满足 train_start_min < outer_test_start <= outer_test_end")

    # 外层 folds = 最终样本外评估段，后续每段独立调参、独立回测、最终拼接。
    outer_folds = _build_outer_folds(
        index=close_matrix.index,
        test_start=pd.Timestamp(test_start),
        test_end=pd.Timestamp(test_end),
        test_months=int(outer_test_months),
        step_months=int(outer_step_months),
    )
    if not outer_folds:
        raise ValueError("未构造出任何外层 OOS 分段")

    pd.DataFrame(
        [
            {
                "outer_fold_id": int(f["outer_fold_id"]),
                "test_start": str(pd.Timestamp(f["test_start"]).date()),
                "test_end": str(pd.Timestamp(f["test_end"]).date()),
                "test_days": int(f["test_days"]),
            }
            for f in outer_folds
        ]
    ).to_csv(run_dir / "STRICT_OOS_outer_folds.csv", index=False)

    print("=" * 70)
    print(
        f"Step 2: 外层分段完成，segments={len(outer_folds)}, "
        f"range=[{pd.Timestamp(test_start).date()}~{pd.Timestamp(test_end).date()}], "
        f"outer={outer_test_months}m/{outer_step_months}m, "
        f"inner={inner_train_months}m/{inner_test_months}m/{inner_step_months}m, "
        f"no_bayes_first_n_outer={no_bayes_first_n_outer}"
    )

    all_trials: List[Dict[str, object]] = []
    selected_rows: List[Dict[str, object]] = []
    segment_perf_rows: List[Dict[str, object]] = []
    segment_navs: List[pd.Series] = []

    # ---------------------------------------------------------------------
    # E. Step 3 - 按 outer fold 逐段执行（严格时间推进）
    # ---------------------------------------------------------------------
    for outer in outer_folds:
        outer_id = int(outer["outer_fold_id"])
        outer_start = pd.Timestamp(outer["test_start"])
        outer_end = pd.Timestamp(outer["test_end"])
        # tune_end: 本段测试开始前最后一个交易日。
        # 约束：该段调参只能看到 tune_end 及以前数据，防前视。
        tune_end = _last_trading_on_or_before(close_matrix.index, outer_start - pd.Timedelta(days=1))
        # 核心防前视：本段 outer 的调参数据只能截至 tune_end（外层开始前一天）。
        if tune_end is None:
            raise ValueError(f"outer_fold#{outer_id} 在测试起点前找不到可用调参数据")

        # 是否对当前 outer 启用锚定参数而非 Bayes 搜索。
        use_anchor_no_bayes = outer_id <= int(no_bayes_first_n_outer)
        # 针对当前 outer 构建内层验证折（用于该段参数选择）。
        inner_folds = _build_inner_folds_with_fallback(
            index=close_matrix.index,
            train_start_min=pd.Timestamp(train_start_min),
            tune_end=pd.Timestamp(tune_end),
            train_months=int(inner_train_months),
            test_months=int(inner_test_months),
            step_months=int(inner_step_months), # 12s
            min_train_days=int(args.min_train_days),
            min_folds=int(args.inner_min_folds),
            fallback_valid_days=int(args.inner_fallback_valid_days),
        )
        if not inner_folds and not use_anchor_no_bayes:
            raise ValueError(f"outer_fold#{outer_id} 内层验证折为空，无法调参")
        if not inner_folds and use_anchor_no_bayes:
            print(f"  警告: outer_fold#{outer_id} 内层验证折为空，锚定模式继续执行。")

        inner_fold_records = [
            {
                "outer_fold_id": outer_id,
                "inner_fold_id": int(f["inner_fold_id"]),
                "train_start": str(pd.Timestamp(f["train_start"]).date()),
                "train_end": str(pd.Timestamp(f["train_end"]).date()),
                "test_start": str(pd.Timestamp(f["test_start"]).date()),
                "test_end": str(pd.Timestamp(f["test_end"]).date()),
                "train_days": int(f["train_days"]),
                "test_days": int(f["test_days"]),
                "fallback": bool(f.get("fallback", False)),
            }
            for f in inner_folds
        ]
        pd.DataFrame(
            inner_fold_records,
            columns=[
                "outer_fold_id",
                "inner_fold_id",
                "train_start",
                "train_end",
                "test_start",
                "test_end",
                "train_days",
                "test_days",
                "fallback",
            ],
        ).to_csv(run_dir / f"STRICT_OOS_inner_folds_outer{outer_id}.csv", index=False)

        print("=" * 70)
        mode_label = (
            f"前{no_bayes_first_n_outer}折锚定(不跑Bayes)" if use_anchor_no_bayes else "Bayes调参"
        )
        print(
            f"Step 3.{outer_id}: 调参与评估 outer_fold#{outer_id} "
            f"[{outer_start.date()}~{outer_end.date()}], tune_end={pd.Timestamp(tune_end).date()}, "
            f"inner_folds={len(inner_folds)}, mode={mode_label}"
        )

        if use_anchor_no_bayes:
            # 锚定模式：跳过 Bayes调参，直接使用固定参数。
            best_params = dict(first_fold_anchor_params)
            selected_rows.append(
                {
                    "outer_fold_id": outer_id,
                    "outer_test_start": str(outer_start.date()),
                    "outer_test_end": str(outer_end.date()),
                    "tune_end": str(pd.Timestamp(tune_end).date()),
                    "inner_folds": int(len(inner_folds)),
                    "selection_mode": "anchor_no_bayes",
                    "best_objective": np.nan,
                    "best_inner_valid_folds": 0,
                    **best_params,
                }
            )
        else:
            # Bayes 模式：每个 outer 独立建 study，互不泄漏未来信息。
            sampler = optuna.samplers.TPESampler(
                seed=int(args.seed) + outer_id * 1000,
                n_startup_trials=max(1, int(args.n_startup_trials)),
                multivariate=True,
            )
            study = optuna.create_study(direction="maximize", sampler=sampler)

            def evaluate_params(
                *,
                top_n: int,
                cvar_alpha: float,
                cov_window: int,
                turnover_lambda: float,
                hybrid_beta: float,
                cvar_method: str,
            ) -> Dict[str, object]:
                """在当前 outer 的 inner folds 上评估一组参数。

                返回字段：
                - objective: 目标函数值（越大越好）
                - valid_folds: 有效验证折数
                - mean/std/min_sharpe: 稳健性统计
                - avg_turnover: 平均换手
                """
                fold_sharpes: List[float] = []
                fold_turnovers: List[float] = []

                # 遍历每个 inner 验证段，汇总性能与换手。
                for inner in inner_folds:
                    fold_start = pd.Timestamp(inner["test_start"])
                    fold_end = pd.Timestamp(inner["test_end"])

                    raw_all = build_optimized_schedule(
                        # 调仓计划只构建到 fold_end；rebal_start 从 fold_start 开始。
                        composite.loc[:fold_end],
                        close_matrix.loc[:fold_end],
                        optimizer=OPTIMIZER_METHOD,
                        top_n=int(top_n),
                        max_weight=MAX_SINGLE_WEIGHT,
                        min_holdings=MIN_HOLDINGS,
                        cov_window=int(cov_window),
                        cvar_alpha=float(cvar_alpha),
                        cvar_method=str(cvar_method),
                        turnover_lambda=float(turnover_lambda),
                        hybrid_beta=float(hybrid_beta),
                        rebal_start=fold_start,
                    )
                    raw_fold = _filter_schedule_by_date(raw_all, start=fold_start, end=fold_end)
                    if not raw_fold:
                        continue

                    adj_fold = apply_position_scale(raw_fold, position_scale)
                    nav = _run_nav_for_period( # 回测, 并且得到归一化nav
                        close_matrix=close_matrix,
                        schedule=adj_fold,
                        start=fold_start,
                        end=fold_end,
                        strategy_name=f"outer{outer_id}_inner{int(inner['inner_fold_id'])}",
                    )
                    if nav is None:
                        continue

                    # 记录该 inner 段风险收益指标。
                    perf = calc_performance(nav)
                    sharpe = _safe_float(perf.get("sharpe"))
                    if np.isfinite(sharpe):
                        fold_sharpes.append(float(sharpe))
                    turn = _average_turnover(adj_fold)
                    if np.isfinite(turn):
                        fold_turnovers.append(float(turn))

                valid_folds = len(fold_sharpes)
                if valid_folds == 0:
                    # 没有任何有效内层结果，返回极差目标值以淘汰该参数。
                    return {
                        "objective": -1e6,
                        "valid_folds": 0,
                        "mean_sharpe": np.nan,
                        "std_sharpe": np.nan,
                        "min_sharpe": np.nan,
                        "avg_turnover": np.nan,
                    }

                sharpe_arr = np.array(fold_sharpes, dtype=float)
                mean_sharpe = float(np.mean(sharpe_arr))
                std_sharpe = float(np.std(sharpe_arr, ddof=1)) if valid_folds > 1 else 0.0
                min_sharpe = float(np.min(sharpe_arr))
                avg_turnover = float(np.mean(fold_turnovers)) if fold_turnovers else 0.0

                objective = mean_sharpe
                # 稳健目标：奖励平均 Sharpe，惩罚不稳定/差尾部/高换手。
                objective -= float(args.obj_std_penalty) * std_sharpe
                objective -= float(args.obj_worst_penalty) * max(0.0, float(args.obj_sharpe_floor) - min_sharpe)
                objective -= float(args.obj_turnover_penalty) * avg_turnover
                # 若内层有效折不足，则加惩罚
                if valid_folds < int(args.inner_min_folds):
                    objective -= 0.5 * float(int(args.inner_min_folds) - valid_folds)

                return {
                    "objective": float(objective),
                    "valid_folds": int(valid_folds),
                    "mean_sharpe": mean_sharpe,
                    "std_sharpe": std_sharpe,
                    "min_sharpe": min_sharpe,
                    "avg_turnover": avg_turnover,
                }

            def objective(trial) -> float:
                """Optuna 单次 trial。

                步骤：
                1) 从预定义搜索空间采样参数；
                2) 调用 evaluate_params 在 inner folds 上打分；
                3) 将该 trial 结果写入 all_trials 供后续导出与追溯。
                """
                top_n = trial.suggest_int("top_n", int(args.top_n_low), int(args.top_n_high), step=int(args.top_n_step))
                cvar_alpha = trial.suggest_float("cvar_alpha", float(args.alpha_low), float(args.alpha_high))
                cov_window = trial.suggest_int(
                    "cov_window",
                    int(args.window_low),
                    int(args.window_high),
                    step=max(1, int(args.window_step)),
                )
                turnover_lambda = trial.suggest_float(
                    "turnover_lambda",
                    float(args.lambda_low),
                    float(args.lambda_high),
                    log=True,
                )
                hybrid_beta = trial.suggest_float(
                    "hybrid_beta",
                    float(args.beta_low),
                    float(args.beta_high),
                    step=float(args.beta_step),
                )
                if len(cvar_methods) == 1:
                    cvar_method = cvar_methods[0]
                else:
                    cvar_method = trial.suggest_categorical("cvar_method", cvar_methods)

                eval_rec = evaluate_params(
                    top_n=int(top_n),
                    cvar_alpha=float(cvar_alpha),
                    cov_window=int(cov_window),
                    turnover_lambda=float(turnover_lambda),
                    hybrid_beta=float(hybrid_beta),
                    cvar_method=str(cvar_method),
                )
                all_trials.append(
                    {
                        "outer_fold_id": outer_id,
                        "trial_number": int(trial.number),
                        "top_n": int(top_n),
                        "cvar_alpha": float(cvar_alpha),
                        "cvar_method": str(cvar_method),
                        "cov_window": int(cov_window),
                        "turnover_lambda": float(turnover_lambda),
                        "hybrid_beta": float(hybrid_beta),
                        "objective": float(eval_rec["objective"]),
                        "valid_folds": int(eval_rec["valid_folds"]),
                        "mean_sharpe": _safe_float(eval_rec["mean_sharpe"]),
                        "std_sharpe": _safe_float(eval_rec["std_sharpe"]),
                        "min_sharpe": _safe_float(eval_rec["min_sharpe"]),
                        "avg_turnover": _safe_float(eval_rec["avg_turnover"]),
                    }
                )
                return float(eval_rec["objective"])

            study.optimize(
                objective,
                n_trials=int(args.n_trials),
                timeout=(None if int(args.timeout) <= 0 else int(args.timeout)),
                n_jobs=1,
                show_progress_bar=False,
            )

            outer_trial_df = pd.DataFrame([r for r in all_trials if int(r["outer_fold_id"]) == outer_id])
            if outer_trial_df.empty:
                raise ValueError(f"outer_fold#{outer_id} 未产出任何 trial")

            # 当前 outer 内择优：objective 高优先；若并列取 trial_number 更小者。
            best_trial = outer_trial_df.sort_values(["objective", "trial_number"], ascending=[False, True]).iloc[0]
            best_params = {
                "top_n": int(best_trial["top_n"]),
                "cvar_alpha": float(best_trial["cvar_alpha"]),
                "cvar_method": str(best_trial["cvar_method"]),
                "cov_window": int(best_trial["cov_window"]),
                "turnover_lambda": float(best_trial["turnover_lambda"]),
                "hybrid_beta": float(best_trial["hybrid_beta"]),
                "max_weight": MAX_SINGLE_WEIGHT,
            }

            selected_rows.append( # 每一个outer折对应参数
                {
                    "outer_fold_id": outer_id,
                    "outer_test_start": str(outer_start.date()),
                    "outer_test_end": str(outer_end.date()),
                    "tune_end": str(pd.Timestamp(tune_end).date()),
                    "inner_folds": int(len(inner_folds)),
                    "selection_mode": "bayes",
                    "best_objective": float(best_trial["objective"]),
                    "best_inner_valid_folds": int(best_trial["valid_folds"]),
                    **best_params,
                }
            )

        # 用“本段独立参数”仅运行“本段 OOS”，避免跨段信息泄漏。
        raw_all = build_optimized_schedule( # 这一段时间的调仓计划
            composite.loc[:outer_end], # 从2019-12-25到Timestamp('2021-12-31 00:00:00')
            close_matrix.loc[:outer_end], # 从2019-11-01到Timestamp('2021-12-31 00:00:00')
            optimizer=OPTIMIZER_METHOD,
            top_n=best_params["top_n"], # 固定的锚点设置是3
            max_weight=MAX_SINGLE_WEIGHT,
            min_holdings=MIN_HOLDINGS,
            cov_window=best_params["cov_window"],
            cvar_alpha=best_params["cvar_alpha"],
            cvar_method=best_params["cvar_method"],
            turnover_lambda=best_params["turnover_lambda"],
            hybrid_beta=best_params["hybrid_beta"],
            rebal_start=outer_start,
        )
        raw_fold = _filter_schedule_by_date(raw_all, start=outer_start, end=outer_end)
        if not raw_fold:
            raise ValueError(f"outer_fold#{outer_id} 调仓计划为空")

        adj_fold = apply_position_scale(raw_fold, position_scale)
        nav_fold = _run_nav_for_period( # 回测, 并且通过函数得到归一化净值
            close_matrix=close_matrix,
            schedule=adj_fold,
            start=outer_start,
            end=outer_end,
            strategy_name=f"strict_outer{outer_id}",
        )
        if nav_fold is None:
            raise ValueError(f"outer_fold#{outer_id} OOS 净值为空")

        # 保存当前 outer 段独立绩效，供后续分段分析。
        perf_fold = calc_performance(nav_fold)
        segment_perf_rows.append(
            {
                "outer_fold_id": outer_id,
                "test_start": str(outer_start.date()),
                "test_end": str(outer_end.date()),
                "trading_days": int(len(nav_fold)),
                "annual_return": float(perf_fold["annual_return"]),
                "annual_vol": float(perf_fold["annual_vol"]),
                "sharpe": float(perf_fold["sharpe"]),
                "max_drawdown": float(perf_fold["max_drawdown"]),
                "calmar": float(perf_fold["calmar"]),
                "total_return": float(perf_fold["total_return"]),
            }
        )
        segment_navs.append(nav_fold)

    # ---------------------------------------------------------------------
    # F. 导出 trial / 分段参数 / 分段绩效
    # ---------------------------------------------------------------------
    if all_trials:
        trials_df = pd.DataFrame(all_trials).sort_values(
            ["outer_fold_id", "objective", "trial_number"], ascending=[True, False, True]
        )
    else:
        trials_df = pd.DataFrame(
            columns=[
                "outer_fold_id",
                "trial_number",
                "top_n",
                "cvar_alpha",
                "cvar_method",
                "cov_window",
                "turnover_lambda",
                "hybrid_beta",
                "objective",
                "valid_folds",
                "mean_sharpe",
                "std_sharpe",
                "min_sharpe",
                "avg_turnover",
            ]
        )
    trials_df.to_csv(run_dir / "STRICT_OOS_trials.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(run_dir / "STRICT_OOS_selected_params.csv", index=False)
    pd.DataFrame(segment_perf_rows).to_csv(run_dir / "STRICT_OOS_segment_performance.csv", index=False)

    # ---------------------------------------------------------------------
    # G. 拼接各段 OOS 净值
    # ---------------------------------------------------------------------
    # 规则：
    # - 每段先归一到 1；
    # - 后一段乘以前一段末值；
    # - 去掉时间重叠点，得到连续净值轨迹。
    stitched: Optional[pd.Series] = None
    for nav in segment_navs:
        seg = nav.sort_index()
        seg = seg / seg.iloc[0]
        if stitched is None:
            stitched = seg.copy()
            continue
        base = float(stitched.iloc[-1])
        # 将下一段净值首点对齐到上一段末值，实现首尾连续拼接。
        seg = seg * base
        seg = seg.loc[seg.index > stitched.index[-1]]
        if seg.empty:
            continue
        stitched = pd.concat([stitched, seg]).sort_index()

    if stitched is None or stitched.empty:
        raise ValueError("严格 OOS 拼接净值为空")

    stitched.name = "strict_oos_stitched_nav"
    stitched_df = stitched.to_frame("nav")
    stitched_df.index.name = "date"
    stitched_path = run_dir / "严格OOS拼接_净值序列.csv"
    stitched_df.to_csv(stitched_path)

    # ---------------------------------------------------------------------
    # H. 拼接后总绩效汇总
    # ---------------------------------------------------------------------
    stitched_perf = calc_performance(stitched)
    summary = {
        "run_mode": "strict_oos_stitch",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "optimizer_method": OPTIMIZER_METHOD,
        "combine_method": COMBINE_METHOD,
        "outer_segments": int(len(segment_navs)),
        "outer_test_start": str(pd.Timestamp(test_start).date()),
        "outer_test_end": str(pd.Timestamp(test_end).date()),
        "annual_return": float(stitched_perf["annual_return"]),
        "annual_vol": float(stitched_perf["annual_vol"]),
        "sharpe": float(stitched_perf["sharpe"]),
        "max_drawdown": float(stitched_perf["max_drawdown"]),
        "calmar": float(stitched_perf["calmar"]),
        "total_return": float(stitched_perf["total_return"]),
        "no_bayes_first_n_outer": int(no_bayes_first_n_outer),
        "first_fold_anchor_params": (
            dict(first_fold_anchor_params) if int(no_bayes_first_n_outer) > 0 else None
        ),
    }
    pd.DataFrame([summary]).to_csv(run_dir / "严格OOS拼接_绩效汇总.csv", index=False)
    with (run_dir / "严格OOS拼接_摘要.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---------------------------------------------------------------------
    # I. 可选图形输出
    # ---------------------------------------------------------------------
    bt_plot_paths = {}
    if args.export_backtest_plots: # False
        bt_plot_paths = export_backtest_plots({"严格OOS拼接": stitched}, output_dir=run_dir)

    print("=" * 70)
    print("严格时间推进 OOS 评估完成")
    print(f"  外层分段数: {len(segment_navs)}")
    print(f"  拼接净值: {stitched_path}")
    print(
        "  拼接绩效: "
        f"annual_return={summary['annual_return']:.2%}, "
        f"sharpe={summary['sharpe']:.4f}, "
        f"max_drawdown={summary['max_drawdown']:.2%}, "
        f"calmar={summary['calmar']:.4f}"
    )
    print(f"  trial 明细: {run_dir / 'STRICT_OOS_trials.csv'}")
    print(f"  分段参数: {run_dir / 'STRICT_OOS_selected_params.csv'}")
    print(f"  分段绩效: {run_dir / 'STRICT_OOS_segment_performance.csv'}")
    print(f"  摘要: {run_dir / '严格OOS拼接_摘要.json'}")
    if bt_plot_paths:
        print(f"  净值图: {bt_plot_paths.get('nav', '')}")
        print(f"  回撤图: {bt_plot_paths.get('drawdown', '')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
