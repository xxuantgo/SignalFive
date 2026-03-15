# -*- coding: utf-8 -*-
"""
CVaR/RP 混合优化器贝叶斯调参（WFO + RP 锚定正则）
=================================================
目标：
  1) 使用滚动步进交叉验证（Walk-Forward Optimization, WFO）做参数搜索
  2) 以 OOS 折的稳健目标函数选参：mean(Sharpe) - std(Sharpe) - worst-fold 惩罚
  3) 在目标函数中加入 RP 锚定正则，抑制过度偏离鲁棒基准
  4) 输出调参明细、最优参数、训练/测试对比与可视化

说明：
  - 候选策略使用 hybrid_cvar_rp（CVaR 与风险平价凸组合）。
  - max_weight 固定 35%，不参与搜索。

数据版本选择（通过命令行参数）：
  --data-version: 选择数据版本
    - "auto": 自动选择最新版本（默认）
    - "v20251030": 使用截止到2025-10-30的数据
    - "v20260313": 使用截止到2026-03-13的数据（增量更新）
    - "current": 使用 current/ 目录下的数据
  
数据日期范围选择：
  --data-start: 数据起始日过滤（应 <= train-start）
  --data-end: 数据截止日过滤（应 >= test-end）
"""
from __future__ import annotations # 协方差估计, 

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from signalfive.analysis.cvar_tuning_plots import generate_cvar_bayes_plots
from signalfive.config import (
    BACKTEST_START,
    COV_LOOKBACK,
    CVAR_METHOD,
    FORWARD_RETURN_PERIODS,
    MIN_HOLDINGS,
    OUTPUT_DIR,
    TOP_N,
)
from signalfive.data_loader.loader import load_all
from signalfive.factors.calc import compute_factors, prepare_factor_matrices
from signalfive.factors.combine import combine_factors, export_composite_factor
from signalfive.factors.testing import test_all_factors, select_effective_factors_from_ic
from signalfive.portfolio.regime import apply_position_scale, calc_position_scale

# 约束口径（与项目里现有评估保持一致）
MDD_WORSE_TOL = 0.005
ANN_RETURN_DROP_TOL = 0.01
FULL_SAMPLE_SHARPE_IMPROVE = 0.02
ABS_SPLIT_PERIODS = [
    ("2021_2022", "2021-01-04", "2022-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024_2025", "2024-01-01", "2025-12-31"),
]

# 用户要求：max_weight 固定 35%，不参与优化
FIXED_MAX_WEIGHT = 0.35


def _require_optuna():
    try:
        import optuna  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "缺少依赖 optuna，无法运行贝叶斯调参。请先安装: `pip install optuna`。"
        ) from exc
    return optuna


def _build_signals_fresh(
    run_dir: Path,
    close_matrix: pd.DataFrame,
    aligned: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    print("=" * 60)
    print("Step 1: 因子计算与合成（fresh）")
    panel_wide, macro_df = compute_factors(aligned)
    processed = prepare_factor_matrices(panel_wide, method="rank")
    _, ic_series_dict = test_all_factors(processed, close_matrix)

    fwd_shift = FORWARD_RETURN_PERIODS[0]
    shifted_ic = {name: ic.shift(fwd_shift) for name, ic in ic_series_dict.items()}
    train_cutoff = pd.Timestamp(BACKTEST_START) - pd.Timedelta(days=1)
    effective, prestart_summary = select_effective_factors_from_ic(
        shifted_ic, cutoff=str(train_cutoff.date())
    )
    prestart_summary.to_csv(run_dir / "单因子测试结果.csv", index=False)
    pd.DataFrame({"因子": effective}).to_csv(
        run_dir / "有效因子.csv", index=False, encoding="utf-8-sig"
    )

    composite = combine_factors(processed, ic_series_dict, effective)
    export_composite_factor(composite, output_path=str(run_dir / "合成因子序列.csv"))

    position_scale = calc_position_scale(macro_df, smooth_window=5)
    ps_df = position_scale.to_frame("position_scale")
    ps_df.index.name = "date"
    ps_df.to_csv(run_dir / "宏观仓位系数.csv")
    return composite, position_scale


def _load_cached_signals(run_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
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


def _tail_metrics(nav: pd.Series, q: float = 0.95) -> Dict[str, float]:
    ret = nav.pct_change().dropna()
    if ret.empty:
        return {"var95_loss": np.nan, "es95_loss": np.nan}
    loss = -ret
    var_q = loss.quantile(q)
    tail = loss[loss >= var_q]
    es_q = tail.mean() if len(tail) > 0 else np.nan
    return {"var95_loss": float(var_q), "es95_loss": float(es_q)}


def _split_sharpes_abs(nav: pd.Series, calc_performance_fn) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for label, start, end in ABS_SPLIT_PERIODS:
        sub = nav.loc[(nav.index >= pd.Timestamp(start)) & (nav.index <= pd.Timestamp(end))]
        if len(sub) < 3:
            out[f"sharpe_{label}"] = np.nan
            continue
        out[f"sharpe_{label}"] = float(calc_performance_fn(sub)["sharpe"])
    return out


def _split_sharpe_ge_count_local(
    nav: pd.Series,
    rp_nav: pd.Series,
    calc_performance_fn,
    n_splits: int = 3,
) -> tuple[int, int]:
    nav = nav.dropna()
    rp_nav = rp_nav.dropna()
    if nav.empty or rp_nav.empty:
        return 0, 0

    idx = nav.index.intersection(rp_nav.index)
    if len(idx) < max(30, n_splits * 6):
        return 0, 0

    nav = nav.reindex(idx)
    rp_nav = rp_nav.reindex(idx)
    parts = np.array_split(np.arange(len(idx)), n_splits)
    ge_cnt = 0
    valid = 0
    for p in parts:
        if len(p) < 3:
            continue
        sn = float(calc_performance_fn(nav.iloc[p])["sharpe"])
        sr = float(calc_performance_fn(rp_nav.iloc[p])["sharpe"])
        if np.isfinite(sn) and np.isfinite(sr):
            valid += 1
            if sn >= sr:
                ge_cnt += 1
    return ge_cnt, valid


def _schedule_structure_metrics(
    raw_schedule: Dict[pd.Timestamp, pd.Series],
    adjusted_schedule: Dict[pd.Timestamp, pd.Series],
    max_weight: float,
) -> Dict[str, float]:
    if not raw_schedule:
        return {
            "avg_holdings": np.nan,
            "avg_hhi": np.nan,
            "avg_cap_hit_ratio": np.nan,
            "avg_turnover": np.nan,
        }

    holdings: List[float] = []
    hhi_vals: List[float] = []
    cap_hit_vals: List[float] = []
    for _, w in sorted(raw_schedule.items(), key=lambda kv: kv[0]):
        w_pos = w[w > 1e-10]
        holdings.append(float(len(w_pos)))
        if len(w_pos) == 0:
            continue
        w_norm = w_pos / w_pos.sum()
        hhi_vals.append(float((w_norm ** 2).sum()))
        cap_hit_vals.append(float((w_pos >= max_weight - 1e-8).mean()))

    turnover_vals: List[float] = []
    prev = None
    for _, w in sorted(adjusted_schedule.items(), key=lambda kv: kv[0]):
        cur = w.sort_index()
        if prev is not None:
            idx = prev.index.union(cur.index)
            turn = 0.5 * (cur.reindex(idx, fill_value=0.0) - prev.reindex(idx, fill_value=0.0)).abs().sum()
            turnover_vals.append(float(turn))
        prev = cur

    return {
        "avg_holdings": float(np.mean(holdings)) if holdings else np.nan,
        "avg_hhi": float(np.mean(hhi_vals)) if hhi_vals else np.nan,
        "avg_cap_hit_ratio": float(np.mean(cap_hit_vals)) if cap_hit_vals else np.nan,
        "avg_turnover": float(np.mean(turnover_vals)) if turnover_vals else np.nan,
    }


def _validate_schedule(
    schedule: Dict[pd.Timestamp, pd.Series],
    max_weight: float,
    min_holdings: int,
    tol: float = 1e-8,
) -> tuple[bool, int]:
    violations = 0
    for _, w in schedule.items():
        if abs(float(w.sum()) - 1.0) > tol:
            violations += 1
        if (w < -tol).any():
            violations += 1
        if (w > max_weight + tol).any():
            violations += 1
        if int((w > 1e-10).sum()) < min_holdings:
            violations += 1
    return violations == 0, violations


def _safe_float(v: object, default: float = float("nan")) -> float:
    try:
        x = float(v)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return default


def _compute_wfo_objective(
    *,
    mean_sharpe: float,
    std_sharpe: float,
    min_sharpe: float,
    mean_excess: float,
    min_excess: float,
    mean_active_share: float,
    mean_turnover: float,
    feasible_all_folds: bool,
    feasible_violations_total: int,
    std_penalty: float,
    worst_penalty: float,
    sharpe_floor: float,
    rp_anchor_lambda: float,
    turnover_penalty: float,
    mean_excess_floor: float,
    worst_excess_floor: float,
    excess_penalty: float,
) -> float:
    obj = float(mean_sharpe)
    obj -= float(std_penalty) * float(std_sharpe)
    obj -= float(worst_penalty) * max(0.0, float(sharpe_floor) - float(min_sharpe))

    if np.isfinite(mean_active_share):
        obj -= float(rp_anchor_lambda) * float(mean_active_share)
    else:
        obj -= float(rp_anchor_lambda)

    if np.isfinite(mean_turnover):
        obj -= float(turnover_penalty) * float(mean_turnover)
    else:
        obj -= float(turnover_penalty)

    if float(mean_excess) < float(mean_excess_floor):
        obj -= float(excess_penalty) * (float(mean_excess_floor) - float(mean_excess))
    if float(min_excess) < float(worst_excess_floor):
        obj -= float(excess_penalty) * (float(worst_excess_floor) - float(min_excess))
    if not bool(feasible_all_folds):
        obj -= 1.0 + 0.05 * float(feasible_violations_total)
    return float(obj)


def _first_trading_on_or_after(index: pd.DatetimeIndex, dt: pd.Timestamp) -> Optional[pd.Timestamp]:
    pos = index.searchsorted(dt, side="left")
    if pos >= len(index):
        return None
    return index[pos]


def _last_trading_on_or_before(index: pd.DatetimeIndex, dt: pd.Timestamp) -> Optional[pd.Timestamp]:
    pos = index.searchsorted(dt, side="right") - 1
    if pos < 0:
        return None
    return index[pos]


def _filter_schedule_by_date(
    schedule: Dict[pd.Timestamp, pd.Series],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Dict[pd.Timestamp, pd.Series]:
    return {dt: w for dt, w in schedule.items() if start <= dt <= end}


def _run_nav_for_period(
    close_matrix: pd.DataFrame,
    schedule: Dict[pd.Timestamp, pd.Series],
    start: pd.Timestamp,
    end: pd.Timestamp,
    strategy_name: str,
    run_backtests_fn,
    extract_nav_fn,
) -> Optional[pd.Series]:
    period_schedule = _filter_schedule_by_date(schedule, start=start, end=end)
    if not period_schedule:
        return None

    px = close_matrix.loc[(close_matrix.index >= start) & (close_matrix.index <= end)]
    if px.empty:
        return None

    try:
        res = run_backtests_fn(px, {strategy_name: period_schedule})
        nav = extract_nav_fn(res, start_date=str(start.date())).get(strategy_name)
    except Exception:  # noqa: BLE001
        return None
    if nav is None or len(nav) < 3:
        return None
    nav = nav.loc[nav.index <= end]
    return nav if len(nav) >= 3 else None


def _nav_row(strategy: str, nav: pd.Series, calc_performance_fn) -> Dict[str, float | str]:
    return {
        "strategy": strategy,
        **calc_performance_fn(nav),
        **_tail_metrics(nav),
        **_split_sharpes_abs(nav, calc_performance_fn),
    }


def _average_active_share(
    schedule_a: Dict[pd.Timestamp, pd.Series],
    schedule_b: Dict[pd.Timestamp, pd.Series],
) -> float:
    common_dates = sorted(set(schedule_a.keys()).intersection(set(schedule_b.keys())))
    if not common_dates:
        return float("nan")

    vals: List[float] = []
    for dt in common_dates:
        wa = schedule_a[dt].sort_index()
        wb = schedule_b[dt].sort_index()
        idx = wa.index.union(wb.index)
        active_share = 0.5 * (
            wa.reindex(idx, fill_value=0.0) - wb.reindex(idx, fill_value=0.0)
        ).abs().sum()
        vals.append(float(active_share))
    return float(np.mean(vals)) if vals else float("nan")


def _build_wfo_folds(
    index: pd.DatetimeIndex,
    train_start_min: pd.Timestamp,
    first_test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    train_years: int,
    test_years: int,
    step_years: int,
    min_train_days: int,
) -> List[Dict[str, object]]:
    folds: List[Dict[str, object]] = []

    k = 0
    while True:
        test_start_raw = first_test_start + pd.DateOffset(years=k * step_years)
        if test_start_raw > test_end:
            break
        test_start = _first_trading_on_or_after(index, pd.Timestamp(test_start_raw))
        if test_start is None or test_start > test_end:
            break

        test_end_raw = pd.Timestamp(test_start_raw) + pd.DateOffset(years=test_years) - pd.Timedelta(days=1)
        test_end_cap = min(pd.Timestamp(test_end_raw), pd.Timestamp(test_end))
        test_end_dt = _last_trading_on_or_before(index, test_end_cap)
        if test_end_dt is None or test_end_dt < test_start:
            k += 1
            continue

        train_end_dt = _last_trading_on_or_before(index, pd.Timestamp(test_start) - pd.Timedelta(days=1))
        if train_end_dt is None:
            k += 1
            continue

        train_start_raw = pd.Timestamp(train_end_dt) - pd.DateOffset(years=train_years) + pd.Timedelta(days=1)
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
                    "fold_id": len(folds) + 1,
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CVaR/RP 混合优化器贝叶斯调参（WFO）")
    
    # 数据版本选择
    parser.add_argument(
        "--data-version",
        type=str,
        default="auto",
        help='数据版本选择: "auto"(自动最新), "v20251030", "v20260313", "current"'
    )
    parser.add_argument(
        "--data-start",
        type=str,
        default=None,
        help="数据起始日期过滤 (YYYY-MM-DD)，用于控制加载的数据量"
    )
    parser.add_argument(
        "--data-end",
        type=str,
        default=None,
        help="数据截止日期过滤 (YYYY-MM-DD)，用于控制加载的数据量"
    )
    
    parser.add_argument("--top-n", type=int, default=TOP_N, help="每期选股数量 Top N（当 low/high 未指定时使用）")
    parser.add_argument("--top-n-low", type=int, default=3, help="TopN 搜索下界")
    parser.add_argument("--top-n-high", type=int, default=7, help="TopN 搜索上界")
    parser.add_argument("--top-n-step", type=int, default=1, help="TopN 搜索步长")
    parser.add_argument("--n-trials", type=int, default=120, help="贝叶斯 trial 数")
    parser.add_argument("--n-startup-trials", type=int, default=24, help="TPE 随机预热 trial 数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--timeout", type=int, default=0, help="调参超时秒数，0 表示不限制")

    parser.add_argument("--alpha-low", type=float, default=0.92)
    parser.add_argument("--alpha-high", type=float, default=0.96)
    parser.add_argument("--window-low", type=int, default=60)
    parser.add_argument("--window-high", type=int, default=220)
    parser.add_argument("--window-step", type=int, default=5)
    parser.add_argument("--lambda-low", type=float, default=1e-3)
    parser.add_argument("--lambda-high", type=float, default=3e-2)
    parser.add_argument("--beta-low", type=float, default=0.1, help="hybrid beta 下界（建议 >=0.1，鼓励 RP 锚定）")
    parser.add_argument("--beta-high", type=float, default=0.5, help="hybrid beta 上界（建议 <=0.5，保留 CVaR 选优能力）")
    parser.add_argument("--beta-step", type=float, default=0.05, help="hybrid beta 步长")
    parser.add_argument(
        "--cvar-methods",
        type=str,
        default="empirical,parametric,cornish_fisher",
        help="参与对照/搜索的 CVaR 方法，逗号分隔：empirical,parametric,cornish_fisher",
    )

    parser.add_argument("--train-start", type=str, default="2019-11-01", help="训练集起始日")
    parser.add_argument("--train-end", type=str, default="2020-12-31", help="训练集结束日")
    parser.add_argument("--test-start", type=str, default="2021-01-04", help="测试集起始日")
    parser.add_argument("--test-end", type=str, default=None, help="测试集结束日（默认=数据最新日期）")
    parser.add_argument("--min-train-days", type=int, default=220, help="训练集最少交易日")

    # WFO 配置：基于 test-start 起点按年滚动
    parser.add_argument("--wfo-train-years", type=int, default=2, help="WFO 每折训练窗口（年）")
    parser.add_argument("--wfo-test-years", type=int, default=1, help="WFO 每折测试窗口（年）")
    parser.add_argument("--wfo-step-years", type=int, default=1, help="WFO 滚动步长（年）")
    parser.add_argument("--wfo-min-folds", type=int, default=3, help="WFO 至少有效折数")

    # WFO 目标函数：mean(OOS Sharpe) - std 惩罚 - worst-fold 惩罚 - RP 锚定正则
    parser.add_argument("--obj-std-penalty", type=float, default=1.0, help="std(OOS Sharpe) 惩罚系数")
    parser.add_argument("--obj-worst-penalty", type=float, default=1.0, help="worst-fold 惩罚系数")
    parser.add_argument("--obj-sharpe-floor", type=float, default=0.0, help="worst-fold Sharpe 下限")
    parser.add_argument("--rp-anchor-lambda", type=float, default=0.2, help="与 RP 偏离(active share)惩罚系数")
    parser.add_argument("--obj-turnover-penalty", type=float, default=0.2, help="平均换手惩罚系数")
    parser.add_argument("--wfo-mean-excess-floor", type=float, default=-0.02, help="mean(OOS Sharpe-RP) 软下限")
    parser.add_argument("--wfo-worst-excess-floor", type=float, default=-0.20, help="worst(OOS Sharpe-RP) 软下限")
    parser.add_argument("--wfo-excess-penalty", type=float, default=1.0, help="Sharpe 超额软下限惩罚系数")
    parser.add_argument(
        "--pruner",
        type=str,
        default="median",
        choices=["none", "median"],
        help="Optuna 剪枝器类型：none=关闭, median=中位数剪枝",
    )
    parser.add_argument("--pruner-startup-trials", type=int, default=20, help="剪枝前最少完整 trial 数")
    parser.add_argument("--pruner-warmup-steps", type=int, default=2, help="每个 trial 的 warmup fold 步数")

    parser.add_argument(
        "--reuse-run-dir",
        type=str,
        default="",
        help="可选：复用某次 run_main 输出目录（读取 合成因子序列.csv 与 宏观仓位系数.csv）",
    )
    parser.add_argument("--export-nav", action="store_true", help="导出训练/测试净值 CSV")
    parser.add_argument("--export-plots", action="store_true", help="导出调参可视化图")
    parser.add_argument(
        "--export-backtest-plots",
        action="store_true",
        help="导出测试集净值/回撤图（RP vs best_candidate）",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    optuna = _require_optuna()

    from signalfive.backtest.engine import (
        build_optimized_schedule,
        calc_performance,
        export_backtest_plots,
        export_nav,
        extract_nav,
        run_backtests,
    )

    if args.alpha_low <= 0 or args.alpha_high >= 1 or args.alpha_low >= args.alpha_high:
        raise ValueError("alpha 搜索区间必须满足 0<low<high<1。")
    if args.window_low < 20 or args.window_high < args.window_low:
        raise ValueError("window 搜索区间非法。")
    if args.lambda_low <= 0 or args.lambda_high <= args.lambda_low:
        raise ValueError("turnover_lambda 搜索区间必须满足 0<low<high。")
    if not (0.0 <= args.beta_low <= args.beta_high <= 1.0):
        raise ValueError("beta 搜索区间必须满足 0<=low<=high<=1。")
    if args.beta_step <= 0:
        raise ValueError("beta_step 必须 > 0。")
    if args.n_trials <= 0:
        raise ValueError("n_trials 必须 > 0。")
    if args.min_train_days <= 0:
        raise ValueError("min_train_days 必须 > 0。")
    if args.top_n_low <= 0 or args.top_n_high < args.top_n_low or args.top_n_step <= 0:
        raise ValueError("TopN 搜索区间非法。")
    if args.top_n_low < MIN_HOLDINGS:
        raise ValueError(f"top_n_low 不能小于最小持仓数 MIN_HOLDINGS={MIN_HOLDINGS}。")
    if args.wfo_train_years <= 0 or args.wfo_test_years <= 0 or args.wfo_step_years <= 0:
        raise ValueError("wfo-train-years / wfo-test-years / wfo-step-years 必须 > 0。")
    if args.wfo_min_folds <= 0:
        raise ValueError("wfo-min-folds 必须 > 0。")
    if args.obj_std_penalty < 0 or args.obj_worst_penalty < 0:
        raise ValueError("obj-std-penalty / obj-worst-penalty 不能为负。")
    if args.rp_anchor_lambda < 0 or args.wfo_excess_penalty < 0 or args.obj_turnover_penalty < 0:
        raise ValueError("rp-anchor-lambda / wfo-excess-penalty / obj-turnover-penalty 不能为负。")
    if args.pruner_startup_trials < 0 or args.pruner_warmup_steps < 0:
        raise ValueError("pruner-startup-trials / pruner-warmup-steps 不能为负。")

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
    if not cvar_methods:
        cvar_methods = [CVAR_METHOD]
    invalid_methods = sorted(set(cvar_methods) - valid_methods)
    if invalid_methods:
        raise ValueError(f"非法 cvar-methods: {invalid_methods}，可选: {sorted(valid_methods)}")
    cvar_methods = sorted(set(cvar_methods))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(OUTPUT_DIR) / f"cvar_hybrid_bayes_split_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {run_dir}")

    print("=" * 60)
    print("配置信息")
    print("=" * 60)
    print(f"数据版本: {args.data_version}")
    if args.data_start:
        print(f"数据起始过滤: {args.data_start}")
    if args.data_end:
        print(f"数据截止过滤: {args.data_end}")
    print()
    
    print("=" * 60)
    print("Step 0: 数据加载")
    data = load_all(
        version=args.data_version,
        data_start=args.data_start,
        data_end=args.data_end,
    )
    close_matrix = data["close_matrix"].sort_index()
    
    # 打印实际加载的数据范围
    print(f"  实际数据范围: {data['data_range']['start']} ~ {data['data_range']['end']}")

    reuse_run_dir = Path(args.reuse_run_dir).expanduser() if args.reuse_run_dir else None
    if reuse_run_dir is not None:
        print("=" * 60)
        print(f"Step 1: 复用信号缓存: {reuse_run_dir}")
        composite, position_scale = _load_cached_signals(reuse_run_dir)
    else:
        composite, position_scale = _build_signals_fresh(
            run_dir=run_dir,
            close_matrix=close_matrix,
            aligned=data["aligned"],
        )
    composite = composite.sort_index()
    position_scale = position_scale.sort_index()

    # 日期切分（映射到交易日）
    train_start_raw = pd.Timestamp(args.train_start)
    train_end_raw = pd.Timestamp(args.train_end)
    test_start_raw = pd.Timestamp(args.test_start)
    test_end_raw = pd.Timestamp(args.test_end) if args.test_end else pd.Timestamp(close_matrix.index.max())

    train_start = _first_trading_on_or_after(close_matrix.index, train_start_raw)
    train_end = _last_trading_on_or_before(close_matrix.index, train_end_raw)
    test_start = _first_trading_on_or_after(close_matrix.index, test_start_raw)
    test_end = _last_trading_on_or_before(close_matrix.index, test_end_raw)
    if any(x is None for x in [train_start, train_end, test_start, test_end]):
        raise ValueError("训练/测试日期不在有效交易日范围内，请检查输入。")

    train_start = pd.Timestamp(train_start)
    train_end = pd.Timestamp(train_end)
    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end)

    if not (train_start <= train_end < test_start <= test_end):
        raise ValueError(
            "日期切分非法，需满足 train_start <= train_end < test_start <= test_end。"
        )
    train_days = int(((close_matrix.index >= train_start) & (close_matrix.index <= train_end)).sum())
    if train_days < int(args.min_train_days):
        raise ValueError(
            f"训练集交易日不足: {train_days} < {int(args.min_train_days)}。"
        )

    split_df = pd.DataFrame(
        [
            {
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "train_days": train_days,
                "top_n_low": int(args.top_n_low),
                "top_n_high": int(args.top_n_high),
                "top_n_step": int(args.top_n_step),
                "fixed_max_weight": FIXED_MAX_WEIGHT,
                "min_holdings": MIN_HOLDINGS,
                "rp_cov_window": COV_LOOKBACK,
                "alpha_low": float(args.alpha_low),
                "alpha_high": float(args.alpha_high),
                "window_low": int(args.window_low),
                "window_high": int(args.window_high),
                "lambda_low": float(args.lambda_low),
                "lambda_high": float(args.lambda_high),
                "beta_low": float(args.beta_low),
                "beta_high": float(args.beta_high),
                "cvar_methods": ",".join(cvar_methods),
                "wfo_train_years": int(args.wfo_train_years),
                "wfo_test_years": int(args.wfo_test_years),
                "wfo_step_years": int(args.wfo_step_years),
                "wfo_min_folds": int(args.wfo_min_folds),
                "obj_std_penalty": float(args.obj_std_penalty),
                "obj_worst_penalty": float(args.obj_worst_penalty),
                "obj_sharpe_floor": float(args.obj_sharpe_floor),
                "rp_anchor_lambda": float(args.rp_anchor_lambda),
                "obj_turnover_penalty": float(args.obj_turnover_penalty),
                "pruner": str(args.pruner),
                "pruner_startup_trials": int(args.pruner_startup_trials),
                "pruner_warmup_steps": int(args.pruner_warmup_steps),
            }
        ]
    )
    split_path = run_dir / "CVAR贝叶斯_split_config.csv"
    split_df.to_csv(split_path, index=False)

    wfo_folds = _build_wfo_folds(
        index=close_matrix.index,
        train_start_min=train_start,
        first_test_start=test_start,
        test_end=test_end,
        train_years=int(args.wfo_train_years),
        test_years=int(args.wfo_test_years),
        step_years=int(args.wfo_step_years),
        min_train_days=int(args.min_train_days),
    )
    if len(wfo_folds) < int(args.wfo_min_folds):
        raise ValueError(
            f"WFO 有效折数不足: {len(wfo_folds)} < {int(args.wfo_min_folds)}，"
            "请扩大样本区间或降低 min-train-days / wfo-min-folds。"
        )
    fold_df = pd.DataFrame(wfo_folds)
    wfo_path = run_dir / "CVAR贝叶斯_WFO折配置.csv"
    fold_df.to_csv(wfo_path, index=False)

    print("=" * 60)
    print(
        f"Step 2: WFO 切分完成 "
        f"base-train[{train_start.date()}~{train_end.date()}], "
        f"test-range[{test_start.date()}~{test_end.date()}], "
        f"folds={len(wfo_folds)}"
    )

    rp_train_cache: Dict[int, Dict[str, object]] = {}

    def _get_rp_train_ref(top_n: int) -> Dict[str, object]:
        top_n = int(top_n)
        if top_n in rp_train_cache:
            return rp_train_cache[top_n]
        rp_train_raw_all = build_optimized_schedule(
            composite.loc[:train_end],
            close_matrix.loc[:train_end],
            optimizer="risk_parity",
            top_n=top_n,
            max_weight=FIXED_MAX_WEIGHT,
            min_holdings=MIN_HOLDINGS,
            cov_window=COV_LOOKBACK,
            rebal_start=train_start,
        )
        rp_train_raw = _filter_schedule_by_date(rp_train_raw_all, start=train_start, end=train_end)
        rp_train_adj = apply_position_scale(rp_train_raw, position_scale)
        rp_train_nav = _run_nav_for_period(
            close_matrix=close_matrix,
            schedule=rp_train_adj,
            start=train_start,
            end=train_end,
            strategy_name=f"rp_train_top{top_n}",
            run_backtests_fn=run_backtests,
            extract_nav_fn=extract_nav,
        )
        if rp_train_nav is None:
            raise ValueError(f"训练期 RP 净值为空，top_n={top_n} 无法调参。")
        rp_train_perf = calc_performance(rp_train_nav)
        ref = {
            "rp_nav": rp_train_nav,
            "mdd_limit": abs(float(rp_train_perf["max_drawdown"])) + MDD_WORSE_TOL,
            "ann_ret_limit": float(rp_train_perf["annual_return"]) - ANN_RETURN_DROP_TOL,
            "sharpe_target": float(rp_train_perf["sharpe"]) + FULL_SAMPLE_SHARPE_IMPROVE,
        }
        rp_train_cache[top_n] = ref
        return ref

    rp_fold_cache: Dict[tuple[int, int], Dict[str, object]] = {}

    def _get_rp_fold_ref(top_n: int, fold: Dict[str, object]) -> Dict[str, object]:
        fold_id = int(fold["fold_id"])
        key = (int(top_n), fold_id)
        if key in rp_fold_cache:
            return rp_fold_cache[key]

        fold_test_start = pd.Timestamp(fold["test_start"])
        fold_test_end = pd.Timestamp(fold["test_end"])
        rp_raw_all = build_optimized_schedule(
            composite.loc[:fold_test_end],
            close_matrix.loc[:fold_test_end],
            optimizer="risk_parity",
            top_n=int(top_n),
            max_weight=FIXED_MAX_WEIGHT,
            min_holdings=MIN_HOLDINGS,
            cov_window=COV_LOOKBACK,
            rebal_start=fold_test_start,
        )
        rp_raw = _filter_schedule_by_date(rp_raw_all, start=fold_test_start, end=fold_test_end)
        if not rp_raw:
            raise ValueError(f"RP 在 WFO fold#{fold_id} 调仓计划为空，top_n={top_n}")
        rp_adj = apply_position_scale(rp_raw, position_scale)
        rp_nav = _run_nav_for_period(
            close_matrix=close_matrix,
            schedule=rp_adj,
            start=fold_test_start,
            end=fold_test_end,
            strategy_name=f"rp_fold{fold_id}_top{top_n}",
            run_backtests_fn=run_backtests,
            extract_nav_fn=extract_nav,
        )
        if rp_nav is None:
            raise ValueError(f"RP 在 WFO fold#{fold_id} 净值为空，top_n={top_n}")

        rp_struct = _schedule_structure_metrics(rp_raw, rp_adj, max_weight=FIXED_MAX_WEIGHT)
        ref = {
            "raw": rp_raw,
            "adj": rp_adj,
            "nav": rp_nav,
            "perf": calc_performance(rp_nav),
            **rp_struct,
        }
        rp_fold_cache[key] = ref
        return ref

    trial_rows: List[Dict[str, object]] = []

    def evaluate_candidate(
        trial_no: int,
        top_n: int,
        cvar_alpha: float,
        cov_window: int,
        turnover_lambda: float,
        hybrid_beta: float,
        cvar_method: str,
        trial: Optional[object] = None,
    ) -> Dict[str, object]:
        rec: Dict[str, object] = {
            "trial_number": int(trial_no),
            "train_start": str(train_start.date()),
            "train_end": str(train_end.date()),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "top_n": int(top_n),
            "cvar_alpha": float(cvar_alpha),
            "cvar_method": str(cvar_method),
            "cov_window": int(cov_window),
            "max_weight": FIXED_MAX_WEIGHT,
            "turnover_lambda": float(turnover_lambda),
            "hybrid_beta": float(hybrid_beta),
            "wfo_total_folds": int(len(wfo_folds)),
            "wfo_valid_folds": 0,
            "wfo_mean_sharpe": np.nan,
            "wfo_std_sharpe": np.nan,
            "wfo_min_sharpe": np.nan,
            "wfo_mean_sharpe_excess": np.nan,
            "wfo_min_sharpe_excess": np.nan,
            "avg_turnover": np.nan,
            "avg_active_share_to_rp": np.nan,
            "feasible_all_dates": False,
            "feasible_violations": 0,
            "enough_wfo_folds": False,
            "worst_sharpe_ok": False,
            "mean_excess_ok": False,
            "worst_excess_ok": False,
            "pass_all_rules": False,
            "status": "ok",
            "error": "",
            "interim_objective": np.nan,
            "pruned_at_fold": np.nan,
            "is_pruned": False,
        }
        try:
            fold_sharpes: List[float] = []
            fold_excess: List[float] = []
            fold_turnover: List[float] = []
            fold_active_share: List[float] = []
            feasible_flags: List[bool] = []
            feasible_violations_total = 0

            for fold_step, fold in enumerate(wfo_folds, start=1):
                fold_id = int(fold["fold_id"])
                fold_test_start = pd.Timestamp(fold["test_start"])
                fold_test_end = pd.Timestamp(fold["test_end"])

                raw_all = build_optimized_schedule(
                    composite.loc[:fold_test_end],
                    close_matrix.loc[:fold_test_end],
                    optimizer="hybrid_cvar_rp",
                    top_n=int(top_n),
                    max_weight=FIXED_MAX_WEIGHT,
                    min_holdings=MIN_HOLDINGS,
                    cov_window=int(cov_window),
                    cvar_alpha=float(cvar_alpha),
                    cvar_method=str(cvar_method),
                    turnover_lambda=float(turnover_lambda),
                    hybrid_beta=float(hybrid_beta),
                    rebal_start=fold_test_start,
                )
                raw_fold = _filter_schedule_by_date(raw_all, start=fold_test_start, end=fold_test_end)
                feasible_all, feasible_violations = _validate_schedule(
                    raw_fold,
                    max_weight=FIXED_MAX_WEIGHT,
                    min_holdings=MIN_HOLDINGS,
                )
                feasible_flags.append(bool(feasible_all))
                feasible_violations_total += int(feasible_violations)
                if not raw_fold:
                    continue

                adj_fold = apply_position_scale(raw_fold, position_scale)
                cand_nav = _run_nav_for_period(
                    close_matrix=close_matrix,
                    schedule=adj_fold,
                    start=fold_test_start,
                    end=fold_test_end,
                    strategy_name=f"cand_t{trial_no}_fold{fold_id}",
                    run_backtests_fn=run_backtests,
                    extract_nav_fn=extract_nav,
                )
                if cand_nav is None:
                    continue

                cand_perf = calc_performance(cand_nav)
                rp_ref = _get_rp_fold_ref(top_n=int(top_n), fold=fold)
                rp_sharpe = float(rp_ref["perf"]["sharpe"])
                cand_sharpe = float(cand_perf["sharpe"])
                excess = cand_sharpe - rp_sharpe
                rec[f"fold{fold_id}_sharpe"] = cand_sharpe
                rec[f"fold{fold_id}_rp_sharpe"] = rp_sharpe
                rec[f"fold{fold_id}_sharpe_excess"] = excess

                fold_sharpes.append(cand_sharpe)
                fold_excess.append(excess)

                struct = _schedule_structure_metrics(raw_fold, adj_fold, max_weight=FIXED_MAX_WEIGHT)
                avg_turn = _safe_float(struct.get("avg_turnover"), default=np.nan)
                if np.isfinite(avg_turn):
                    fold_turnover.append(float(avg_turn))
                active_share = _average_active_share(raw_fold, rp_ref["raw"])
                if np.isfinite(active_share):
                    fold_active_share.append(float(active_share))

                if trial is not None and fold_sharpes:
                    sharpe_arr_tmp = np.array(fold_sharpes, dtype=float)
                    excess_arr_tmp = np.array(fold_excess, dtype=float)
                    mean_sharpe_tmp = float(np.mean(sharpe_arr_tmp))
                    std_sharpe_tmp = float(np.std(sharpe_arr_tmp, ddof=1)) if len(sharpe_arr_tmp) > 1 else 0.0
                    min_sharpe_tmp = float(np.min(sharpe_arr_tmp))
                    mean_excess_tmp = float(np.mean(excess_arr_tmp))
                    min_excess_tmp = float(np.min(excess_arr_tmp))
                    mean_turnover_tmp = float(np.mean(fold_turnover)) if fold_turnover else np.nan
                    mean_active_share_tmp = float(np.mean(fold_active_share)) if fold_active_share else np.nan
                    feasible_all_folds_tmp = bool(all(feasible_flags)) if feasible_flags else False
                    interim_obj = _compute_wfo_objective(
                        mean_sharpe=mean_sharpe_tmp,
                        std_sharpe=std_sharpe_tmp,
                        min_sharpe=min_sharpe_tmp,
                        mean_excess=mean_excess_tmp,
                        min_excess=min_excess_tmp,
                        mean_active_share=mean_active_share_tmp,
                        mean_turnover=mean_turnover_tmp,
                        feasible_all_folds=feasible_all_folds_tmp,
                        feasible_violations_total=feasible_violations_total,
                        std_penalty=float(args.obj_std_penalty),
                        worst_penalty=float(args.obj_worst_penalty),
                        sharpe_floor=float(args.obj_sharpe_floor),
                        rp_anchor_lambda=float(args.rp_anchor_lambda),
                        turnover_penalty=float(args.obj_turnover_penalty),
                        mean_excess_floor=float(args.wfo_mean_excess_floor),
                        worst_excess_floor=float(args.wfo_worst_excess_floor),
                        excess_penalty=float(args.wfo_excess_penalty),
                    )
                    trial.report(interim_obj, step=int(fold_step))
                    if trial.should_prune():
                        rec.update(
                            {
                                "wfo_total_folds": int(len(wfo_folds)),
                                "wfo_valid_folds": int(len(sharpe_arr_tmp)),
                                "wfo_mean_sharpe": mean_sharpe_tmp,
                                "wfo_std_sharpe": std_sharpe_tmp,
                                "wfo_min_sharpe": min_sharpe_tmp,
                                "wfo_mean_sharpe_excess": mean_excess_tmp,
                                "wfo_min_sharpe_excess": min_excess_tmp,
                                "avg_turnover": mean_turnover_tmp,
                                "avg_active_share_to_rp": mean_active_share_tmp,
                                "feasible_all_dates": bool(feasible_all_folds_tmp),
                                "feasible_violations": int(feasible_violations_total),
                                "enough_wfo_folds": int(len(sharpe_arr_tmp)) >= int(args.wfo_min_folds),
                                "worst_sharpe_ok": min_sharpe_tmp >= float(args.obj_sharpe_floor),
                                "mean_excess_ok": mean_excess_tmp >= float(args.wfo_mean_excess_floor),
                                "worst_excess_ok": min_excess_tmp >= float(args.wfo_worst_excess_floor),
                                "pass_all_rules": False,
                                "sharpe": mean_sharpe_tmp,
                                "status": "pruned",
                                "interim_objective": float(interim_obj),
                                "pruned_at_fold": int(fold_id),
                                "is_pruned": True,
                                "objective": -1e6,
                            }
                        )
                        return rec

            valid_folds = len(fold_sharpes)
            if valid_folds == 0:
                rec["status"] = "empty_wfo"
                rec["objective"] = -1e6
                return rec

            sharpe_arr = np.array(fold_sharpes, dtype=float)
            excess_arr = np.array(fold_excess, dtype=float)
            mean_sharpe = float(np.mean(sharpe_arr))
            std_sharpe = float(np.std(sharpe_arr, ddof=1)) if valid_folds > 1 else 0.0
            min_sharpe = float(np.min(sharpe_arr))
            mean_excess = float(np.mean(excess_arr))
            min_excess = float(np.min(excess_arr))
            mean_turnover = float(np.mean(fold_turnover)) if fold_turnover else np.nan
            mean_active_share = float(np.mean(fold_active_share)) if fold_active_share else np.nan
            feasible_all_folds = bool(all(feasible_flags)) if feasible_flags else False

            enough_folds = valid_folds >= int(args.wfo_min_folds)
            worst_sharpe_ok = min_sharpe >= float(args.obj_sharpe_floor)
            mean_excess_ok = mean_excess >= float(args.wfo_mean_excess_floor)
            worst_excess_ok = min_excess >= float(args.wfo_worst_excess_floor)
            pass_all_rules = (
                enough_folds
                and feasible_all_folds
                and worst_sharpe_ok
                and mean_excess_ok
                and worst_excess_ok
            )

            rec.update(
                {
                    "wfo_total_folds": int(len(wfo_folds)),
                    "wfo_valid_folds": int(valid_folds),
                    "wfo_mean_sharpe": mean_sharpe,
                    "wfo_std_sharpe": std_sharpe,
                    "wfo_min_sharpe": min_sharpe,
                    "wfo_mean_sharpe_excess": mean_excess,
                    "wfo_min_sharpe_excess": min_excess,
                    "avg_turnover": mean_turnover,
                    "avg_active_share_to_rp": mean_active_share,
                    "feasible_all_dates": bool(feasible_all_folds),
                    "feasible_violations": int(feasible_violations_total),
                    "enough_wfo_folds": bool(enough_folds),
                    "worst_sharpe_ok": bool(worst_sharpe_ok),
                    "mean_excess_ok": bool(mean_excess_ok),
                    "worst_excess_ok": bool(worst_excess_ok),
                    "pass_all_rules": bool(pass_all_rules),
                    # 兼容现有可视化与排序字段
                    "sharpe": mean_sharpe,
                }
            )

            if not enough_folds:
                rec["status"] = "insufficient_wfo_folds"
                rec["objective"] = -1e6
                return rec

            obj = _compute_wfo_objective(
                mean_sharpe=mean_sharpe,
                std_sharpe=std_sharpe,
                min_sharpe=min_sharpe,
                mean_excess=mean_excess,
                min_excess=min_excess,
                mean_active_share=mean_active_share,
                mean_turnover=mean_turnover,
                feasible_all_folds=feasible_all_folds,
                feasible_violations_total=feasible_violations_total,
                std_penalty=float(args.obj_std_penalty),
                worst_penalty=float(args.obj_worst_penalty),
                sharpe_floor=float(args.obj_sharpe_floor),
                rp_anchor_lambda=float(args.rp_anchor_lambda),
                turnover_penalty=float(args.obj_turnover_penalty),
                mean_excess_floor=float(args.wfo_mean_excess_floor),
                worst_excess_floor=float(args.wfo_worst_excess_floor),
                excess_penalty=float(args.wfo_excess_penalty),
            )

            rec["objective"] = float(obj)
            return rec
        except Exception as exc:  # noqa: BLE001
            rec["status"] = "exception"
            rec["error"] = str(exc)
            rec["objective"] = -1e6
            return rec

    sampler = optuna.samplers.TPESampler(
        seed=int(args.seed),
        n_startup_trials=max(1, int(args.n_startup_trials)),
        multivariate=True,
    )
    if str(args.pruner).lower() == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=int(args.pruner_startup_trials),
            n_warmup_steps=int(args.pruner_warmup_steps),
            interval_steps=1,
        )
    else:
        pruner = optuna.pruners.NopPruner()
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    def objective(trial) -> float:
        top_n = trial.suggest_int(
            "top_n",
            int(args.top_n_low),
            int(args.top_n_high),
            step=int(args.top_n_step),
        )
        cvar_alpha = trial.suggest_float("cvar_alpha", args.alpha_low, args.alpha_high)
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
        rec = evaluate_candidate(
            trial_no=trial.number,
            top_n=top_n,
            cvar_alpha=cvar_alpha,
            cov_window=cov_window,
            turnover_lambda=turnover_lambda,
            hybrid_beta=hybrid_beta,
            cvar_method=cvar_method,
            trial=trial,
        )
        trial_rows.append(rec)
        if rec.get("status") == "pruned":
            raise optuna.TrialPruned(f"pruned_at_fold={rec.get('pruned_at_fold')}")
        return float(rec["objective"])

    print("=" * 60)
    print(
        f"Step 3: 启动贝叶斯调参（optimizer=hybrid_cvar_rp, "
        f"fixed_max_weight={FIXED_MAX_WEIGHT:.2f}, "
        f"cvar_methods={cvar_methods}, "
        f"objective=mean_sharpe-std_penalty*std-worst_penalty*shortfall-rp_anchor-turnover_penalty*avg_turnover, "
        f"pruner={str(args.pruner).lower()}, "
        f"top_n_range=[{int(args.top_n_low)},{int(args.top_n_high)}], "
        f"n_trials={int(args.n_trials)}）"
    )
    study.optimize(
        objective,
        n_trials=int(args.n_trials),
        timeout=(None if int(args.timeout) <= 0 else int(args.timeout)),
        n_jobs=1,
        gc_after_trial=True,
        show_progress_bar=False,
    )

    if not trial_rows:
        raise ValueError("未产出任何 trial 结果。")

    trials_df = pd.DataFrame(trial_rows).sort_values(
        ["objective", "trial_number"], ascending=[False, True]
    ).reset_index(drop=True)
    trials_path = run_dir / "CVAR贝叶斯_trials.csv"
    trials_df.to_csv(trials_path, index=False)

    method_cmp_path = run_dir / "CVAR贝叶斯_CVaR方法对照.csv"
    if "cvar_method" in trials_df.columns:
        method_cmp_df = (
            trials_df.groupby("cvar_method", dropna=False)
            .agg(
                trials=("objective", "count"),
                mean_objective=("objective", "mean"),
                best_objective=("objective", "max"),
                mean_wfo_sharpe=("wfo_mean_sharpe", "mean"),
                best_wfo_sharpe=("wfo_mean_sharpe", "max"),
                mean_wfo_excess=("wfo_mean_sharpe_excess", "mean"),
                best_wfo_excess=("wfo_mean_sharpe_excess", "max"),
                mean_active_share=("avg_active_share_to_rp", "mean"),
                pass_rate=("pass_all_rules", lambda x: float(pd.Series(x).fillna(False).astype(bool).mean())),
            )
            .reset_index()
            .sort_values(["best_objective", "mean_objective"], ascending=[False, False])
        )
        method_cmp_df.to_csv(method_cmp_path, index=False)
    else:
        method_cmp_df = pd.DataFrame()

    pass_df = trials_df[trials_df["pass_all_rules"].fillna(False).astype(bool)].copy()
    pass_path = run_dir / "CVAR贝叶斯_通过规则.csv"
    pass_df.to_csv(pass_path, index=False)

    top20 = trials_df.sort_values(["objective", "trial_number"], ascending=[False, True]).head(20)
    top20_path = run_dir / "CVAR贝叶斯_top20.csv"
    top20.to_csv(top20_path, index=False)

    if not pass_df.empty:
        best_row = pass_df.sort_values(
            [
                "objective",
                "wfo_mean_sharpe",
                "wfo_mean_sharpe_excess",
                "wfo_min_sharpe",
                "avg_active_share_to_rp",
                "avg_turnover",
            ],
            ascending=[False, False, False, False, True, True],
        ).iloc[0]
        selected_from = "pass_all_rules"
    else:
        fallback_df = trials_df[~trials_df["status"].isin(["pruned", "exception"])].copy()
        best_row = (fallback_df if not fallback_df.empty else trials_df).iloc[0]
        selected_from = "objective_fallback"

    best_params = {
        "top_n": int(best_row["top_n"]),
        "cvar_alpha": float(best_row["cvar_alpha"]),
        "cvar_method": str(best_row.get("cvar_method", CVAR_METHOD)),
        "cov_window": int(best_row["cov_window"]),
        "max_weight": FIXED_MAX_WEIGHT,
        "turnover_lambda": float(best_row["turnover_lambda"]),
        "hybrid_beta": float(best_row["hybrid_beta"]),
    }

    rp_train_ref_best = _get_rp_train_ref(int(best_params["top_n"]))
    rp_train_nav = rp_train_ref_best["rp_nav"]

    # 用最优参数重跑训练集（用于输出对比与可选导出）
    cand_train_raw_all = build_optimized_schedule(
        composite.loc[:train_end],
        close_matrix.loc[:train_end],
        optimizer="hybrid_cvar_rp",
        top_n=best_params["top_n"],
        max_weight=FIXED_MAX_WEIGHT,
        min_holdings=MIN_HOLDINGS,
        cov_window=best_params["cov_window"],
        cvar_alpha=best_params["cvar_alpha"],
        cvar_method=best_params["cvar_method"],
        turnover_lambda=best_params["turnover_lambda"],
        hybrid_beta=best_params["hybrid_beta"],
        rebal_start=train_start,
    )
    cand_train_raw = _filter_schedule_by_date(cand_train_raw_all, start=train_start, end=train_end)
    cand_train_adj = apply_position_scale(cand_train_raw, position_scale)
    cand_train_nav = _run_nav_for_period(
        close_matrix=close_matrix,
        schedule=cand_train_adj,
        start=train_start,
        end=train_end,
        strategy_name="cand_train_best",
        run_backtests_fn=run_backtests,
        extract_nav_fn=extract_nav,
    )
    if cand_train_nav is None:
        raise ValueError("最优参数在训练集的净值为空。")

    # 测试集 OOS：candidate vs RP
    cand_test_raw_all = build_optimized_schedule(
        composite.loc[:test_end],
        close_matrix.loc[:test_end],
        optimizer="hybrid_cvar_rp",
        top_n=best_params["top_n"],
        max_weight=FIXED_MAX_WEIGHT,
        min_holdings=MIN_HOLDINGS,
        cov_window=best_params["cov_window"],
        cvar_alpha=best_params["cvar_alpha"],
        cvar_method=best_params["cvar_method"],
        turnover_lambda=best_params["turnover_lambda"],
        hybrid_beta=best_params["hybrid_beta"],
        rebal_start=test_start,
    )
    cand_test_raw = _filter_schedule_by_date(cand_test_raw_all, start=test_start, end=test_end)
    cand_test_adj = apply_position_scale(cand_test_raw, position_scale)
    cand_test_nav = _run_nav_for_period(
        close_matrix=close_matrix,
        schedule=cand_test_adj,
        start=test_start,
        end=test_end,
        strategy_name="cand_test_best",
        run_backtests_fn=run_backtests,
        extract_nav_fn=extract_nav,
    )
    if cand_test_nav is None:
        raise ValueError("最优参数在测试集的净值为空。")

    rp_test_raw_all = build_optimized_schedule(
        composite.loc[:test_end],
        close_matrix.loc[:test_end],
        optimizer="risk_parity",
        top_n=best_params["top_n"],
        max_weight=FIXED_MAX_WEIGHT,
        min_holdings=MIN_HOLDINGS,
        cov_window=COV_LOOKBACK,
        rebal_start=test_start,
    )
    rp_test_raw = _filter_schedule_by_date(rp_test_raw_all, start=test_start, end=test_end)
    rp_test_adj = apply_position_scale(rp_test_raw, position_scale)
    rp_test_nav = _run_nav_for_period(
        close_matrix=close_matrix,
        schedule=rp_test_adj,
        start=test_start,
        end=test_end,
        strategy_name="rp_test",
        run_backtests_fn=run_backtests,
        extract_nav_fn=extract_nav,
    )
    if rp_test_nav is None:
        raise ValueError("测试集 RP 净值为空。")

    train_rows = []
    if rp_train_nav is not None:
        train_rows.append(_nav_row("RP_baseline_train", rp_train_nav, calc_performance))
    train_rows.append(_nav_row("best_candidate_train", cand_train_nav, calc_performance))
    train_compare_df = pd.DataFrame(train_rows)
    train_compare_path = run_dir / "CVAR贝叶斯_训练集对比.csv"
    train_compare_df.to_csv(train_compare_path, index=False)

    test_compare_df = pd.DataFrame(
        [
            _nav_row("RP_baseline_test", rp_test_nav, calc_performance),
            _nav_row("best_candidate_test", cand_test_nav, calc_performance),
        ]
    )
    test_compare_path = run_dir / "CVAR贝叶斯_测试集对比.csv"
    test_compare_df.to_csv(test_compare_path, index=False)

    oos_sharpe_excess = float(test_compare_df.iloc[1]["sharpe"] - test_compare_df.iloc[0]["sharpe"])
    summary_df = pd.DataFrame(
        [
            {
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "selected_from": selected_from,
                **best_params,
                "wfo_total_folds": _safe_float(best_row.get("wfo_total_folds")),
                "wfo_valid_folds": _safe_float(best_row.get("wfo_valid_folds")),
                "wfo_best_objective": _safe_float(best_row.get("objective")),
                "wfo_best_mean_sharpe": _safe_float(best_row.get("wfo_mean_sharpe")),
                "wfo_best_std_sharpe": _safe_float(best_row.get("wfo_std_sharpe")),
                "wfo_best_min_sharpe": _safe_float(best_row.get("wfo_min_sharpe")),
                "wfo_best_mean_excess": _safe_float(best_row.get("wfo_mean_sharpe_excess")),
                "wfo_best_min_excess": _safe_float(best_row.get("wfo_min_sharpe_excess")),
                "wfo_best_avg_active_share_to_rp": _safe_float(best_row.get("avg_active_share_to_rp")),
                "wfo_pass_all_rules": bool(best_row.get("pass_all_rules", False)),
                "test_rp_sharpe": float(test_compare_df.iloc[0]["sharpe"]),
                "test_cand_sharpe": float(test_compare_df.iloc[1]["sharpe"]),
                "test_sharpe_excess": oos_sharpe_excess,
            }
        ]
    )
    summary_path = run_dir / "CVAR贝叶斯_摘要.csv"
    summary_df.to_csv(summary_path, index=False)

    wfo_folds_payload = [
        {
            "fold_id": int(f["fold_id"]),
            "train_start": str(pd.Timestamp(f["train_start"]).date()),
            "train_end": str(pd.Timestamp(f["train_end"]).date()),
            "test_start": str(pd.Timestamp(f["test_start"]).date()),
            "test_end": str(pd.Timestamp(f["test_end"]).date()),
            "train_days": int(f["train_days"]),
            "test_days": int(f["test_days"]),
        }
        for f in wfo_folds
    ]

    best_params_payload = {
        "mode": "wfo_bayes_hybrid_cvar_rp",
        "train_start": str(train_start.date()),
        "train_end": str(train_end.date()),
        "test_start": str(test_start.date()),
        "test_end": str(test_end.date()),
        "fixed_max_weight": FIXED_MAX_WEIGHT,
        "objective_config": {
            "obj_std_penalty": float(args.obj_std_penalty),
            "obj_worst_penalty": float(args.obj_worst_penalty),
            "obj_sharpe_floor": float(args.obj_sharpe_floor),
            "rp_anchor_lambda": float(args.rp_anchor_lambda),
            "obj_turnover_penalty": float(args.obj_turnover_penalty),
            "wfo_mean_excess_floor": float(args.wfo_mean_excess_floor),
            "wfo_worst_excess_floor": float(args.wfo_worst_excess_floor),
            "wfo_excess_penalty": float(args.wfo_excess_penalty),
        },
        "pruner_config": {
            "pruner": str(args.pruner),
            "pruner_startup_trials": int(args.pruner_startup_trials),
            "pruner_warmup_steps": int(args.pruner_warmup_steps),
        },
        "cvar_methods": cvar_methods,
        "cvar_method_compare_file": str(method_cmp_path),
        "wfo_folds": wfo_folds_payload,
        "optimizer": "hybrid_cvar_rp",
        "best_params": best_params,
        "selected_from": selected_from,
        "wfo_best_objective": _safe_float(best_row.get("objective")),
        "wfo_best_mean_sharpe": _safe_float(best_row.get("wfo_mean_sharpe")),
        "wfo_best_std_sharpe": _safe_float(best_row.get("wfo_std_sharpe")),
        "wfo_best_min_sharpe": _safe_float(best_row.get("wfo_min_sharpe")),
        "wfo_best_mean_excess": _safe_float(best_row.get("wfo_mean_sharpe_excess")),
        "wfo_best_min_excess": _safe_float(best_row.get("wfo_min_sharpe_excess")),
        "wfo_best_avg_active_share_to_rp": _safe_float(best_row.get("avg_active_share_to_rp")),
        "test_sharpe_excess_vs_rp": oos_sharpe_excess,
    }
    best_path = run_dir / "CVAR贝叶斯_best_params.json"
    with best_path.open("w", encoding="utf-8") as f:
        json.dump(best_params_payload, f, ensure_ascii=False, indent=2)

    if args.export_nav:
        nav_map = {
            "best_candidate_train": cand_train_nav,
            "RP_baseline_test": rp_test_nav,
            "best_candidate_test": cand_test_nav,
        }
        if rp_train_nav is not None:
            nav_map["RP_baseline_train"] = rp_train_nav
        export_nav(nav_map, output_dir=run_dir)

    plot_paths: List[str] = []
    if args.export_plots:
        rp_train_row = train_compare_df[train_compare_df["strategy"] == "RP_baseline_train"]
        ref_row = rp_train_row.iloc[0] if not rp_train_row.empty else train_compare_df.iloc[0]
        rp_plot_ref = {k: float(v) for k, v in ref_row.items() if k != "strategy"}
        plot_paths = generate_cvar_bayes_plots(
            trials_df=trials_df,
            rp_ref=rp_plot_ref,
            output_dir=run_dir / "plots",
            top_k=10,
        )

    bt_plot_paths = {}
    if args.export_backtest_plots:
        bt_plot_paths = export_backtest_plots(
            {"RP_baseline": rp_test_nav, "best_candidate": cand_test_nav},
            output_dir=run_dir,
        )

    print("=" * 60)
    print("WFO 贝叶斯调参摘要")
    print(f"  base-train 区间: {train_start.date()} ~ {train_end.date()}")
    print(f"  test-range 区间: {test_start.date()} ~ {test_end.date()}")
    print(f"  WFO 折数: {len(wfo_folds)}")
    print(f"  trial 数: {len(trials_df)}")
    if "status" in trials_df.columns:
        print(f"  剪枝 trial 数: {int((trials_df['status'] == 'pruned').sum())}")
    print(f"  通过全部规则数: {int(pass_df.shape[0])}")
    print(f"  最优来源: {selected_from}")
    print(f"  最优 CVaR 方法: {best_params['cvar_method']}")
    print(f"  最优 WFO mean/std/min Sharpe: "
          f"{_safe_float(best_row.get('wfo_mean_sharpe')):.4f} / "
          f"{_safe_float(best_row.get('wfo_std_sharpe')):.4f} / "
          f"{_safe_float(best_row.get('wfo_min_sharpe')):.4f}")
    print(f"  最优 WFO mean/min Sharpe 超额(相对RP): "
          f"{_safe_float(best_row.get('wfo_mean_sharpe_excess')):.4f} / "
          f"{_safe_float(best_row.get('wfo_min_sharpe_excess')):.4f}")
    print(f"  测试集 Sharpe 超额 (cand - RP): {oos_sharpe_excess:.4f}")
    print(f"  Split 配置: {split_path}")
    print(f"  WFO 折配置: {wfo_path}")
    print(f"  Trials 明细: {trials_path}")
    print(f"  CVaR 方法对照: {method_cmp_path}")
    print(f"  Top20: {top20_path}")
    print(f"  训练集对比: {train_compare_path}")
    print(f"  测试集对比: {test_compare_path}")
    print(f"  摘要: {summary_path}")
    print(f"  最优参数: {best_path}")
    if args.export_plots:
        print(f"  调参图数量: {len(plot_paths)}")
        print(f"  调参图目录: {run_dir / 'plots'}")
    if bt_plot_paths:
        print(f"  回测净值图: {bt_plot_paths.get('nav', '')}")
        print(f"  回测回撤图: {bt_plot_paths.get('drawdown', '')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
