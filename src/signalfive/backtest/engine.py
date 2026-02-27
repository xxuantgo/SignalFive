# -*- coding: utf-8 -*-
"""
回测引擎封装（基于 bt）
======================
职责：
  1. 生成调仓日权重计划（等权 / 优化权重）
  2. 构建 bt.Strategy 并运行回测（含手续费）
  3. 计算净值与绩效指标（年化收益、Sharpe、最大回撤）
  4. 导出净值序列

赛题约束：
  - 周度调仓，每周首个交易日
  - 调仓日执行使用上一交易日信号（避免当日 close 信息泄露）
  - 至少 3 只 ETF，单只 ≤ 35%
  - 手续费万 2.5（已在 bt.Backtest 中设置）
  - 回测基于收盘价，起始日 2021-01-04
"""
import numpy as np
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import bt

def _ensure_writable_plot_cache_dirs() -> None:
    """为 matplotlib/fontconfig 预留可写缓存目录（需在导入 bt 前执行）。"""
    mpl_config_dir = Path(os.environ.get("MPLCONFIGDIR", "/tmp/signalfive_mplconfig"))
    font_cache_dir = Path(os.environ.get("XDG_CACHE_HOME", "/tmp/signalfive_cache"))
    try:
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        font_cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(font_cache_dir))


_ensure_writable_plot_cache_dirs()


from signalfive.config import (
    BACKTEST_START, TOP_N, MAX_SINGLE_WEIGHT, MIN_HOLDINGS,
    COV_LOOKBACK, SHRINKAGE_FACTOR, CVAR_ALPHA, CVAR_TURNOVER_LAMBDA, CVAR_METHOD, HYBRID_BETA,
    TRANSACTION_COST, OUTPUT_DIR,
)
from signalfive.data_loader.loader import get_rebalance_dates_from_start
from signalfive.portfolio.optimizer import (
    compute_cov_from_returns,
    min_variance_weights,
    risk_parity_weights,
    cvar_weights,
    hybrid_cvar_rp_weights,
)


# ---------------------------------------------------------------------------
# 选股
# ---------------------------------------------------------------------------

def _select_top_n(composite_row: pd.Series, top_n: int) -> list:
    """从合成因子截面中选出得分最高的 Top N 只 ETF"""
    ranked = composite_row.dropna().sort_values(ascending=False)
    return ranked.head(top_n).index.tolist()


def _last_available_date(index: pd.DatetimeIndex,
                         dt: pd.Timestamp) -> Optional[pd.Timestamp]:
    """返回 index 中 <= dt 的最后一个日期。"""
    pos = index.searchsorted(dt, side="right") - 1
    if pos < 0:
        return None
    return index[pos]


def _previous_trading_date(index: pd.DatetimeIndex,
                           dt: pd.Timestamp) -> Optional[pd.Timestamp]:
    """返回 dt 的前一交易日。"""
    pos = index.searchsorted(dt, side="left") - 1
    if pos < 0:
        return None
    return index[pos]


# ---------------------------------------------------------------------------
# 调仓计划生成
# ---------------------------------------------------------------------------

def build_equal_weight_schedule(composite: pd.DataFrame,
                                price_matrix: pd.DataFrame,
                                top_n: int = None,
                                max_weight: float = None,
                                min_holdings: int = None,
                                rebal_start: Optional[pd.Timestamp] = None) -> dict:
    """
    生成等权调仓计划：{date: pd.Series(weights)}
    每个调仓日选 Top N，等权并满足单资产上限。
    """
    top_n = TOP_N if top_n is None else top_n
    max_weight = MAX_SINGLE_WEIGHT if max_weight is None else max_weight
    min_holdings = MIN_HOLDINGS if min_holdings is None else min_holdings

    schedule = {}
    start_dt = BACKTEST_START if rebal_start is None else rebal_start
    rebal_dates = get_rebalance_dates_from_start(price_matrix, start_dt)
    trading_dates = price_matrix.index
    signal_dates = composite.index

    for dt in rebal_dates:
        # 避免使用调仓日收盘价构建当日信号：改用前一交易日信号
        signal_dt = _previous_trading_date(trading_dates, dt)
        if signal_dt is None:
            continue
        signal_dt = _last_available_date(signal_dates, signal_dt)
        if signal_dt is None:
            continue

        picks = _select_top_n(composite.loc[signal_dt], top_n)
        if len(picks) < min_holdings:
            continue
        w = np.ones(len(picks)) / len(picks)
        w = np.minimum(w, max_weight)
        w = w / w.sum()
        schedule[dt] = pd.Series(w, index=picks)

    return schedule


def build_optimized_schedule(composite: pd.DataFrame,
                             price_matrix: pd.DataFrame,
                             optimizer: str = "risk_parity",
                             top_n: int = None,
                             max_weight: float = None,
                             min_holdings: int = None,
                             cov_window: int = None,
                             shrinkage: float = None,
                             cvar_alpha: float = None,
                             cvar_method: str = None,
                             turnover_lambda: float = None,
                             hybrid_beta: float = None,
                             rebal_start: Optional[pd.Timestamp] = None) -> dict:
    """
    生成优化权重调仓计划：{date: pd.Series(weights)}  根据composite, price_matrix 计算从 rebal_start 开始的调仓计划
    先选 Top N，再基于过去 cov_window 日窗口做优化。
    支持:
      - risk_parity: 风险平价（协方差）
      - min_variance: 最小方差（协方差）
      - cvar: 最小 CVaR（历史收益场景）
      - hybrid_cvar_rp: CVaR 与风险平价凸组合
    """
    top_n = TOP_N if top_n is None else top_n
    max_weight = MAX_SINGLE_WEIGHT if max_weight is None else max_weight
    min_holdings = MIN_HOLDINGS if min_holdings is None else min_holdings
    cov_window = COV_LOOKBACK if cov_window is None else cov_window
    shrinkage = SHRINKAGE_FACTOR if shrinkage is None else shrinkage
    cvar_alpha = CVAR_ALPHA if cvar_alpha is None else cvar_alpha
    cvar_method = CVAR_METHOD if cvar_method is None else cvar_method
    turnover_lambda = CVAR_TURNOVER_LAMBDA if turnover_lambda is None else turnover_lambda
    hybrid_beta = HYBRID_BETA if hybrid_beta is None else hybrid_beta

    schedule = {}
    start_dt = BACKTEST_START if rebal_start is None else rebal_start
    rebal_dates = get_rebalance_dates_from_start(price_matrix, start_dt)
    returns = price_matrix.pct_change()
    trading_dates = price_matrix.index
    signal_dates = composite.index

    optimizer = optimizer.lower().strip()
    valid_optimizers = {"risk_parity", "min_variance", "cvar", "hybrid_cvar_rp"}
    if optimizer not in valid_optimizers:
        raise ValueError(f"未知优化器: {optimizer}，可选: {sorted(valid_optimizers)}")

    prev_weights = None
    for dt in rebal_dates:
        # 避免使用调仓日收盘价构建当日信号：改用前一交易日信号
        signal_dt = _previous_trading_date(trading_dates, dt)
        if signal_dt is None:
            continue
        signal_dt = _last_available_date(signal_dates, signal_dt)
        if signal_dt is None:
            continue

        picks = _select_top_n(composite.loc[signal_dt], top_n)
        if len(picks) < min_holdings:
            continue

        # 收益窗口严格截止到信号日，避免使用调仓日收盘收益
        ret_win = returns.loc[:signal_dt].iloc[-cov_window:]
        ret_win = ret_win[picks].dropna(how="all", axis=1)
        if ret_win.shape[0] < max(20, min_holdings):
            continue

        if optimizer == "cvar":
            weights = cvar_weights(
                ret_win,
                alpha=cvar_alpha,
                max_weight=max_weight,
                min_holdings=min_holdings,
                prev_weights=prev_weights,
                turnover_lambda=turnover_lambda,
                cvar_method=cvar_method,
            )
        elif optimizer == "hybrid_cvar_rp":
            weights = hybrid_cvar_rp_weights(
                ret_win,
                beta=hybrid_beta,
                alpha=cvar_alpha,
                max_weight=max_weight,
                min_holdings=min_holdings,
                shrinkage=shrinkage,
                prev_weights=prev_weights,
                turnover_lambda=turnover_lambda,
                cvar_method=cvar_method,
            )
        else:
            cov = compute_cov_from_returns(ret_win, shrinkage)
            opt_func = risk_parity_weights if optimizer == "risk_parity" else min_variance_weights
            weights = opt_func(cov, max_weight=max_weight, min_holdings=min_holdings)

        weights = weights.dropna()
        weights = weights[weights > 0]
        if len(weights) < min_holdings:
            continue
        schedule[dt] = weights
        prev_weights = weights

    return schedule


# ---------------------------------------------------------------------------
# bt 自定义 Algo：按调仓计划设定权重
# ---------------------------------------------------------------------------

class WeighFromSchedule(bt.Algo if bt is not None else object):
    """
    仅在调仓日设定 temp['weights']。
    非调仓日不设定 → Rebalance() 检测到无 weights 时自动跳过 → 组合按市价漂移。
    """
    def __init__(self, schedule: dict):
        if bt is not None:
            super().__init__()
        self.schedule = schedule

    def __call__(self, target):
        if target.now in self.schedule:
            target.temp['weights'] = self.schedule[target.now].to_dict()
        return True  # 始终放行；无 weights 时 Rebalance 自动跳过


# ---------------------------------------------------------------------------
# 回测构建与运行
# ---------------------------------------------------------------------------

def _commission_fn(quantity, price):
    """手续费函数：成交额 × 万 2.5，买卖各收一次"""
    return abs(quantity) * price * TRANSACTION_COST


def run_backtests(price_matrix: pd.DataFrame,
                  schedules: dict):
    """
    构建多个策略并批量回测。

    Parameters
    ----------
    price_matrix : pd.DataFrame
        收盘价矩阵 (date × sec)，DatetimeIndex
    schedules : dict[str, dict]
        {策略名: {date: pd.Series(weights)}} 调仓计划

    Returns
    -------
    bt.Result
        回测结果对象，可用 .prices / .stats 访问
    """
    if bt is None:
        raise ModuleNotFoundError("缺少依赖包 `bt`，请先安装后再运行回测。")

    tests = []
    for name, schedule in schedules.items():
        rebal_dates = sorted(schedule.keys())
        if not rebal_dates:
            print(f"  警告: 策略 {name} 无调仓日，跳过")
            continue

        strat = bt.Strategy(name, [
            bt.algos.RunOnDate(*rebal_dates),   # 仅在调仓日触发
            WeighFromSchedule(schedule),         # 设定目标权重
            bt.algos.Rebalance(),                # 按目标权重调仓
        ])

        test = bt.Backtest(
            strat,
            price_matrix,
            commissions=_commission_fn,          # 赛题要求：万 2.5
            integer_positions=False,             # ETF 允许小数份额
        )
        tests.append(test)

    if not tests:
        raise ValueError("没有有效的策略可供回测！")

    return bt.run(*tests)


# ---------------------------------------------------------------------------
# 绩效指标
# ---------------------------------------------------------------------------

def calc_performance(nav: pd.Series, ann_factor: int = 252) -> dict:
    """
    计算策略绩效指标（赛题要求）。

    Parameters
    ----------
    nav : pd.Series
        归一化净值序列（起始 = 1.0）
    ann_factor : int
        年化因子（交易日数）

    Returns
    -------
    dict
        annual_return, annual_vol, sharpe, max_drawdown, total_return, calmar
    """
    returns = nav.pct_change().dropna()

    total_days = (nav.index[-1] - nav.index[0]).days
    total_years = total_days / 365.25
    total_return = nav.iloc[-1] / nav.iloc[0] - 1

    # 年化收益（复利）
    ann_return = (1 + total_return) ** (1 / total_years) - 1 if total_years > 0 else 0.0

    # 年化波动率
    ann_vol = returns.std() * np.sqrt(ann_factor)

    # 年化夏普比率（无风险利率 0%）
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    # 最大回撤
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax
    max_dd = drawdown.min()

    # Calmar 比率
    calmar = ann_return / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "annual_return": ann_return,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "total_return": total_return,
        "calmar": calmar,
    }


def extract_nav(result,
                start_date: str = None) -> dict:
    """
    从 bt 结果提取各策略的归一化净值（从回测起始日归一到 1.0）。

    Returns
    -------
    dict[str, pd.Series]
        {策略名: 净值序列}
    """
    start_date = pd.Timestamp(start_date or BACKTEST_START)
    navs = {}
    for col in result.prices.columns: # 获取价格数据
        full_nav = result.prices[col] # Name: 等权组合, Length: 1455, dtype: float64
        # 截取回测起始日之后
        nav = full_nav.loc[full_nav.index >= start_date].copy() # # Name: 等权组合, Length: 1168, dtype: float64
        if len(nav) == 0:
            continue
        # 归一化到 1.0
        nav = nav / nav.iloc[0]
        navs[col] = nav
    return navs


def export_nav(navs: dict, output_dir=None):
    """导出净值序列到 CSV"""
    output_dir = output_dir or OUTPUT_DIR
    for name, nav in navs.items():
        path = output_dir / f"{name}_净值序列.csv"
        nav_df = nav.to_frame(name="nav")
        nav_df.index.name = "date"
        nav_df.to_csv(path)
        print(f"  {name} 净值已保存: {path}")


def export_backtest_plots(navs: dict, output_dir=None) -> dict:
    """
    导出回测可视化图：
      1) 各策略净值曲线 -- 净值曲线展示了投资策略在回测期间的累计收益情况(盈利能力), 它将每个策略的净值随时间的变化绘制成曲线，使我们能够直观地看到策略的盈利能力和增长趋势。
      2) 各策略回撤曲线 -- 回撤曲线展示了投资策略面临的最大损失情况(风险水平), 回撤是指净值从前期高点下跌的幅度，

    Returns
    -------
    dict[str, pathlib.Path]
        {"nav": 净值图路径, "drawdown": 回撤图路径}
    """
    if not navs:
        print("  警告: 无净值数据，跳过回测图绘制。")
        return {}

    output_dir = Path(output_dir or OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 在受限环境中为 matplotlib/fontconfig 提供可写缓存目录
    mpl_config_dir = output_dir / ".mplconfig"
    font_cache_dir = output_dir / ".cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    font_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(font_cache_dir))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ModuleNotFoundError:
        print("  警告: 环境未安装 matplotlib，跳过回测图绘制。")
        return {}

    # 自动选择可用中文字体，避免中文图例/标题乱码
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    cjk_candidates = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Songti SC",
        "Heiti SC",
        "Heiti TC",
        "Arial Unicode MS",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "WenQuanYi Zen Hei",
    ]
    selected_font = next((f for f in cjk_candidates if f in available_fonts), None)
    if selected_font is not None:
        matplotlib.rcParams["font.sans-serif"] = [selected_font, "DejaVu Sans"]
    else:
        print("  警告: 未检测到可用中文字体，图中中文可能无法正常显示。")
    matplotlib.rcParams["axes.unicode_minus"] = False

    nav_df = pd.concat(navs, axis=1).sort_index()
    nav_df = nav_df.ffill().dropna(how="all")
    if nav_df.empty:
        print("  警告: 净值序列为空，跳过回测图绘制。")
        return {}

    # 净值曲线
    fig, ax = plt.subplots(figsize=(12, 6), dpi=120)
    for name in nav_df.columns:
        ax.plot(nav_df.index, nav_df[name], linewidth=1.8, label=str(name))
    ax.set_title("回测净值曲线")
    ax.set_xlabel("日期")
    ax.set_ylabel("净值")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    nav_name = f"回测净值曲线_{ts}.png"
    nav_path = output_dir / nav_name
    fig.savefig(nav_path, bbox_inches="tight")
    plt.close(fig)

    # 回撤曲线
    drawdown_df = nav_df.div(nav_df.cummax()).sub(1.0)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=120)
    for name in drawdown_df.columns:
        ax.plot(drawdown_df.index, drawdown_df[name], linewidth=1.8, label=str(name))
    ax.set_title("回测回撤曲线")
    ax.set_xlabel("日期")
    ax.set_ylabel("回撤")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    dd_name = f"回测回撤曲线_{ts}.png"
    dd_path = output_dir / dd_name
    fig.savefig(dd_path, bbox_inches="tight")
    plt.close(fig)

    return {"nav": nav_path, "drawdown": dd_path}


__all__ = [
    "build_equal_weight_schedule",
    "build_optimized_schedule",
    "run_backtests",
    "calc_performance",
    "extract_nav",
    "export_nav",
    "export_backtest_plots",
]
