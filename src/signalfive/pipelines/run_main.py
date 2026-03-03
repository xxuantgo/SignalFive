"""
主流程脚本_1 -- “等权组合”与“优化组合(固定优化参数)”结果
================================
项目主流程（5步）：
  1) 因子设计和测试
  2) 因子合成
  3) 组合优化
  4) 宏观仓位调整
  5) 模型调参（本脚本不执行）

说明：
  - run_main 仅使用 base.py 固定参数，不做命令行覆盖和在线调参。-- 读取并校验组合参数（top_n/cov_window/cvar_alpha/cvar_method/turnover_lambda/hybrid_beta/max_weight）。
  - 回测区间口径：BACKTEST_START(2021-01-04) ~ 2025-10-30。
  - 一次运行同时产出“等权组合”与“优化组合(固定优化参数)”两套基线回测结果。
  - 关键防泄漏约束：
  - 调仓日为每周首个交易日；
  - 调仓信号使用前一交易日；
  - 优化窗口仅使用 <= 信号日 的历史收益。
  1) 创建时间戳输出目录，统一保存本次运行的中间文件与结果。
  2) 加载量价与宏观数据，构建 close_matrix。
  3) 计算因子并做截面预处理（rank）。
  4) 单因子 Rank IC 测试（IC 向后平移，避免前视偏差），保存测试报告。
  5) 使用 base.py 固定有效因子 + 固定合成方法，生成 composite 截面得分。
  6) 计算 Regime 仓位系数（或关闭 Regime 时恒为 1.0），并保存统计。
  7) 生成两类调仓计划：
     - 等权计划：调仓日选 TopN，等权分配并满足单票上限/最少持仓约束；
     - 优化计划：调仓日先选 TopN，再用历史窗口收益进行风险优化（RP/MV/CVaR/Hybrid）。
  8) 将 Regime 仓位系数应用到两类调仓计划（总仓位可小于 1，剩余为现金）。
  9) 批量运行回测，导出净值、图表与绩效汇总。
  
"""


from datetime import datetime

import pandas as pd

from signalfive.config import (
    OUTPUT_DIR,
    TOP_N,
    MAX_SINGLE_WEIGHT,
    MIN_HOLDINGS,
    BACKTEST_START,
    FORWARD_RETURN_PERIODS,
    COV_LOOKBACK,
    CVAR_ALPHA,
    CVAR_METHOD,
    CVAR_TURNOVER_LAMBDA,
    HYBRID_BETA,
    OPTIMIZER_METHOD,
    COMBINE_METHOD,
    DEFAULT_PORTFOLIO_PARAMS,
    DEFAULT_EFFECTIVE_FACTORS,
    REGIME_MODE,
    REGIME_RELAX_GAMMA,
    REGIME_STRESS_THRESHOLD,
    REGIME_MAX_STEP,
)
from signalfive.data_loader.loader import load_all
from signalfive.factors.calc import compute_factors, prepare_factor_matrices
from signalfive.factors.testing import test_all_factors, select_effective_factors_from_ic
from signalfive.factors.combine import combine_factors, export_composite_factor
from signalfive.backtest.engine import (
    build_equal_weight_schedule,
    build_optimized_schedule,
    run_backtests,
    calc_performance,
    extract_nav,
    export_nav,
    export_backtest_plots,
)
from signalfive.portfolio.regime import calc_position_scale, apply_position_scale, summarize_regime


STRATEGY_EQ_NAME = "等权组合"
STRATEGY_OPT_FIXED_NAME = "优化组合(固定优化参数)"
BACKTEST_END = "2025-10-30"


def _resolve_fixed_run_configs() -> tuple[dict, dict, dict, dict]:
    """按流程分组整理 run_main 使用的固定参数。"""
    default_params = dict(DEFAULT_PORTFOLIO_PARAMS)

    factor_cfg = {
        "combine_method": str(COMBINE_METHOD),
        "effective_factors": list(DEFAULT_EFFECTIVE_FACTORS),
        "forward_shift": int(FORWARD_RETURN_PERIODS[0]),
    }

    optimization_cfg = {
        "optimizer": str(OPTIMIZER_METHOD),
        "top_n": int(default_params.get("top_n", TOP_N)),
        "cov_window": int(default_params.get("cov_window", COV_LOOKBACK)),
        "cvar_alpha": float(default_params.get("cvar_alpha", CVAR_ALPHA)),
        "cvar_method": str(default_params.get("cvar_method", CVAR_METHOD)),
        "turnover_lambda": float(default_params.get("turnover_lambda", CVAR_TURNOVER_LAMBDA)),
        "hybrid_beta": float(default_params.get("hybrid_beta", HYBRID_BETA)),
        "max_weight": float(default_params.get("max_weight", MAX_SINGLE_WEIGHT)),
    }

    regime_cfg = {
        "mode": str(REGIME_MODE),
        "relax_gamma": float(REGIME_RELAX_GAMMA),
        "stress_threshold": (
            float(REGIME_STRESS_THRESHOLD) if REGIME_STRESS_THRESHOLD is not None else None
        ),
        "max_daily_step": float(REGIME_MAX_STEP) if float(REGIME_MAX_STEP) > 0 else None,
    }

    backtest_cfg = {
        "start": str(BACKTEST_START),
        "end": str(BACKTEST_END),
    }

    _validate_fixed_configs(optimization_cfg=optimization_cfg, regime_cfg=regime_cfg)
    return factor_cfg, optimization_cfg, regime_cfg, backtest_cfg


def _validate_fixed_configs(optimization_cfg: dict, regime_cfg: dict) -> None:
    """固定参数合法性校验（run_main 不涉及模型调参参数）。"""
    max_weight = float(optimization_cfg["max_weight"])
    top_n = int(optimization_cfg["top_n"])
    cov_window = int(optimization_cfg["cov_window"])
    cvar_alpha = float(optimization_cfg["cvar_alpha"])
    turnover_lambda = float(optimization_cfg["turnover_lambda"])
    hybrid_beta = float(optimization_cfg["hybrid_beta"])
    relax_gamma = float(regime_cfg["relax_gamma"])
    max_daily_step = regime_cfg["max_daily_step"]

    if max_weight > MAX_SINGLE_WEIGHT:
        print(f"  警告: max_weight={max_weight:.4f} 超过上限 {MAX_SINGLE_WEIGHT:.4f}，已截断。")
        optimization_cfg["max_weight"] = float(MAX_SINGLE_WEIGHT)
        max_weight = float(MAX_SINGLE_WEIGHT)
    if top_n < MIN_HOLDINGS:
        raise ValueError(f"top_n={top_n} 小于最小持仓数 MIN_HOLDINGS={MIN_HOLDINGS}")
    if cov_window <= 0:
        raise ValueError(f"cov_window 必须 > 0，当前为 {cov_window}")
    if not (0.0 < cvar_alpha < 1.0):
        raise ValueError(f"cvar_alpha 必须在 (0,1) 内，当前为 {cvar_alpha}")
    if turnover_lambda < 0:
        raise ValueError(f"turnover_lambda 不能为负，当前为 {turnover_lambda}")
    if not (0.0 <= hybrid_beta <= 1.0):
        raise ValueError(f"hybrid_beta 必须在 [0,1] 内，当前为 {hybrid_beta}")
    if max_weight <= 0:
        raise ValueError(f"max_weight 必须 > 0，当前为 {max_weight}")
    if not (0.0 <= relax_gamma <= 1.0):
        raise ValueError(f"REGIME_RELAX_GAMMA 必须在 [0,1] 内，当前为 {relax_gamma}")
    if max_daily_step is not None and float(max_daily_step) < 0:
        raise ValueError(f"REGIME_MAX_STEP 不能为负，当前为 {max_daily_step}")


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f") # 20260227_212804_405372
    run_dir = OUTPUT_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"本次运行输出目录: {run_dir}\n")

    factor_cfg, optimization_cfg, regime_cfg, backtest_cfg = _resolve_fixed_run_configs()

    print("参数来源: base.py（固定参数，无命令行覆盖）")
    print("模型调参: run_main 不执行模型调参，直接使用固定优化参数。")
    print(
        "因子参数: "
        f"combine_method={factor_cfg['combine_method']}, "
        f"forward_shift={factor_cfg['forward_shift']}, "
        f"fixed_effective_factors={len(factor_cfg['effective_factors'])}"
    )
    print(
        "固定优化参数: "
        f"optimizer={optimization_cfg['optimizer']}, "
        f"top_n={optimization_cfg['top_n']}, "
        f"cov_window={optimization_cfg['cov_window']}, "
        f"cvar_alpha={optimization_cfg['cvar_alpha']:.6f}, "
        f"cvar_method={optimization_cfg['cvar_method']}, "
        f"turnover_lambda={optimization_cfg['turnover_lambda']:.6g}, "
        f"hybrid_beta={optimization_cfg['hybrid_beta']:.4f}, "
        f"max_weight={optimization_cfg['max_weight']:.4f}"
    )
    print(
        "Regime 参数: "
        f"mode={regime_cfg['mode']}, "
        f"relax_gamma={regime_cfg['relax_gamma']:.3f}, "
        f"stress_threshold={regime_cfg['stress_threshold']}, "
        f"max_daily_step={regime_cfg['max_daily_step']}"
    )

    # ====================================================================
    # Step 1) 因子设计和测试（含数据加载）
    # ====================================================================
    print("=" * 60)
    print("Step 1: 因子设计和测试（含数据加载）")
    data = load_all()
    close_matrix = data["close_matrix"]
    print(f"  收盘价矩阵: {close_matrix.shape}")

    panel_wide, macro_df = compute_factors(data["aligned"])
    processed = prepare_factor_matrices(panel_wide, method="rank")
    print(f"  面板因子数: {len(processed)}")

    _, ic_series_dict = test_all_factors(processed, close_matrix)

    # 严格防前视偏差：IC[t] 依赖 t->t+fwd 收益，需向后平移 fwd 才可在实盘时点观测
    shifted_ic = {
        name: ic.shift(int(factor_cfg["forward_shift"])) for name, ic in ic_series_dict.items()
    }
    train_cutoff = pd.Timestamp(backtest_cfg["start"]) - pd.Timedelta(days=1)
    auto_effective, prestart_summary = select_effective_factors_from_ic(
        shifted_ic, cutoff=str(train_cutoff.date())
    )
    prestart_path = run_dir / "单因子测试结果.csv"
    prestart_summary.to_csv(prestart_path, index=False)
    print(f"  回测前样本测试已保存: {prestart_path}")
    print(f"  自动筛选有效因子 ({len(auto_effective)}): {auto_effective}")

    effective = list(factor_cfg["effective_factors"])
    missing = [f for f in effective if f not in processed]
    if missing:
        raise ValueError(f"base.py 固定因子中存在未计算因子: {missing}")
    print(f"  使用 base.py 固定有效因子 ({len(effective)}): {effective}")

    effective_path = run_dir / "有效因子.csv"
    pd.DataFrame({"因子": effective}).to_csv(effective_path, index=False, encoding="utf-8-sig")
    print(f"  有效因子列表已保存: {effective_path}")

    # ====================================================================
    # Step 2) 因子合成
    # ====================================================================
    print("=" * 60)
    print("Step 2: 因子合成")
    composite = combine_factors(
        processed, ic_series_dict, effective, method=str(factor_cfg["combine_method"])
    )
    export_composite_factor(composite, output_path=str(run_dir / "合成因子序列.csv"))

    # ====================================================================
    # Step 3) 组合优化（固定参数）
    # ====================================================================
    print("=" * 60)
    print("Step 3: 组合优化（固定参数）")
    # 回测区间说明：本脚本回测起点为 BACKTEST_START(2021-01-04)，终点为 2025-10-30。
    eq_schedule = build_equal_weight_schedule(
        composite,
        close_matrix,
        top_n=int(optimization_cfg["top_n"]),
        max_weight=float(optimization_cfg["max_weight"]),
        min_holdings=MIN_HOLDINGS,
    )
    print(f"  等权调仓日数: {len(eq_schedule)}")

    # 回测区间说明：本脚本回测起点为 BACKTEST_START(2021-01-04)，终点为 2025-10-30。
    opt_schedule = build_optimized_schedule(
        composite,
        close_matrix,
        optimizer=str(optimization_cfg["optimizer"]),
        top_n=int(optimization_cfg["top_n"]),
        max_weight=float(optimization_cfg["max_weight"]),
        min_holdings=MIN_HOLDINGS,
        cov_window=int(optimization_cfg["cov_window"]),
        cvar_alpha=float(optimization_cfg["cvar_alpha"]),
        cvar_method=str(optimization_cfg["cvar_method"]),
        turnover_lambda=float(optimization_cfg["turnover_lambda"]),
        hybrid_beta=float(optimization_cfg["hybrid_beta"]),
    )
    print(f"  优化调仓日数: {len(opt_schedule)}")

    # ====================================================================
    # Step 4) 宏观仓位调整
    # ====================================================================
    print("=" * 60)
    print("Step 4: 宏观仓位调整")
    if str(regime_cfg["mode"]) == "off":
        position_scale = pd.Series(1.0, index=macro_df.index, name="position_scale")
        print("  Regime 已关闭：position_scale 恒为 1.0")
    else:
        position_scale = calc_position_scale(
            macro_df,
            smooth_window=5,
            relax_gamma=float(regime_cfg["relax_gamma"]),
            stress_threshold=regime_cfg["stress_threshold"],
            max_daily_step=regime_cfg["max_daily_step"],
            stress_factors=("F01", "F02", "F04"),
        )
        regime_stats = summarize_regime(position_scale, start_date=backtest_cfg["start"])
        print(
            "  Regime参数: "
            f"relax_gamma={float(regime_cfg['relax_gamma']):.3f}, "
            f"stress_threshold={regime_cfg['stress_threshold']}, "
            f"max_daily_step={regime_cfg['max_daily_step']}"
        )
        print("  仓位系数统计 (回测期):")
        print(f"    均值={regime_stats['mean']:.2%}, 中位数={regime_stats['median']:.2%}")
        print(f"    最小={regime_stats['min']:.2%}, 最大={regime_stats['max']:.2%}")
        print(
            f"    低仓(<50%): {regime_stats['pct_below_50']:.1%}, "
            f"中仓(50-80%): {regime_stats['pct_50_80']:.1%}, "
            f"高仓(>80%): {regime_stats['pct_above_80']:.1%}"
        )

    ps_df = position_scale.to_frame("position_scale")
    ps_df.index.name = "date"
    ps_path = run_dir / "宏观仓位系数.csv"
    ps_df.to_csv(ps_path)
    print(f"  仓位系数已保存: {ps_path}")

    eq_schedule = apply_position_scale(eq_schedule, position_scale)
    opt_schedule = apply_position_scale(opt_schedule, position_scale)
    if str(regime_cfg["mode"]) == "off":
        print("  Regime 已关闭：未进行额外仓位缩放")
    else:
        print("  已应用宏观 Regime 仓位调节")

    # ====================================================================
    # Step 5) 模型调参（run_main 不执行）
    # ====================================================================
    print("=" * 60)
    print("Step 5: 模型调参")
    print("  run_main 为固定参数基线流程：模型调参已关闭。")

    # ====================================================================
    # Step 6) 回测与结果导出
    # ====================================================================
    print("=" * 60)
    print("Step 6: 运行回测并导出结果")
    print(f"  回测区间: {backtest_cfg['start']} ~ {backtest_cfg['end']}")
    schedules = {
        STRATEGY_EQ_NAME: eq_schedule,
        STRATEGY_OPT_FIXED_NAME: opt_schedule,
    }
    res = run_backtests(close_matrix, schedules)

    navs = extract_nav(res, start_date=backtest_cfg["start"])
    export_nav(navs, output_dir=run_dir)
    plot_paths = export_backtest_plots(navs, output_dir=run_dir)
    if plot_paths:
        print(f"  回测净值图已保存: {plot_paths['nav']}")
        print(f"  回测回撤图已保存: {plot_paths['drawdown']}")

    print("\n" + "=" * 60)
    print("绩效摘要")
    print("=" * 60)
    perf_rows = []
    for name, nav in navs.items():
        perf = calc_performance(nav)
        perf_rows.append({"策略": name, **perf})
        print(f"\n【{name}】")
        print(f"  年化收益率  : {perf['annual_return']:>8.2%}")
        print(f"  年化波动率  : {perf['annual_vol']:>8.2%}")
        print(f"  夏普比率    : {perf['sharpe']:>8.3f}")
        print(f"  最大回撤    : {perf['max_drawdown']:>8.2%}")
        print(f"  Calmar比率  : {perf['calmar']:>8.3f}")
        print(f"  累计收益率  : {perf['total_return']:>8.2%}")

    perf_df = pd.DataFrame(perf_rows)
    perf_path = run_dir / "绩效汇总.csv"
    perf_df.to_csv(perf_path, index=False)
    print(f"\n绩效汇总已保存: {perf_path}")

    print("\n" + "=" * 60)
    print("全部完成！输出目录:", run_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
