# -*- coding: utf-8 -*-
"""
主流程脚本（提交版，纯配置驱动）
================================
步骤：
 1) 加载数据
 2) 计算因子并预处理
 3) 单因子测试，保存报告
 4) 使用 base.py 内置有效因子与合成方法
 5) 使用 base.py 内置 Regime 参数
 6) 生成等权 / 组合优化权重调仓计划
 7) 运行回测，输出绩效摘要与净值序列
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


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = OUTPUT_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"本次运行输出目录: {run_dir}\n")

    # 固定提交参数（全部来自 base.py）
    default_params = dict(DEFAULT_PORTFOLIO_PARAMS)
    top_n_use = int(default_params.get("top_n", TOP_N))
    cov_window_use = int(default_params.get("cov_window", COV_LOOKBACK))
    cvar_alpha_use = float(default_params.get("cvar_alpha", CVAR_ALPHA))
    cvar_method_use = str(default_params.get("cvar_method", CVAR_METHOD))
    turnover_lambda_use = float(default_params.get("turnover_lambda", CVAR_TURNOVER_LAMBDA))
    hybrid_beta_use = float(default_params.get("hybrid_beta", HYBRID_BETA))
    max_weight_use = float(default_params.get("max_weight", MAX_SINGLE_WEIGHT))

    if max_weight_use > MAX_SINGLE_WEIGHT:
        print(f"  警告: max_weight={max_weight_use:.4f} 超过上限 {MAX_SINGLE_WEIGHT:.4f}，已截断。")
        max_weight_use = MAX_SINGLE_WEIGHT
    if top_n_use < MIN_HOLDINGS:
        raise ValueError(f"top_n={top_n_use} 小于最小持仓数 MIN_HOLDINGS={MIN_HOLDINGS}")
    if not (0.0 < cvar_alpha_use < 1.0):
        raise ValueError(f"cvar_alpha 必须在 (0,1) 内，当前为 {cvar_alpha_use}")
    if cov_window_use <= 0:
        raise ValueError(f"cov_window 必须 > 0，当前为 {cov_window_use}")
    if turnover_lambda_use < 0:
        raise ValueError(f"turnover_lambda 不能为负，当前为 {turnover_lambda_use}")
    if not (0.0 <= hybrid_beta_use <= 1.0):
        raise ValueError(f"hybrid_beta 必须在 [0,1] 内，当前为 {hybrid_beta_use}")
    if max_weight_use <= 0:
        raise ValueError(f"max_weight 必须 > 0，当前为 {max_weight_use}")
    if not (0.0 <= float(REGIME_RELAX_GAMMA) <= 1.0):
        raise ValueError(f"REGIME_RELAX_GAMMA 必须在 [0,1] 内，当前为 {REGIME_RELAX_GAMMA}")
    if float(REGIME_MAX_STEP) < 0:
        raise ValueError(f"REGIME_MAX_STEP 不能为负，当前为 {REGIME_MAX_STEP}")

    print("参数来源: base.py（无命令行覆盖）")
    print(
        "固定优化参数: "
        f"top_n={top_n_use}, "
        f"cov_window={cov_window_use}, "
        f"alpha={cvar_alpha_use:.6f}, "
        f"method={cvar_method_use}, "
        f"turnover_lambda={turnover_lambda_use:.6g}, "
        f"hybrid_beta={hybrid_beta_use:.4f}, "
        f"max_weight={max_weight_use:.4f}"
    )
    print(
        "固定 Regime 参数: "
        f"mode={REGIME_MODE}, "
        f"relax_gamma={float(REGIME_RELAX_GAMMA):.3f}, "
        f"stress_threshold={REGIME_STRESS_THRESHOLD}, "
        f"max_daily_step={REGIME_MAX_STEP}"
    )
    print(f"固定合成方法: {COMBINE_METHOD}")
    print(f"固定有效因子({len(DEFAULT_EFFECTIVE_FACTORS)}): {list(DEFAULT_EFFECTIVE_FACTORS)}")

    # ====================================================================
    # 1) 数据加载
    # ====================================================================
    print("=" * 60)
    print("Step 1: 加载数据")
    data = load_all()
    close_matrix = data["close_matrix"]
    print(f"  收盘价矩阵: {close_matrix.shape}")

    # ====================================================================
    # 2) 因子计算与预处理
    # ====================================================================
    print("=" * 60)
    print("Step 2: 计算因子并预处理")
    panel_wide, macro_df = compute_factors(data["aligned"])
    processed = prepare_factor_matrices(panel_wide, method="rank")
    print(f"  面板因子数: {len(processed)}")

    # ====================================================================
    # 3) 单因子测试
    # ====================================================================
    print("=" * 60)
    print("Step 3: 单因子 Rank IC 测试")
    _, ic_series_dict = test_all_factors(processed, close_matrix)

    # 严格防前视偏差：IC[t] 依赖 t->t+fwd 收益，需向后平移 fwd 才可在实盘时点观测
    fwd_shift = FORWARD_RETURN_PERIODS[0]
    shifted_ic = {name: ic.shift(fwd_shift) for name, ic in ic_series_dict.items()}
    train_cutoff = pd.Timestamp(BACKTEST_START) - pd.Timedelta(days=1)
    auto_effective, prestart_summary = select_effective_factors_from_ic(
        shifted_ic, cutoff=str(train_cutoff.date())
    )
    prestart_path = run_dir / "单因子测试结果.csv"
    prestart_summary.to_csv(prestart_path, index=False)
    print(f"  回测前样本测试已保存: {prestart_path}")
    print(f"  自动筛选有效因子 ({len(auto_effective)}): {auto_effective}")

    effective = list(DEFAULT_EFFECTIVE_FACTORS)
    missing = [f for f in effective if f not in processed]
    if missing:
        raise ValueError(f"base.py 固定因子中存在未计算因子: {missing}")
    print(f"  使用 base.py 固定有效因子 ({len(effective)}): {effective}")

    effective_path = run_dir / "有效因子.csv"
    pd.DataFrame({"因子": effective}).to_csv(effective_path, index=False, encoding="utf-8-sig")
    print(f"  有效因子列表已保存: {effective_path}")

    # ====================================================================
    # 4) 因子合成
    # ====================================================================
    print("=" * 60)
    print("Step 4: 因子合成")
    composite = combine_factors(processed, ic_series_dict, effective, method=COMBINE_METHOD)
    export_composite_factor(composite, output_path=str(run_dir / "合成因子序列.csv"))

    # ====================================================================
    # 5) 宏观 Regime 仓位信号
    # ====================================================================
    print("=" * 60)
    print("Step 5: 宏观 Regime 仓位调节")
    if REGIME_MODE == "off":
        position_scale = pd.Series(1.0, index=macro_df.index, name="position_scale")
        print("  Regime 已关闭：position_scale 恒为 1.0")
    else:
        stress_threshold_use = (
            float(REGIME_STRESS_THRESHOLD)
            if REGIME_STRESS_THRESHOLD is not None
            else None
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
        print(
            "  Regime参数: "
            f"relax_gamma={float(REGIME_RELAX_GAMMA):.3f}, "
            f"stress_threshold={stress_threshold_use}, "
            f"max_daily_step={max_step_use}"
        )
        regime_stats = summarize_regime(position_scale, start_date=BACKTEST_START)
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

    # ====================================================================
    # 6) 调仓计划
    # ====================================================================
    print("=" * 60)
    print("Step 6: 生成调仓计划")
    eq_schedule = build_equal_weight_schedule(
        composite,
        close_matrix,
        top_n=top_n_use, # 等权调仓时，每次选 Top N 只 ETF    
        max_weight=max_weight_use,
        min_holdings=MIN_HOLDINGS,
    )
    print(f"  等权调仓日数: {len(eq_schedule)}")

    opt_schedule = build_optimized_schedule( # 生成优化权重调仓计划：{date: pd.Series(weights)} 先选 Top N，再用过去 cov_window 日的协方差做优化。
        composite, close_matrix,
        optimizer=OPTIMIZER_METHOD,
        top_n=top_n_use,
        max_weight=max_weight_use,
        min_holdings=MIN_HOLDINGS,
        cov_window=cov_window_use,
        cvar_alpha=cvar_alpha_use,
        cvar_method=cvar_method_use,
        turnover_lambda=turnover_lambda_use,
        hybrid_beta=hybrid_beta_use,
    )
    print(f"  优化器: {OPTIMIZER_METHOD}")
    print(f"  TopN: {top_n_use}")
    print(
        "  优化参数: "
        f"cov_window={cov_window_use}, "
        f"alpha={cvar_alpha_use:.6f}, "
        f"method={cvar_method_use}, "
        f"turnover_lambda={turnover_lambda_use:.6g}, "
        f"hybrid_beta={hybrid_beta_use:.4f}, "
        f"max_weight={max_weight_use:.4f}"
    )
    print(f"  优化调仓日数: {len(opt_schedule)}")

    # 应用宏观仓位调节
    eq_schedule = apply_position_scale(eq_schedule, position_scale) # 将仓位缩放系数应用到调仓计划上。
    opt_schedule = apply_position_scale(opt_schedule, position_scale)
    if REGIME_MODE == "off":
        print("  Regime 已关闭：未进行额外仓位缩放")
    else:
        print("  已应用宏观 Regime 仓位调节")

    # ====================================================================
    # 7) 回测
    # ====================================================================
    print("=" * 60)
    print("Step 7: 运行回测")
    schedules = {
        "等权组合": eq_schedule,
        "优化后组合": opt_schedule,
    }
    res = run_backtests(close_matrix, schedules)

    navs = extract_nav(res)
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
