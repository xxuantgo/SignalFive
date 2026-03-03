# -*- coding: utf-8 -*-
"""
全局配置参数
============
集中管理策略参数、文件路径、因子列表等，方便调参和复现。
"""
from pathlib import Path

# ========== 路径配置 ==========
# src/signalfive/config/base.py -> 项目根目录为上三级
PROJECT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 原始数据文件
PRICE_FILE = DATA_DIR / "附件2 ETF日频量价数据（开盘、收盘、高、低、成交量、成交额）.csv"
MACRO_FILE = DATA_DIR / "附件3 高频经济指标（信用利差、期限利差、汇率等）.csv"
PRODUCT_POOL_FILE = DATA_DIR / "附件1 28只非债券ETF产品池.xlsx"

# ========== 回测参数 ==========
BACKTEST_START = "2021-01-04"          # 回测起始日（赛题硬性要求）
RISK_FREE_RATE = 0.0                   # 无风险利率（赛题要求 0%）
TRANSACTION_COST = 2.5 / 10000         # 交易手续费 万分之2.5
REBALANCE_FREQ = "W"                   # 调仓频率：周度
TOP_N = 3                              # 要求每期选取 Top N 只 ETF（≥3）
MAX_SINGLE_WEIGHT = 0.35               # 单只 ETF 权重上限 35%
MIN_HOLDINGS = 3                       # 最少持仓数量

# ========== 因子配置 ==========
# 截面因子（用于 ETF 排序）
PANEL_FACTORS = [
    "A01", "A02", "A03", "A04", "A05", "A06",
    "B01", "B02", "B03", "B04", "B05", "B06",
    "C01", "C02", "C03", "C04",
    "D01", "D02", "D03", "D04",
    "E01", "E02",
    "G01", "G02", "G03",
    "H01",
    "K02", "K04",
    "M01", "M03",
    "N01",
    "T01",
]

# 附加的辅助因子（收益/波动率，用于因子构造但不直接纳入合成）
AUX_FACTORS = ["ret_1", "ret_5", "ret_20", "vol_5", "vol_20"]

# 宏观因子（用于 Regime 分析，不参与截面排序）
MACRO_FACTORS = ["F01", "F02", "F03", "F04", "F05", "F06"]

# ========== 因子测试参数 ==========
FORWARD_RETURN_PERIODS = [5]           # 前瞻收益期限(交易日)：5日≈一周持有期
IC_ROLLING_WINDOW = 60                 # 滚动 IC 计算窗口（交易日）
IC_MIN_OBS = 10                        # 截面最少有效观测数

# ========== 因子合成参数 ==========
# 合成方法：ic / icir / equal / icir_robust
# icir_robust: 在滚动ICIR基础上叠加
#   1) 回测前样本显著性质量分
#   2) 因子相关性惩罚（降低冗余因子权重）
#   3) 时间维度权重平滑（抑制权重抖动）
COMBINE_METHOD = "icir_robust"
COMBINE_ROLLING_WINDOW = 60            # 合成权重的滚动窗口
COMBINE_ROBUST_CORR_PENALTY = 0.8      # icir_robust 的相关性惩罚强度
COMBINE_ROBUST_TURNOVER_SMOOTH = 0.35  # icir_robust 的时间平滑强度(0,1]

# ========== 组合优化参数 ==========
COV_LOOKBACK = 180                      # 优化窗口（协方差或历史收益场景）
SHRINKAGE_FACTOR = 0.05                 # Ledoit-Wolf 收缩系数（简化版）
CVAR_ALPHA = 0.94                      # CVaR 置信水平（尾部比例 = 1 - alpha）
CVAR_TURNOVER_LAMBDA = 0.01            # CVaR 换手惩罚系数 λ（0 表示关闭）
CVAR_METHOD = "empirical"             # CVaR 估计方法：empirical / parametric / cornish_fisher
HYBRID_BETA = 0.10                     # 混合比例：0=纯CVaR, 1=纯风险平价

# 优化器选择：risk_parity / min_variance / cvar / hybrid_cvar_rp
OPTIMIZER_METHOD = "hybrid_cvar_rp"

# ========== 主流程默认参数（run_main） ==========
# 说明：由上方标量参数自动组装，避免同一参数多处手工维护
DEFAULT_PORTFOLIO_PARAMS = {
    "top_n": TOP_N,
    "cvar_alpha": CVAR_ALPHA,
    "cvar_method": CVAR_METHOD,
    "cov_window": COV_LOOKBACK,
    "max_weight": MAX_SINGLE_WEIGHT,
    "turnover_lambda": CVAR_TURNOVER_LAMBDA,
    "hybrid_beta": HYBRID_BETA,
}

DEFAULT_EFFECTIVE_FACTORS = (
    "A02",
    "A06",
    "B01",
    "B03",
    "B06",
    "C03",
    "D01",
    "D04",
    "K02",
    "K04",
    "M01",
    "M03",
    "T01",
)

# Regime 默认模式（rule_v2 参数）
REGIME_MODE = "rule"                   # rule / off
REGIME_RELAX_GAMMA = 0.40              # 向满仓混合比例
REGIME_STRESS_THRESHOLD = 0.8          # 压力门控阈值（基于 F01/F02/F04）
REGIME_MAX_STEP = 0.03                 # 仓位日度变化上限
