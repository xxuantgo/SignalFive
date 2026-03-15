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

# ========== 数据版本配置 ==========
# 数据版本管理：支持多版本数据存储和灵活切换
# 数据目录结构：
#   data/
#   ├── versions/              # 各版本数据存储
#   │   ├── v20251030/         # 版本1：截止到2025-10-30
#   │   │   ├── price.csv
#   │   │   └── macro.csv
#   │   └── v20260313/         # 版本2：截止到2026-03-13（增量更新）
#   │       ├── price.csv
#   │       └── macro.csv
#   ├── current/               # 当前使用版本（软链接或复制）
#   │   ├── price.csv
#   │   └── macro.csv
#   └── raw/                   # 原始附件文件（保留）
#
# 使用方式：
#   1. 设置 DATA_VERSION = "v20251030" 或 "v20260313" 使用特定版本
#   2. 设置 DATA_VERSION = "current" 使用当前版本
#   3. 设置 DATA_VERSION = "auto" 自动选择最新版本（默认）

DATA_VERSION = "auto"  # 可选: "auto", "current", "v20251030", "v20260313" 等

# 数据版本目录
VERSIONS_DIR = DATA_DIR / "versions"
CURRENT_DIR = DATA_DIR / "current"
RAW_DIR = DATA_DIR / "raw"


def resolve_data_paths(version: str = "auto") -> tuple[Path, Path]:
    """
    解析数据文件路径
    
    Args:
        version: 数据版本标识
            - "auto": 自动选择最新版本（versions目录中日期最大的版本）
            - "current": 使用 current/ 目录下的数据
            - "vYYYYMMDD": 使用特定版本目录
    
    Returns:
        (price_file, macro_file) 的 Path 元组
    """
    if version == "current":
        price_file = CURRENT_DIR / "price.csv"
        macro_file = CURRENT_DIR / "macro.csv"
    elif version.startswith("v") and (VERSIONS_DIR / version).exists():
        version_dir = VERSIONS_DIR / version
        price_file = version_dir / "price.csv"
        macro_file = version_dir / "macro.csv"
    elif version == "auto":
        # 自动选择最新版本：扫描 versions 目录，选择日期最大的版本
        version_dirs = [d for d in VERSIONS_DIR.iterdir() if d.is_dir() and d.name.startswith("v")]
        if not version_dirs:
            # 回退到原始数据文件
            price_file = DATA_DIR / "附件2 ETF日频量价数据（开盘、收盘、高、低、成交量、成交额）.csv"
            macro_file = DATA_DIR / "附件3 高频经济指标（信用利差、期限利差、汇率等）.csv"
        else:
            # 按版本名称排序，取最后一个（最新的）
            latest_version = sorted(version_dirs, key=lambda x: x.name)[-1]
            price_file = latest_version / "price.csv"
            macro_file = latest_version / "macro.csv"
    else:
        # 默认回退到原始数据文件
        price_file = DATA_DIR / "附件2 ETF日频量价数据（开盘、收盘、高、低、成交量、成交额）.csv"
        macro_file = DATA_DIR / "附件3 高频经济指标（信用利差、期限利差、汇率等）.csv"
    
    return price_file, macro_file


# 动态解析当前版本的数据路径
PRICE_FILE, MACRO_FILE = resolve_data_paths(DATA_VERSION)

# 原始数据文件（保留向后兼容）
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
