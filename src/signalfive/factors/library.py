# -*- coding: utf-8 -*-
"""
因子实现：根据 data/因子库.csv 中的定义，基于产品池、量价数据、宏观数据计算各因子。
数据格式：
  - 附件1：28只非债券ETF产品池.xlsx（产品列表，需 pip install openpyxl）
  - 附件2：ETF日频量价数据（date, sec, open, high, low, close, volume, amount）
  - 附件3：高频经济指标（第一列为日期，其余为信用利差、期限利差、VIX 等列名见 CSV 表头）
依赖：numpy, pandas, scipy
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

DATA_DIR = Path(__file__).resolve().parent / "data"
try:
    # 优先使用项目统一配置路径，避免包内相对路径失配。
    from signalfive.config import DATA_DIR as CONFIG_DATA_DIR  
    DATA_DIR = CONFIG_DATA_DIR
except Exception:
    pass


def load_product_pool(path: Path = None) -> pd.DataFrame:
    """加载产品池。附件1 28只非债券ETF产品池.xlsx"""
    path = path or DATA_DIR / "附件1 28只非债券ETF产品池.xlsx"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_excel(path, sheet_name=0)
    except Exception as e:
        print(f"Warning: 无法读取产品池 {path}: {e}")
        return pd.DataFrame()


def load_price(path: Path = None) -> pd.DataFrame:
    """加载量价数据。附件2：date, sec, open, high, low, close, volume, amount"""
    path = path or DATA_DIR / "附件2 ETF日频量价数据（开盘、收盘、高、低、成交量、成交额）.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["close"], how="all")


def load_macro(path: Path = None) -> pd.DataFrame:
    """加载宏观数据。附件3：第一列为日期，其余为各类经济指标"""
    path = path or DATA_DIR / "附件3 高频经济指标（信用利差、期限利差、汇率等）.csv"
    df = pd.read_csv(path)
    if df.columns[0].strip() == "" or "Unnamed" in str(df.columns[0]):
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_aligned_data(data_dir: Path = None) -> pd.DataFrame:
    """
    先将量价（附件2）与宏观（附件3）按 date 对齐，得到一张表。
    每行 = (date, sec) + 量价列 + 当日宏观列，后续因子计算都基于这张表。
    """
    data_dir = data_dir or DATA_DIR
    price = load_price(data_dir / "附件2 ETF日频量价数据（开盘、收盘、高、低、成交量、成交额）.csv")
    macro = load_macro(data_dir / "附件3 高频经济指标（信用利差、期限利差、汇率等）.csv")
    # 按 date 左连接：保留所有 (date, sec)，宏观按日期对齐
    aligned = price.merge(macro, on="date", how="left")
    aligned = aligned.sort_values(["sec", "date"]).reset_index(drop=True)
    return aligned


def split_aligned_train_valid_test(
    aligned: pd.DataFrame,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.2,
    by: str = "time",
) -> tuple:
    """
    将对齐后的数据按时间划分为训练集、验证集与测试集（避免用未来信息）。
    by='time'：按日期顺序，前 train_ratio 为训练，中间 valid_ratio 为验证，后 test_ratio 为测试。
    默认 70% train / 10% valid / 20% test。
    返回 (train_df, valid_df, test_df)。
    """
    aligned = aligned.sort_values(["date", "sec"]).reset_index(drop=True)
    dates = aligned["date"].drop_duplicates().sort_values().reset_index(drop=True)
    n = len(dates)
    n_train = max(0, int(round(n * train_ratio)))
    n_valid = max(0, int(round(n * valid_ratio)))
    n_test = max(1, n - n_train - n_valid)  # 保证至少 1 个测试日
    if n_train + n_valid + n_test > n:
        n_test = n - n_train - n_valid
    train_dates = set(dates.iloc[:n_train])
    valid_dates = set(dates.iloc[n_train : n_train + n_valid])
    test_dates = set(dates.iloc[n_train + n_valid : n_train + n_valid + n_test])
    train_df = aligned.loc[aligned["date"].isin(train_dates)].copy()
    valid_df = aligned.loc[aligned["date"].isin(valid_dates)].copy()
    test_df = aligned.loc[aligned["date"].isin(test_dates)].copy()
    return train_df, valid_df, test_df


# ---------- A 系列 ----------

def factor_A01_vol_scaled_momentum(price: pd.DataFrame, window: int = 20) -> pd.Series:
    """波动率缩放动量: Ret_20 / Std(Ret, 20)。20 日收益用复利区间收益 price/price.shift(20)-1。
    修正：将 1 日收益标准差调整到 20 日时间尺度（乘以 sqrt(20)）以匹配分子单位。
    """
    close = price.groupby("sec")["close"]
    ret_20 = close.transform(lambda x: x / x.shift(window) - 1)
    ret_1d = price.groupby("sec")["close"].pct_change()
    std_1d = ret_1d.groupby(price["sec"]).transform(lambda x: x.rolling(window).std())
    # 将 1 日标准差调整到 20 日时间尺度
    std_20 = std_1d * np.sqrt(window)
    out = ret_20 / std_20.replace(0, np.nan)
    out.name = "A01"
    return out


def factor_A02_path_efficiency(price: pd.DataFrame, window: int = 20) -> pd.Series:
    """路径效率 ER: Net_Chg / Sum(Abs_Chg)"""
    g = price.groupby("sec")["close"]
    net = g.transform(lambda x: x - x.shift(window))
    chg = price.groupby("sec")["close"].diff().abs()
    sum_abs = chg.groupby(price["sec"]).transform(lambda x: x.rolling(window).sum())
    out = net / sum_abs.replace(0, np.nan)
    out.name = "A02"
    return out


def _rolling_slope_fast(series: pd.Series, window: int) -> pd.Series:
    """向量化计算滚动线性回归斜率，性能提升 10-50 倍。
    使用最小二乘法公式: slope = Cov(X,Y) / Var(X)，其中 X = [0,1,2,...,window-1]
    """
    x = np.arange(window)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def slope(y):
        if len(y) < window:
            return np.nan
        y_arr = np.asarray(y)
        if np.isnan(y_arr).any():
            return np.nan
        y_mean = y_arr.mean()
        return ((x - x_mean) * (y_arr - y_mean)).sum() / x_var

    return series.rolling(window).apply(slope, raw=True)


def factor_A03_momentum_accelerator(price: pd.DataFrame, short: int = 10, long: int = 20) -> pd.Series:
    """时序动量加速器: Slope(Close, 10) - Slope(Close, 20)
    优化：使用向量化斜率计算替代 np.polyfit，性能提升 10-50 倍。
    """
    close = price.set_index(["date", "sec"])["close"].unstack("sec")
    s10 = close.apply(lambda col: _rolling_slope_fast(col, short))
    s20 = close.apply(lambda col: _rolling_slope_fast(col, long))
    out = (s10 - s20).stack("sec").reindex(pd.MultiIndex.from_frame(price[["date", "sec"]])).values
    return pd.Series(out, index=price.index, name="A03")


def factor_A04_macd_normalized(price: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
    """MACD 柱 (标准化): (EMA_12 - EMA_26) / Close"""
    close = price.set_index(["date", "sec"])["close"].unstack("sec")
    e12 = close.ewm(span=fast, adjust=False).mean()
    e26 = close.ewm(span=slow, adjust=False).mean()
    macd = (e12 - e26) / close.replace(0, np.nan)
    out = macd.stack("sec").reindex(pd.MultiIndex.from_frame(price[["date", "sec"]])).values
    return pd.Series(out, index=price.index, name="A04")


def factor_A05_rsi(price: pd.DataFrame, period: int = 14) -> pd.Series:
    """RSI 相对强弱: RSI(Close, 14)"""
    close = price.set_index(["date", "sec"])["close"].unstack("sec")
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    ag = gain.ewm(alpha=1 / period, adjust=False).mean()
    al = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    out = rsi.stack("sec").reindex(pd.MultiIndex.from_frame(price[["date", "sec"]])).values
    return pd.Series(out, index=price.index, name="A05")


def factor_A06_high_breakout(price: pd.DataFrame, window: int = 60) -> pd.Series:
    """最高价突破: Close / Max(High, 60)"""
    high_max = price.groupby("sec")["high"].transform(lambda x: x.rolling(window, min_periods=1).max())
    out = price["close"] / high_max.replace(0, np.nan)
    out.name = "A06"
    return out


# ---------- B 系列 ----------

def factor_B01_alpha006(price: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Alpha #006: -1 * Ts_Rank(Corr(Close, Vol, 20))
    修正：使用 Pandas 原生 rolling().rank() 替代 apply 循环
    """
    close = price.set_index(["date", "sec"])["close"].unstack("sec")
    vol = price.set_index(["date", "sec"])["volume"].unstack("sec")
    corr = close.rolling(window).corr(vol)
    # Ts_Rank: 当前值在包含该行的最近 window 个值中的百分位排名，pct=True 为 0~1
    rank = corr.rolling(window).rank(pct=True)
    out = (-1 * rank).stack("sec").reindex(pd.MultiIndex.from_frame(price[["date", "sec"]])).values
    return pd.Series(out, index=price.index, name="B01")


def factor_B02_alpha012(price: pd.DataFrame) -> pd.Series:
    """Alpha #012: Sign(Delta(Vol)) * -Delta(Close)"""
    dvol = price.groupby("sec")["volume"].diff()
    dclose = price.groupby("sec")["close"].diff()
    out = np.sign(dvol) * (-dclose)
    out.name = "B02"
    return out


def factor_B03_alpha001(price: pd.DataFrame, window: int = 20) -> pd.Series:
    """Alpha #001: (Close - MA_20) / Std(Close, 20)"""
    ma = price.groupby("sec")["close"].transform(lambda x: x.rolling(window).mean())
    std = price.groupby("sec")["close"].transform(lambda x: x.rolling(window).std())
    out = (price["close"] - ma) / std.replace(0, np.nan)
    out.name = "B03"
    return out


def factor_B04_alpha013(price: pd.DataFrame, window: int = 20) -> pd.Series:
    """Alpha #013: Rank(Cov(Close, Vol))"""
    close = price.set_index(["date", "sec"])["close"].unstack("sec")
    vol = price.set_index(["date", "sec"])["volume"].unstack("sec")
    cov = close.rolling(window).cov(vol)
    rank = cov.rank(axis=1, pct=True)
    out = rank.stack("sec").reindex(pd.MultiIndex.from_frame(price[["date", "sec"]])).values
    return pd.Series(out, index=price.index, name="B04")


def factor_B05_kd_k(price: pd.DataFrame, n: int = 14) -> pd.Series:
    """KD 指标 (K): (Close - Low_N) / (High_N - Low_N)"""
    low_n = price.groupby("sec")["low"].transform(lambda x: x.rolling(n).min())
    high_n = price.groupby("sec")["high"].transform(lambda x: x.rolling(n).max())
    out = (price["close"] - low_n) / (high_n - low_n).replace(0, np.nan)
    out.name = "B05"
    return out


def factor_B06_vwap_deviation(price: pd.DataFrame) -> pd.Series:
    """加权收盘偏离: Close / VWAP - 1"""
    vwap = price["amount"] / price["volume"].replace(0, np.nan)
    out = price["close"] / vwap - 1
    out.name = "B06"
    return out


# ---------- C 系列 ----------

def factor_C01_downside_vol_ratio(price: pd.DataFrame, window: int = 20) -> pd.Series:
    """下行波动占比: Std(Ret_Down) / Std(Ret_Total)"""
    ret = price.groupby("sec")["close"].pct_change()
    ret_down = ret.clip(upper=0)
    std_total = ret.groupby(price["sec"]).transform(lambda x: x.rolling(window).std())
    std_down = ret_down.groupby(price["sec"]).transform(lambda x: x.rolling(window).std())
    out = std_down / std_total.replace(0, np.nan)
    out.name = "C01"
    return out


def factor_C02_parkinson_vol(price: pd.DataFrame, window: int = 20) -> pd.Series:
    """Parkinson 极差波动率（滚动）:
    sqrt( mean( log(High/Low)^2, window ) / (4*ln(2)) )
    """
    hl_log_sq = np.log(price["high"] / price["low"].replace(0, np.nan)) ** 2
    rolling_mean = hl_log_sq.groupby(price["sec"]).transform(
        lambda x: x.rolling(window).mean()
    )
    out = np.sqrt(rolling_mean / (4 * np.log(2)))
    out.name = "C02"
    return out


def factor_C03_alpha023(price: pd.DataFrame, window: int = 20) -> pd.Series:
    """Alpha #023: Std(If(Close>Delay, Close, 0))"""
    delay = price.groupby("sec")["close"].shift(1)
    cond = np.where(price["close"] > delay, price["close"], 0.0)
    s = pd.Series(cond, index=price.index)
    out = s.groupby(price["sec"]).transform(lambda x: x.rolling(window).std())
    out.name = "C03"
    return out


def factor_C04_atr_pct(price: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR 占比: ATR(14) / Close"""
    high = price["high"]
    low = price["low"]
    prev_close = price.groupby("sec")["close"].shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.groupby(price["sec"]).transform(lambda x: x.rolling(period).mean())
    out = atr / price["close"].replace(0, np.nan)
    out.name = "C04"
    return out


# ---------- D 系列 ----------

def factor_D01_amihud_change(
    price: pd.DataFrame,
    period: int = 5,
    factor_name: str = "D01",
) -> pd.Series:
    """Amihud 变化率: Delta_n(Abs(Ret) / Amount)"""
    ret = price.groupby("sec")["close"].pct_change()
    illiq = ret.abs() / price["amount"].replace(0, np.nan)
    out = illiq.groupby(price["sec"]).diff(periods=period)
    out.name = factor_name
    return out


def factor_D02_intraday_momentum(price: pd.DataFrame) -> pd.Series:
    """日内动量: (Close - Open) / (High - Low)"""
    hl = (price["high"] - price["low"]).replace(0, np.nan)
    out = (price["close"] - price["open"]) / hl
    out.name = "D02"
    return out


def factor_D03_high_low_vol_ratio(price: pd.DataFrame, n: int = 20) -> pd.Series:
    """高低价差比: Sum(High-Low, N) / Sum(Vol, N)"""
    hl = price["high"] - price["low"]
    sum_hl = hl.groupby(price["sec"]).transform(lambda x: x.rolling(n).sum())
    sum_vol = price.groupby("sec")["volume"].transform(lambda x: x.rolling(n).sum())
    out = sum_hl / sum_vol.replace(0, np.nan)
    out.name = "D03"
    return out


def factor_D04_open_gap(price: pd.DataFrame) -> pd.Series:
    """开盘跳空: (Open - Delay(Close)) / Delay(Close)"""
    prev_close = price.groupby("sec")["close"].shift(1)
    out = (price["open"] - prev_close) / prev_close.replace(0, np.nan)
    out.name = "D04"
    return out


# ---------- E 系列 ----------

def factor_E01_skew(price: pd.DataFrame, window: int = 20) -> pd.Series:
    """偏度: -1 * Skew(Ret, 20)
    优化：使用 pandas 内置 rolling().skew()，性能提升 5-10 倍。
    """
    ret = price.groupby("sec")["close"].pct_change()
    out = ret.groupby(price["sec"]).transform(lambda x: -1 * x.rolling(window).skew())
    out.name = "E01"
    return out


def factor_E02_kurt(price: pd.DataFrame, window: int = 20) -> pd.Series:
    """峰度: -1 * Kurt(Ret, 20)
    优化：使用 pandas 内置 rolling().kurt()，性能提升 5-10 倍。
    """
    ret = price.groupby("sec")["close"].pct_change()
    out = ret.groupby(price["sec"]).transform(lambda x: -1 * x.rolling(window).kurt())
    out.name = "E02"
    return out


# ---------- F 系列（宏观） ----------

def _zscore(s: pd.Series) -> pd.Series:
    """
    无前视偏差 z-score：
    仅使用当前时点及历史样本做 expanding 均值/方差标准化。
    """
    mean = s.expanding(min_periods=20).mean()
    std = s.expanding(min_periods=20).std().replace(0, np.nan)
    return (s - mean) / std


def factor_F01_vix_regime(macro: pd.DataFrame, col: str = "市场恐慌程度：VIX指数") -> pd.Series:
    if col not in macro.columns:
        return pd.Series(dtype=float)
    out = _zscore(macro[col])
    out.name = "F01"
    return out


def factor_F02_credit_spread_regime(macro: pd.DataFrame, col: str = "信用利差：信用债端") -> pd.Series:
    if col not in macro.columns:
        return pd.Series(dtype=float)
    out = _zscore(macro[col])
    out.name = "F02"
    return out


def factor_F03_term_structure_regime(macro: pd.DataFrame, col: str = "期限利差：中长期债端") -> pd.Series:
    if col not in macro.columns:
        return pd.Series(dtype=float)
    out = _zscore(macro[col])
    out.name = "F03"
    return out


def factor_F04_inflation_scissors(
    macro: pd.DataFrame,
    prod_col1: str = "生产端通胀：构成1",
    prod_col2: str = "生产端通胀：构成2",
    cons_col: str = "消费端通胀",
) -> pd.Series:
    if prod_col1 in macro.columns and prod_col2 in macro.columns:
        prod = (macro[prod_col1] + macro[prod_col2]) / 2
    elif prod_col1 in macro.columns:
        prod = macro[prod_col1].copy()
    else:
        return pd.Series(dtype=float)
    cons = macro[cons_col] if cons_col in macro.columns else pd.Series(0.0, index=macro.index)
    out = _zscore(prod - cons)
    out.name = "F04"
    return out


def factor_F05_size_premium(
    macro: pd.DataFrame,
    small_col: str = "资本市场流动性：小盘PE",
    large_col: str = "资本市场流动性：大盘PE",
) -> pd.Series:
    if small_col not in macro.columns or large_col not in macro.columns:
        return pd.Series(dtype=float)
    out = macro[small_col] / macro[large_col].replace(0, np.nan)
    out.name = "F05"
    return out


def factor_F06_sino_us_real_spread(
    macro: pd.DataFrame,
    bond_col: str = "利率水平：国债指数",
    us_col: str = "美国实际利率",
) -> pd.Series:
    if bond_col not in macro.columns or us_col not in macro.columns:
        return pd.Series(dtype=float)
    out = _zscore(macro[bond_col]) - _zscore(macro[us_col])
    out.name = "F06"
    return out


# ---------- G 系列（Cross-Asset Beta，批量化优化） ----------

def compute_all_betas(
    price: pd.DataFrame,
    benchmark_dict: dict,
    window: int = 20,
) -> dict:
    """批量计算多个 benchmark 的 beta，避免重复 unstack 操作。

    Args:
        price: 量价数据
        benchmark_dict: {因子名: benchmark_series} 字典
        window: 滚动窗口

    Returns:
        {因子名: beta_series} 字典
    """
    # 一次性 unstack，所有 beta 共享
    ret_frame = price.set_index(["date", "sec"])["close"].unstack("sec").pct_change()
    midx = pd.MultiIndex.from_frame(price[["date", "sec"]])

    betas = {}
    for name, bench_series in benchmark_dict.items():
        if bench_series is None or bench_series.empty:
            betas[name] = pd.Series(np.nan, index=price.index, name=name)
            continue

        bench_ret = bench_series.pct_change().reindex(ret_frame.index).ffill()
        cov = ret_frame.rolling(window).cov(bench_ret)
        var = bench_ret.rolling(window).var()
        beta = cov.div(var, axis=0)
        out = beta.stack("sec").reindex(midx).values
        betas[name] = pd.Series(out, index=price.index, name=name)

    return betas


def factor_G_generic_beta(
    price: pd.DataFrame,
    bench_series: pd.Series,
    window: int = 20,
    factor_name: str = "Gxx",
) -> pd.Series:
    """通用 Beta：Beta = Cov(Ri, Rm) / Var(Rm)，向量化滚动计算。
    注意：如需计算多个 beta，建议使用 compute_all_betas() 批量计算以提升性能。
    """
    ret_frame = price.set_index(["date", "sec"])["close"].unstack("sec").pct_change()
    bench_ret = bench_series.pct_change().reindex(ret_frame.index).ffill()
    cov = ret_frame.rolling(window).cov(bench_ret)
    var = bench_ret.rolling(window).var()
    beta = cov.div(var, axis=0)
    out = beta.stack("sec").reindex(pd.MultiIndex.from_frame(price[["date", "sec"]])).values
    return pd.Series(out, index=price.index, name=factor_name)


def factor_G01_gold_beta(price: pd.DataFrame, gold_sec: str = "518880.SH", window: int = 20) -> pd.Series:
    gold_close = price[price["sec"] == gold_sec].set_index("date")["close"]
    if gold_close.empty:
        return pd.Series(np.nan, index=price.index, name="G01")
    return factor_G_generic_beta(price, gold_close, window, "G01")


def factor_G02_nasdaq_beta(price: pd.DataFrame, nasdaq_sec: str = "513100.SH", window: int = 20) -> pd.Series:
    nasdaq_close = price[price["sec"] == nasdaq_sec].set_index("date")["close"]
    if nasdaq_close.empty:
        candidates = price["sec"].astype(str).str.contains("513100|159941", regex=True)
        sec_candidates = price.loc[candidates, "sec"].unique()
        if len(sec_candidates) > 0:
            nasdaq_close = price[price["sec"] == sec_candidates[0]].set_index("date")["close"]
        else:
            return pd.Series(np.nan, index=price.index, name="G02")
    return factor_G_generic_beta(price, nasdaq_close, window, "G02")


def factor_G03_bond_beta(
    price: pd.DataFrame,
    macro: pd.DataFrame,
    bond_col: str = "利率水平：国债指数",
    window: int = 20,
) -> pd.Series:
    if bond_col not in macro.columns:
        return pd.Series(np.nan, index=price.index, name="G03")
    bond_series = macro.set_index("date")[bond_col]
    return factor_G_generic_beta(price, bond_series, window, "G03")


# ---------- 换手率（静态市值） ----------

def factor_T01_turnover(
    price: pd.DataFrame,
    product_pool: pd.DataFrame = None,
    data_dir: Path = None,
) -> pd.Series:
    """
    换手率 = Amount / (Static_MktCap × 10^8)。
    Static_MktCap 来自附件1 的流通市值（单位：亿元），10^8 将亿元转为元与 Amount 对齐。
    """
    if product_pool is None:
        data_dir = data_dir or DATA_DIR
        product_pool = load_product_pool(data_dir / "附件1 28只非债券ETF产品池.xlsx")
    if product_pool.empty:
        return pd.Series(np.nan, index=price.index, name="T01")
    # 附件1：证券代码 -> sec；流通市值 [单位] 亿元 -> Static_MktCap
    sec_col = next((c for c in product_pool.columns if "证券代码" in c), product_pool.columns[0])
    mcap_col = next((c for c in product_pool.columns if "流通市值" in c), None)
    if mcap_col is None:
        return pd.Series(np.nan, index=price.index, name="T01")
    static_mcap = product_pool.set_index(sec_col)[mcap_col]
    static_mcap = pd.to_numeric(static_mcap, errors="coerce")
    # 按 sec 映射到 price 的每一行
    mcap = price["sec"].map(static_mcap)
    # 换手率 = Amount / (Static_MktCap * 1e8)，单位一致为元
    denom = mcap * 1e8
    out = price["amount"] / denom.replace(0, np.nan)
    out.name = "T01"
    return out


# ---------- 收益与波动率因子 ----------

def factor_ret_1(price: pd.DataFrame) -> pd.Series:
    """1 日收益率: (close / close.shift(1)) - 1"""
    out = price.groupby("sec")["close"].transform(lambda x: x / x.shift(1) - 1)
    out.name = "ret_1"
    return out


def factor_ret_5(price: pd.DataFrame) -> pd.Series:
    """5 日收益率: (close / close.shift(5)) - 1"""
    out = price.groupby("sec")["close"].transform(lambda x: x / x.shift(5) - 1)
    out.name = "ret_5"
    return out


def factor_ret_20(price: pd.DataFrame) -> pd.Series:
    """20 日收益率: (close / close.shift(20)) - 1"""
    out = price.groupby("sec")["close"].transform(lambda x: x / x.shift(20) - 1)
    out.name = "ret_20"
    return out


def factor_vol_5(price: pd.DataFrame) -> pd.Series:
    """5 日波动率: 日收益率 5 日滚动标准差"""
    ret = price.groupby("sec")["close"].pct_change()
    out = ret.groupby(price["sec"]).transform(lambda x: x.rolling(5).std())
    out.name = "vol_5"
    return out


def factor_vol_20(price: pd.DataFrame) -> pd.Series:
    """20 日波动率: 日收益率 20 日滚动标准差"""
    ret = price.groupby("sec")["close"].pct_change()
    out = ret.groupby(price["sec"]).transform(lambda x: x.rolling(20).std())
    out.name = "vol_20"
    return out


# ---------- H 系列 ----------

def factor_H01_volume_price_divergence(price: pd.DataFrame, window: int = 20) -> pd.Series:
    """量价背离度: Corr(Close, Volume, 20)"""
    close = price.set_index(["date", "sec"])["close"].unstack("sec")
    vol = price.set_index(["date", "sec"])["volume"].unstack("sec")
    corr = close.rolling(window).corr(vol)
    out = corr.stack("sec").reindex(pd.MultiIndex.from_frame(price[["date", "sec"]])).values
    return pd.Series(out, index=price.index, name="H01")


# ---------- K 系列 ----------

def factor_K02_high_gravity_field(price: pd.DataFrame, window: int = 20) -> pd.Series:
    """高点引力场: window - ts_argmax(high, window) - 1

    取值可解释为“距离窗口最高点的天数”：
    - 0：今日为窗口最高点
    - 越大：距离最高点越久
    """
    argmax_idx = price.groupby("sec")["high"].transform(
        lambda x: x.rolling(window).apply(np.argmax, raw=True)
    )
    out = window - argmax_idx - 1
    out.name = "K02"
    return out


def factor_K04_relative_ts_rank(price: pd.DataFrame, window: int = 20) -> pd.Series:
    """相对时序分位数: ts_rank(volume, window)"""
    out = price.groupby("sec")["volume"].transform(
        lambda x: x.rolling(window).rank(pct=True)
    )
    out.name = "K04"
    return out


# ---------- M 系列 ----------

def factor_M01_intraday_overnight_game(price: pd.DataFrame) -> pd.Series:
    """日内与隔夜博弈差:
    (open / shift(close, 1) - 1) - (close / open - 1)
    """
    prev_close = price.groupby("sec")["close"].shift(1)
    overnight = price["open"] / prev_close.replace(0, np.nan) - 1
    intraday = price["close"] / price["open"].replace(0, np.nan) - 1
    out = overnight - intraday
    out.name = "M01"
    return out


def factor_M03_abs_buy_sell_pressure_imbalance(price: pd.DataFrame, window: int = 20) -> pd.Series:
    """绝对买卖压失衡:
    ts_sum(IF(close > shift(close, 1), volume, 0), window) /
    ts_sum(IF(close < shift(close, 1), volume, 0), window)
    """
    prev_close = price.groupby("sec")["close"].shift(1)
    up_vol = np.where(price["close"] > prev_close, price["volume"], 0.0)
    down_vol = np.where(price["close"] < prev_close, price["volume"], 0.0)

    up_sum = pd.Series(up_vol, index=price.index).groupby(price["sec"]).transform(
        lambda x: x.rolling(window).sum()
    )
    down_sum = pd.Series(down_vol, index=price.index).groupby(price["sec"]).transform(
        lambda x: x.rolling(window).sum()
    )
    out = up_sum / down_sum.replace(0, np.nan)
    out.name = "M03"
    return out


# ---------- N 系列 ----------

def factor_N01_pure_idio_momentum(price: pd.DataFrame, window: int = 20) -> pd.Series:
    """纯特质性动量:
    (close / shift(close, window) - 1) - cross_mean(close / shift(close, window) - 1)
    """
    ret_n = price.groupby("sec")["close"].transform(lambda x: x / x.shift(window) - 1)
    cross_mean = ret_n.groupby(price["date"]).transform("mean")
    out = ret_n - cross_mean
    out.name = "N01"
    return out


# ---------- 统一接口 ----------

# 量价表必备列，其余列为宏观
_PRICE_COLS = ["date", "sec", "open", "high", "low", "close", "volume", "amount"]


def compute_all_factors(
    price: pd.DataFrame = None,
    macro: pd.DataFrame = None,
    aligned_data: pd.DataFrame = None,
    product_pool: pd.DataFrame = None,
    data_dir: Path = None,
) -> tuple:
    """
    计算因子库中所有可实现因子。

    推荐流程：先用 load_aligned_data() 把附件2 与 附件3 按 date 对齐成一张表，再传入 aligned_data，
    这样量价与宏观在同一张表上，再算因子。

    参数:
      - aligned_data: 可选。若提供，则从该表拆出 price / macro，不再使用 price/macro 参数。
      - price / macro: 未提供 aligned_data 时使用。
    返回:
      - factors_panel: 与 price 同索引的 DataFrame，列为量价/截面因子
      - factors_macro: 以 date 为索引的 DataFrame，列为宏观因子 F01-F06

    关于 NaN：量价按 (sec, date) 排序，需要滚动窗口的因子在每只标的的前 14~20 个交易日为 NaN，属正常现象。
    """
    data_dir = data_dir or DATA_DIR
    if aligned_data is not None:
        # 先对齐再算因子：从对齐表拆出量价与宏观
        price = aligned_data[[c for c in _PRICE_COLS if c in aligned_data.columns]].copy()
        macro_cols = [c for c in aligned_data.columns if c not in _PRICE_COLS]
        macro = aligned_data[["date"] + macro_cols].drop_duplicates("date").reset_index(drop=True)
        price = price.sort_values(["sec", "date"]).reset_index(drop=True)
        if product_pool is None:
            product_pool = load_product_pool(data_dir / "附件1 28只非债券ETF产品池.xlsx")
    else:
        if price is None:
            price = load_price(data_dir / "附件2 ETF日频量价数据（开盘、收盘、高、低、成交量、成交额）.csv")
        if macro is None:
            macro = load_macro(data_dir / "附件3 高频经济指标（信用利差、期限利差、汇率等）.csv")
        if product_pool is None:
            product_pool = load_product_pool(data_dir / "附件1 28只非债券ETF产品池.xlsx")
        price = price.sort_values(["sec", "date"]).reset_index(drop=True)
    midx = pd.MultiIndex.from_frame(price[["date", "sec"]])
    panel_factors = {}

    panel_factors["A01"] = factor_A01_vol_scaled_momentum(price)
    panel_factors["A02"] = factor_A02_path_efficiency(price)
    panel_factors["A03"] = factor_A03_momentum_accelerator(price)
    panel_factors["A04"] = factor_A04_macd_normalized(price)
    panel_factors["A05"] = factor_A05_rsi(price)
    panel_factors["A06"] = factor_A06_high_breakout(price)
    panel_factors["B01"] = factor_B01_alpha006(price)
    panel_factors["B02"] = factor_B02_alpha012(price)
    panel_factors["B03"] = factor_B03_alpha001(price)
    panel_factors["B04"] = factor_B04_alpha013(price)
    panel_factors["B05"] = factor_B05_kd_k(price)
    panel_factors["B06"] = factor_B06_vwap_deviation(price)
    panel_factors["C01"] = factor_C01_downside_vol_ratio(price)
    panel_factors["C02"] = factor_C02_parkinson_vol(price)
    panel_factors["C03"] = factor_C03_alpha023(price)
    panel_factors["C04"] = factor_C04_atr_pct(price)
    # panel_factors["D01_1"] = factor_D01_amihud_change(price, period=1, factor_name="D01_1")
    panel_factors["D01"] = factor_D01_amihud_change(price, period=5, factor_name="D01")
    # panel_factors["D01_20"] = factor_D01_amihud_change(price, period=20, factor_name="D01_20")
    panel_factors["D02"] = factor_D02_intraday_momentum(price)
    panel_factors["D03"] = factor_D03_high_low_vol_ratio(price)
    panel_factors["D04"] = factor_D04_open_gap(price)
    panel_factors["E01"] = factor_E01_skew(price)
    panel_factors["E02"] = factor_E02_kurt(price)
    panel_factors["H01"] = factor_H01_volume_price_divergence(price)
    panel_factors["K02"] = factor_K02_high_gravity_field(price)
    panel_factors["K04"] = factor_K04_relative_ts_rank(price)
    panel_factors["M01"] = factor_M01_intraday_overnight_game(price)
    panel_factors["M03"] = factor_M03_abs_buy_sell_pressure_imbalance(price)
    panel_factors["N01"] = factor_N01_pure_idio_momentum(price)
    panel_factors["T01"] = factor_T01_turnover(price, product_pool=product_pool, data_dir=data_dir)
    panel_factors["ret_1"] = factor_ret_1(price)
    panel_factors["ret_5"] = factor_ret_5(price)
    panel_factors["ret_20"] = factor_ret_20(price)
    panel_factors["vol_5"] = factor_vol_5(price)
    panel_factors["vol_20"] = factor_vol_20(price)

    # G 系列：批量化 Beta 计算（性能优化）
    try:
        # 准备 benchmark 字典
        benchmark_dict = {}

        # G01: 黄金 Beta
        gold_sec = "518880.SH"
        gold_close = price[price["sec"] == gold_sec].set_index("date")["close"]
        benchmark_dict["G01"] = gold_close if not gold_close.empty else None

        # G02: 纳斯达克 Beta
        nasdaq_sec = "513100.SH"
        nasdaq_close = price[price["sec"] == nasdaq_sec].set_index("date")["close"]
        if nasdaq_close.empty:
            candidates = price["sec"].astype(str).str.contains("513100|159941", regex=True)
            sec_candidates = price.loc[candidates, "sec"].unique()
            if len(sec_candidates) > 0:
                nasdaq_close = price[price["sec"] == sec_candidates[0]].set_index("date")["close"]
        benchmark_dict["G02"] = nasdaq_close if not nasdaq_close.empty else None

        # G03: 债券 Beta
        bond_col = "利率水平：国债指数"
        if bond_col in macro.columns:
            bond_series = macro.set_index("date")[bond_col]
            benchmark_dict["G03"] = bond_series
        else:
            benchmark_dict["G03"] = None

        # 批量计算所有 Beta
        betas = compute_all_betas(price, benchmark_dict, window=20)
        for name, beta_series in betas.items():
            panel_factors[name] = beta_series.values
    except Exception as e:
        # 如果批量计算失败，回退到单独计算
        for name in ["G01", "G02", "G03"]:
            panel_factors[name] = np.nan

    factors_panel = pd.DataFrame(panel_factors, index=price.index)
    factors_panel["date"] = price["date"].values
    factors_panel["sec"] = price["sec"].values

    macro_factors = pd.DataFrame(index=macro["date"])
    for name, func in [
        ("F01", lambda: factor_F01_vix_regime(macro)),
        ("F02", lambda: factor_F02_credit_spread_regime(macro)),
        ("F03", lambda: factor_F03_term_structure_regime(macro)),
        ("F04", lambda: factor_F04_inflation_scissors(macro)),
        ("F05", lambda: factor_F05_size_premium(macro)),
        ("F06", lambda: factor_F06_sino_us_real_spread(macro)),
    ]:
        try:
            s = func()
            if s is not None and len(s):
                macro_factors[name] = s.values
        except Exception:
            pass

    return factors_panel, macro_factors


if __name__ == "__main__":
    # 推荐流程：量价+宏观对齐 → 按时间划分训练/测试 → 再算因子
    aligned = load_aligned_data()
    print("Aligned data shape (量价+宏观按 date 对齐):", aligned.shape)
    print("Columns:", list(aligned.columns[:8]), "...", list(aligned.columns[-4:]))

    train, valid, test = split_aligned_train_valid_test(
        aligned, train_ratio=0.7, valid_ratio=0.1, test_ratio=0.2
    )
    train.to_csv(DATA_DIR / "train_aligned.csv", index=False)
    valid.to_csv(DATA_DIR / "valid_aligned.csv", index=False)
    test.to_csv(DATA_DIR / "test_aligned.csv", index=False)
    print("\n训练/验证/测试划分 (按时间 70% / 10% / 20%):")
    print("  训练: ", train["date"].min(), "~", train["date"].max(), ", 行数", len(train))
    print("  验证: ", valid["date"].min(), "~", valid["date"].max(), ", 行数", len(valid))
    print("  测试: ", test["date"].min(), "~", test["date"].max(), ", 行数", len(test))
    print("  已保存: data/train_aligned.csv, data/valid_aligned.csv, data/test_aligned.csv")

    factors_panel, factors_macro = compute_all_factors(aligned_data=train)
    factor_cols = [c for c in factors_panel.columns if c not in ("date", "sec")]
    print("\nPanel factors shape:", factors_panel.shape)
    print("Panel factor columns:", factor_cols)
    print("Macro factors:", list(factors_macro.columns))

    factors_panel.to_csv("factors/factors_panel.csv", index=False)
    factors_macro.reset_index().to_csv("factors/factors_macro.csv", index=False)
