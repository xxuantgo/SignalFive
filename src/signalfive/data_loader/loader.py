# -*- coding: utf-8 -*-
"""
数据加载与预处理模块
====================
职责：
  1. 从原始 CSV / Excel 文件加载量价、宏观、产品池数据
  2. 按 date 对齐量价与宏观，生成 aligned 宽表
  3. 提取收盘价矩阵（bt 框架需要 DatetimeIndex × ETF 列名 的 DataFrame）
  4. 标记每周首个交易日（调仓日）
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from signalfive.config import (
    DATA_DIR, PRICE_FILE, MACRO_FILE, PRODUCT_POOL_FILE, BACKTEST_START, resolve_data_paths,
)


# ---------------------------------------------------------------------------
# 原始数据加载
# ---------------------------------------------------------------------------

def load_product_pool(path: Optional[Path] = None) -> pd.DataFrame:
    """加载附件1：28 只非债券 ETF 产品池"""
    path = path or PRODUCT_POOL_FILE
    if not path.exists():
        print(f"Warning: 产品池文件不存在 {path}")
        return pd.DataFrame()
    return pd.read_excel(path, sheet_name=0)


def _normalize_sec_code(code) -> str:
    """统一证券代码格式到 6 位数字 + .SH/.SZ。"""
    if pd.isna(code):
        return ""
    sec = str(code).strip().upper()
    if sec in {"", "NAN", "NONE"}:
        return ""

    # 处理 Excel 数值型代码（如 513500.0）
    if sec.endswith(".0") and sec[:-2].isdigit():
        sec = sec[:-2]

    if sec.endswith(".SH") or sec.endswith(".SZ"):
        num, market = sec.split(".")
        return f"{num.zfill(6)}.{market}" if num.isdigit() else sec

    if sec.isdigit():
        num = sec.zfill(6)
        market = "SH" if num.startswith(("5", "6", "9")) else "SZ"
        return f"{num}.{market}"

    return sec


def _extract_allowed_secs(product_pool: pd.DataFrame) -> set[str]:
    if product_pool.empty:
        return set()
    sec_col = next(
        (c for c in product_pool.columns if "证券代码" in str(c)),
        product_pool.columns[0],
    )
    allowed = {
        _normalize_sec_code(x) for x in product_pool[sec_col].tolist()
    }
    allowed.discard("")
    return allowed


def load_price(path: Optional[Path] = None,
               product_pool: Optional[pd.DataFrame] = None,
               enforce_pool: bool = True) -> pd.DataFrame:
    """
    加载附件2：日频量价数据
    返回 DataFrame，列 = [date, sec, open, high, low, close, volume, amount]
    """
    path = path or PRICE_FILE
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "sec" in df.columns:
        df["sec"] = df["sec"].map(_normalize_sec_code)
    # 必须有 close 才是有效行
    df = df.dropna(subset=["close"])

    # 赛题硬约束：仅允许附件1 产品池中的 ETF
    if enforce_pool:
        product_pool = product_pool if product_pool is not None else load_product_pool()
        allowed_secs = _extract_allowed_secs(product_pool)
        if allowed_secs:
            before_n = df["sec"].nunique()
            df = df[df["sec"].isin(allowed_secs)].copy()
            after_n = df["sec"].nunique()
            if after_n < before_n:
                print(f"Warning: 已按产品池过滤 ETF，{before_n} -> {after_n}")

    return df.sort_values(["sec", "date"]).reset_index(drop=True)


def load_macro(path: Optional[Path] = None) -> pd.DataFrame:
    """
    加载附件3：高频宏观经济指标
    返回 DataFrame，第一列为 date，其余列为各类指标
    """
    path = path or MACRO_FILE
    df = pd.read_csv(path)
    # 首列可能是空列名或 Unnamed
    if df.columns[0].strip() == "" or "Unnamed" in str(df.columns[0]):
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# 数据对齐
# ---------------------------------------------------------------------------

def build_aligned(price: Optional[pd.DataFrame] = None,
                  macro: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    将量价（附件2）与宏观（附件3）按 date 左连接对齐
    返回 aligned = (date, sec, open, high, low, close, volume, amount, 宏观列...)
    """
    if price is None:
        price = load_price()
    if macro is None:
        macro = load_macro()
    aligned = price.merge(macro, on="date", how="left")
    aligned = aligned.sort_values(["sec", "date"]).reset_index(drop=True)
    return aligned


# ---------------------------------------------------------------------------
# bt 框架所需的收盘价矩阵
# ---------------------------------------------------------------------------

def get_close_price_matrix(price: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    构造 bt 需要的价格矩阵：
      - index: DatetimeIndex (交易日)
      - columns: ETF 代码
      - values: 收盘价

    缺失值用前值填充（处理停牌），剩余填 NaN（bt 会自动跳过）。
    """
    if price is None:
        price = load_price()
    close = price.pivot_table(index="date", columns="sec", values="close")
    close = close.sort_index()
    close = close.ffill()  # 停牌用前一日收盘价填充
    return close


# ---------------------------------------------------------------------------
# 调仓日标记
# ---------------------------------------------------------------------------

def get_rebalance_dates(close_matrix: pd.DataFrame) -> pd.DatetimeIndex:
    """
    获取所有调仓日（每周第一个交易日）。
    逻辑：按自然周分组，取每周最小日期。
    """
    dates = pd.DatetimeIndex(close_matrix.index)
    date_series = pd.Series(dates, index=dates)
    # 以 ISO 年-周 作为分组键，取每组最小日期（该周首个交易日）
    iso_week_key = dates.strftime("%G-%V")
    weekly_first = date_series.groupby(iso_week_key).min()
    return pd.DatetimeIndex(weekly_first.to_numpy()).sort_values()


def get_rebalance_dates_from_start(close_matrix: pd.DataFrame,
                                    start: Optional[str] = None) -> pd.DatetimeIndex:
    """获取从回测起始日开始的调仓日"""
    start_ts = pd.Timestamp(start or BACKTEST_START)
    all_dates = get_rebalance_dates(close_matrix)
    return all_dates[all_dates >= start_ts]


# ---------------------------------------------------------------------------
# 便捷接口
# ---------------------------------------------------------------------------

def load_all(
    version: Optional[str] = None,
    price_path: Optional[Path] = None,
    macro_path: Optional[Path] = None,
    data_start: Optional[str] = None,
    data_end: Optional[str] = None,
):
    """
    一键加载所有数据，返回字典：
      {
        'price': 长表格式的量价数据,
        'macro': 宏观数据,
        'aligned': 对齐后的宽表,
        'close_matrix': 收盘价矩阵 (date × sec),
        'product_pool': 产品池,
        'data_range': {'start': 数据起始日, 'end': 数据截止日},
      }
    
    Args:
        version: 数据版本，如 "v20251030", "v20260313", "current", "auto"
                 为 None 时使用 base.py 中配置的 DATA_VERSION
        price_path: 直接指定量价数据文件路径（优先级高于version）
        macro_path: 直接指定宏观数据文件路径（优先级高于version）
        data_start: 数据起始日期过滤（格式：YYYY-MM-DD），只加载 >= 该日期的数据
        data_end: 数据截止日期过滤（格式：YYYY-MM-DD），只加载 <= 该日期的数据
    """
    # 解析数据路径
    if price_path is None or macro_path is None:
        if version is not None:
            resolved_price, resolved_macro = resolve_data_paths(version)
        else:
            resolved_price, resolved_macro = PRICE_FILE, MACRO_FILE
        price_path = price_path or resolved_price
        macro_path = macro_path or resolved_macro
    
    print(f"数据路径: price={price_path}, macro={macro_path}")
    
    print("加载量价数据...")
    price = load_price(path=price_path)
    
    # 应用日期范围过滤
    if data_start is not None:
        start_ts = pd.Timestamp(data_start)
        price = price[price["date"] >= start_ts].copy()
    if data_end is not None:
        end_ts = pd.Timestamp(data_end)
        price = price[price["date"] <= end_ts].copy()
    
    print(f"  量价: {price['date'].min().date()} ~ {price['date'].max().date()}, "
          f"{price['sec'].nunique()} 只 ETF, {len(price)} 行")

    print("加载宏观数据...")
    macro = load_macro(path=macro_path)
    
    # 应用日期范围过滤
    if data_start is not None:
        macro = macro[macro["date"] >= start_ts].copy()
    if data_end is not None:
        macro = macro[macro["date"] <= end_ts].copy()
    
    print(f"  宏观: {len(macro)} 行, {len(macro.columns)-1} 个指标")

    print("构建对齐宽表...")
    aligned = build_aligned(price, macro)

    print("构建收盘价矩阵...")
    close_matrix = get_close_price_matrix(price)
    print(f"  矩阵: {close_matrix.shape[0]} 天 × {close_matrix.shape[1]} 只 ETF")

    product_pool = load_product_pool()
    
    data_range = {
        "start": price["date"].min().date().isoformat() if not price.empty else None,
        "end": price["date"].max().date().isoformat() if not price.empty else None,
    }

    return {
        "price": price,
        "macro": macro,
        "aligned": aligned,
        "close_matrix": close_matrix,
        "product_pool": product_pool,
        "data_range": data_range,
    }


if __name__ == "__main__":
    data = load_all()
    print("\n调仓日示例 (前10个，从回测起点开始):")
    reb_dates = get_rebalance_dates_from_start(data["close_matrix"])
    print(reb_dates)
