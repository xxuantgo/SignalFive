# -*- coding: utf-8 -*-
"""
对指定 run 目录中的「合成因子序列.csv」做单因子测试，并写回「单因子测试结果.csv」。

严格样本边界说明（与 run_main 口径一致）：
1) 回测起点由 BACKTEST_START 定义（默认 2021-01-04）；
2) 单因子统计仅使用回测前样本（截至回测起点前一日）；
3) IC 使用 t->t+fwd 的前瞻收益，先右移 fwd 个交易日后再截断，
   以确保统计不使用回测区间信息。

用法示例:
    python src/signalfive/pipelines/run_composite_factor_test.py 20260227_085729_869879
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from signalfive.config import BACKTEST_START, FORWARD_RETURN_PERIODS, OUTPUT_DIR
from signalfive.data_loader.loader import load_all
from signalfive.factors.testing import select_effective_factors_from_ic, test_all_factors


COMPOSITE_FACTOR_NAME = "合成因子"
COMPOSITE_FILE_NAME = "合成因子序列.csv"
SINGLE_TEST_FILE_NAME = "单因子测试结果.csv"


def _resolve_run_dir(run_dir: str) -> Path:
    p = Path(run_dir)
    if not p.is_absolute():
        p = OUTPUT_DIR / p
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"未找到 run 目录: {p}")
    return p


def _load_composite_matrix(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到文件: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required_cols = {"date", "sec", "composite_score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} 缺少字段: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    mat = (
        df.pivot(index="date", columns="sec", values="composite_score")
        .sort_index()
        .sort_index(axis=1)
    )
    return mat


def _calc_composite_summary(
    composite_mat: pd.DataFrame,
    close_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    _, ic_series_dict = test_all_factors(
        factor_matrices={COMPOSITE_FACTOR_NAME: composite_mat},
        close_matrix=close_matrix,
    )

    raw_ic = ic_series_dict[COMPOSITE_FACTOR_NAME]
    fwd_shift = int(FORWARD_RETURN_PERIODS[0])
    train_cutoff = pd.Timestamp(BACKTEST_START) - pd.Timedelta(days=1)

    shifted_ic = raw_ic.shift(fwd_shift)
    shifted_prestart = shifted_ic.loc[:train_cutoff].dropna()

    # 与 run_main 保持同口径：对 shift 后 IC 在回测前区间做统计
    _, summary_df = select_effective_factors_from_ic(
        {COMPOSITE_FACTOR_NAME: shifted_ic},
        cutoff=str(train_cutoff.date()),
    )
    summary_df["factor"] = COMPOSITE_FACTOR_NAME

    # 严格性检查：shift 后用于统计的每个样本，其前瞻收益结束日都必须 <= 截止日
    idx = pd.Index(raw_ic.index)
    pos_map = pd.Series(range(len(idx)), index=idx)
    used_shift_pos = pos_map.loc[shifted_prestart.index].to_numpy(dtype=int) if len(shifted_prestart) > 0 else []
    used_raw_pos = [p - fwd_shift for p in used_shift_pos]
    used_forward_end_pos = [p + fwd_shift for p in used_raw_pos]
    used_forward_end_dates = idx[used_forward_end_pos] if len(used_forward_end_pos) > 0 else pd.Index([])

    strict_ok = True
    if len(used_forward_end_dates) > 0:
        strict_ok = bool((used_forward_end_dates <= train_cutoff).all())
        if not strict_ok:
            raise RuntimeError("检测到回测区间泄漏：合成因子测试样本包含回测起点之后的标签。")

    meta = {
        "fwd_shift": fwd_shift,
        "train_cutoff": train_cutoff,
        "raw_ic_start": raw_ic.dropna().index.min() if raw_ic.notna().any() else None,
        "raw_ic_end": raw_ic.dropna().index.max() if raw_ic.notna().any() else None,
        "shifted_prestart_start": shifted_prestart.index.min() if len(shifted_prestart) > 0 else None,
        "shifted_prestart_end": shifted_prestart.index.max() if len(shifted_prestart) > 0 else None,
        "used_count": int(len(shifted_prestart)),
        "forward_end_max": used_forward_end_dates.max() if len(used_forward_end_dates) > 0 else None,
        "strict_ok": strict_ok,
    }
    return summary_df, meta


def _merge_and_save(summary_path: Path, composite_summary: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if summary_path.exists():
        base = pd.read_csv(summary_path, encoding="utf-8-sig")
    else:
        base = pd.DataFrame(columns=composite_summary.columns)

    replaced_count = 0
    if "factor" in base.columns:
        replaced_count = int((base["factor"] == COMPOSITE_FACTOR_NAME).sum())
        base = base[base["factor"] != COMPOSITE_FACTOR_NAME]

    # 对齐列，避免列顺序不一致
    for col in base.columns:
        if col not in composite_summary.columns:
            composite_summary[col] = pd.NA
    composite_summary = composite_summary[base.columns] if len(base.columns) > 0 else composite_summary

    merged = pd.concat([base, composite_summary], ignore_index=True)
    merged.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return merged, replaced_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对指定输出目录中的合成因子做单因子测试，并并入单因子测试结果.csv。"
    )
    parser.add_argument(
        "run_dir",
        help="outputs 子目录名（例如 20260227_085729_869879）或绝对路径",
    )
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    composite_csv = run_dir / COMPOSITE_FILE_NAME
    summary_csv = run_dir / SINGLE_TEST_FILE_NAME
    train_cutoff = pd.Timestamp(BACKTEST_START) - pd.Timedelta(days=1)
    fwd_shift = int(FORWARD_RETURN_PERIODS[0])

    print("合成因子单因子测试（严格回测前样本）")
    print(f"目标 run 目录: {run_dir}")
    print(f"回测起点: {BACKTEST_START}")
    print(f"测试样本截止: {train_cutoff.date()}（仅回测前）")
    print(f"前瞻收益周期: ret_{fwd_shift}d")
    print("边界处理: 先算 IC(t->t+fwd)，再右移 fwd 个交易日，最后按截止日截断。")
    print(f"读取合成因子: {composite_csv}")

    composite_mat = _load_composite_matrix(composite_csv)
    print(f"合成因子矩阵形状: {composite_mat.shape}")

    data = load_all()
    close_matrix = data["close_matrix"]
    composite_summary, meta = _calc_composite_summary(composite_mat, close_matrix)

    print("样本边界检查:")
    print(f"  原始IC区间: {meta['raw_ic_start']} ~ {meta['raw_ic_end']}")
    print(
        f"  shift后且截止前可观测区间: "
        f"{meta['shifted_prestart_start']} ~ {meta['shifted_prestart_end']}"
    )
    print(f"  统计样本点数: {meta['used_count']}")
    print(f"  标签结束日最大值: {meta['forward_end_max']}")
    print(f"  严格性检查: {'通过' if meta['strict_ok'] else '失败'}")

    print("新增/更新记录:")
    print(composite_summary.to_string(index=False))
    
    merged, replaced_count = _merge_and_save(summary_csv, composite_summary)
    print(f"写入单因子测试结果: {summary_csv}")
    print(f"替换旧『{COMPOSITE_FACTOR_NAME}』记录: {replaced_count} 条")
    print(f"结果总行数: {len(merged)}")

    


if __name__ == "__main__":
    main()
