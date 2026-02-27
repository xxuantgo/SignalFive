# -*- coding: utf-8 -*-
"""
CVaR 调参结果可视化
===================
输入 trials DataFrame，导出常用诊断图：
  1) 优化历史（objective）
  2) 参数重要性（Spearman 相关近似）
  3) turnover_lambda 与 Sharpe/Turnover
  4) alpha-window 热力图
  5) 收益-波动散点
  6) 分段 Sharpe（TopK）
  7) 最优点邻域稳定性（距离-目标）
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "缺少依赖 matplotlib，无法导出调参图。请先安装: `pip install matplotlib`。"
        ) from exc
    return plt


def _savefig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")


def _safe_bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].fillna(False).astype(bool)


def generate_cvar_bayes_plots(
    trials_df: pd.DataFrame,
    rp_ref: Dict[str, float],
    output_dir: str | Path,
    top_k: int = 10,
) -> List[str]:
    """
    生成贝叶斯调参图，返回已保存图路径列表。
    """
    plt = _require_matplotlib()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = trials_df.copy()
    if df.empty:
        return []

    if "trial_number" in df.columns:
        df = df.sort_values("trial_number").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
        df["trial_number"] = np.arange(len(df), dtype=int)

    saved: List[str] = []

    # 1) 优化历史
    if "objective" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(df["trial_number"], df["objective"], lw=1.0, alpha=0.65, label="objective")
        ax.plot(df["trial_number"], df["objective"].cummax(), lw=1.8, label="best-so-far")
        ax.set_title("Bayesian Optimization History")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Objective")
        ax.grid(alpha=0.25)
        ax.legend()
        p = out_dir / "01_optimization_history.png"
        _savefig(fig, p)
        plt.close(fig)
        saved.append(str(p))

    # 2) 参数重要性（Spearman |rho|）
    params = ["cvar_alpha", "cov_window", "max_weight", "turnover_lambda"]
    imp_rows = []
    if "objective" in df.columns:
        for p in params:
            if p not in df.columns or df[p].nunique(dropna=True) < 2:
                continue
            corr = df[[p, "objective"]].corr(method="spearman").iloc[0, 1]
            if np.isfinite(corr):
                imp_rows.append((p, abs(float(corr))))
    if imp_rows:
        imp_df = pd.DataFrame(imp_rows, columns=["param", "importance"]).sort_values("importance")
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.barh(imp_df["param"], imp_df["importance"])
        ax.set_title("Parameter Importance (|Spearman rho|)")
        ax.set_xlabel("Importance")
        ax.grid(axis="x", alpha=0.25)
        p = out_dir / "02_param_importance.png"
        _savefig(fig, p)
        plt.close(fig)
        saved.append(str(p))

    # 3) lambda vs sharpe/turnover
    if {"turnover_lambda", "sharpe", "avg_turnover"}.issubset(df.columns):
        g = (
            df.groupby("turnover_lambda", dropna=True)
            .agg(
                best_sharpe=("sharpe", "max"),
                mean_turnover=("avg_turnover", "mean"),
                pass_rate=("pass_all_rules", lambda x: float(pd.Series(x).fillna(False).astype(bool).mean())),
            )
            .sort_index()
        )
        if not g.empty:
            fig, ax1 = plt.subplots(figsize=(8, 4.5))
            ax1.plot(g.index, g["best_sharpe"], marker="o", lw=1.6, color="tab:blue", label="best Sharpe")
            ax1.set_xlabel("turnover_lambda")
            ax1.set_ylabel("best Sharpe", color="tab:blue")
            ax1.tick_params(axis="y", labelcolor="tab:blue")
            if (g.index > 0).all():
                ax1.set_xscale("log")
            ax1.grid(alpha=0.25)

            ax2 = ax1.twinx()
            ax2.plot(
                g.index, g["mean_turnover"], marker="s", lw=1.2, color="tab:orange", label="mean turnover"
            )
            ax2.set_ylabel("mean turnover", color="tab:orange")
            ax2.tick_params(axis="y", labelcolor="tab:orange")
            ax1.set_title("turnover_lambda vs Sharpe/Turnover")
            p = out_dir / "03_lambda_vs_sharpe_turnover.png"
            _savefig(fig, p)
            plt.close(fig)
            saved.append(str(p))

    # 4) alpha-window 热力图（按均值 Sharpe）
    if {"cvar_alpha", "cov_window", "sharpe"}.issubset(df.columns):
        hdf = df[["cvar_alpha", "cov_window", "sharpe"]].dropna()
        if not hdf.empty:
            hdf["alpha_bin"] = hdf["cvar_alpha"].round(2)
            hdf["window_bin"] = (np.round(hdf["cov_window"] / 10.0) * 10).astype(int)
            pivot = hdf.pivot_table(
                index="window_bin",
                columns="alpha_bin",
                values="sharpe",
                aggfunc="mean",
            )
            if not pivot.empty:
                fig, ax = plt.subplots(figsize=(8, 5.5))
                im = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower")
                ax.set_title("Sharpe Heatmap (alpha x window)")
                ax.set_xlabel("cvar_alpha")
                ax.set_ylabel("cov_window")
                ax.set_xticks(np.arange(len(pivot.columns)))
                ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns], rotation=45, ha="right")
                ax.set_yticks(np.arange(len(pivot.index)))
                ax.set_yticklabels([str(x) for x in pivot.index])
                fig.colorbar(im, ax=ax, shrink=0.85, label="mean Sharpe")
                p = out_dir / "04_alpha_window_heatmap.png"
                _savefig(fig, p)
                plt.close(fig)
                saved.append(str(p))

    # 5) 收益-波动散点
    if {"annual_return", "annual_vol", "sharpe"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        size = np.full(len(df), 32.0)
        if "avg_turnover" in df.columns:
            v = df["avg_turnover"].fillna(df["avg_turnover"].median()).to_numpy(dtype=float)
            size = 24.0 + 180.0 * np.clip(v, 0.0, 1.0)
        sc = ax.scatter(
            df["annual_vol"],
            df["annual_return"],
            c=df["sharpe"],
            s=size,
            alpha=0.72,
            cmap="viridis",
        )
        if "annual_vol" in rp_ref and "annual_return" in rp_ref:
            ax.scatter([rp_ref["annual_vol"]], [rp_ref["annual_return"]], c="red", marker="X", s=120, label="RP")
            ax.legend()
        ax.set_title("Return-Volatility Scatter (color=Sharpe, size=turnover)")
        ax.set_xlabel("Annual Volatility")
        ax.set_ylabel("Annual Return")
        ax.grid(alpha=0.25)
        fig.colorbar(sc, ax=ax, shrink=0.85, label="Sharpe")
        p = out_dir / "05_return_vol_scatter.png"
        _savefig(fig, p)
        plt.close(fig)
        saved.append(str(p))

    # 6) 分段 Sharpe 柱状图（TopK）
    split_cols = ["sharpe_2021_2022", "sharpe_2023", "sharpe_2024_2025"]
    if set(split_cols).issubset(df.columns):
        order_cols = ["pass_all_rules", "objective"] if "objective" in df.columns else ["sharpe"]
        top_df = df.sort_values(order_cols, ascending=False).head(max(1, int(top_k))).copy()
        if not top_df.empty:
            fig, ax = plt.subplots(figsize=(10, 4.8))
            x = np.arange(len(top_df))
            width = 0.24
            ax.bar(x - width, top_df["sharpe_2021_2022"], width=width, label="2021-2022")
            ax.bar(x, top_df["sharpe_2023"], width=width, label="2023")
            ax.bar(x + width, top_df["sharpe_2024_2025"], width=width, label="2024-2025")

            rp_s1 = rp_ref.get("sharpe_2021_2022")
            rp_s2 = rp_ref.get("sharpe_2023")
            rp_s3 = rp_ref.get("sharpe_2024_2025")
            if np.isfinite(rp_s1):
                ax.axhline(rp_s1, ls="--", lw=1.0, color="tab:blue", alpha=0.55)
            if np.isfinite(rp_s2):
                ax.axhline(rp_s2, ls="--", lw=1.0, color="tab:orange", alpha=0.55)
            if np.isfinite(rp_s3):
                ax.axhline(rp_s3, ls="--", lw=1.0, color="tab:green", alpha=0.55)

            ax.set_xticks(x)
            ax.set_xticklabels([f"T{int(n)}" for n in top_df["trial_number"]], rotation=0)
            ax.set_title(f"Split Sharpe for Top-{len(top_df)} Trials")
            ax.set_xlabel("Trial")
            ax.set_ylabel("Sharpe")
            ax.legend(ncols=3)
            ax.grid(axis="y", alpha=0.25)
            p = out_dir / "06_split_sharpe_bar_topk.png"
            _savefig(fig, p)
            plt.close(fig)
            saved.append(str(p))

    # 7) 最优点邻域稳定性：参数距离 vs objective
    req_cols = ["cvar_alpha", "cov_window", "max_weight", "turnover_lambda", "objective"]
    if set(req_cols).issubset(df.columns):
        best_idx = df["objective"].idxmax()
        best = df.loc[best_idx]

        # 归一化距离（turnover_lambda 用 log 空间）
        alpha_range = max(df["cvar_alpha"].max() - df["cvar_alpha"].min(), 1e-12)
        window_range = max(df["cov_window"].max() - df["cov_window"].min(), 1e-12)
        weight_range = max(df["max_weight"].max() - df["max_weight"].min(), 1e-12)

        lam_all = np.log10(df["turnover_lambda"].clip(lower=1e-8))
        lam_best = np.log10(max(float(best["turnover_lambda"]), 1e-8))
        lam_range = max(lam_all.max() - lam_all.min(), 1e-12)

        dist = np.sqrt(
            ((df["cvar_alpha"] - best["cvar_alpha"]) / alpha_range) ** 2
            + ((df["cov_window"] - best["cov_window"]) / window_range) ** 2
            + ((df["max_weight"] - best["max_weight"]) / weight_range) ** 2
            + ((np.log10(df["turnover_lambda"].clip(lower=1e-8)) - lam_best) / lam_range) ** 2
        )

        pass_flag = _safe_bool_series(df, "pass_all_rules")
        fig, ax = plt.subplots(figsize=(8, 4.8))
        sc = ax.scatter(dist, df["objective"], c=pass_flag.astype(int), cmap="coolwarm", alpha=0.72, s=40)
        ax.set_title("Local Stability Around Best Trial")
        ax.set_xlabel("Normalized Distance to Best Params")
        ax.set_ylabel("Objective")
        ax.grid(alpha=0.25)
        cb = fig.colorbar(sc, ax=ax, shrink=0.85)
        cb.set_label("pass_all_rules (0/1)")
        p = out_dir / "07_local_stability_best.png"
        _savefig(fig, p)
        plt.close(fig)
        saved.append(str(p))

    return saved


__all__ = ["generate_cvar_bayes_plots"]

