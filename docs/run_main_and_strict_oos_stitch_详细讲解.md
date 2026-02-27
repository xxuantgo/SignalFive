# run_main.py 与 run_strict_oos_stitch.py 详细讲解

本文面向“能看懂 Python，但对量化回测链路不熟”的读者，目标是把这两个脚本的思路讲清楚，而不只是解释代码行数。

## 1. 先说结论：两个脚本各做什么

- `run_main.py`：项目主流程，一次性跑完“因子 -> 选股 -> 优化 -> 回测”，用于产出主结果与提交版复现结果。
- `run_strict_oos_stitch.py`：更严格的样本外评估流程，按时间分段，每段都在当时可用历史上重新调参，再把各段 OOS 净值拼接，用于检验策略是否“真实可推广”。

一句话理解：
- `run_main.py` 解决“策略怎么跑起来并拿结果”。
- `run_strict_oos_stitch.py` 解决“这个结果是否经得起严格时间推进检验”。

---

## 2. 项目主链路（两脚本共享的底层逻辑）

无论哪条脚本，核心都依赖这条链路：

1. 读取数据：ETF 量价 + 宏观 + 产品池约束（仅 28 只 ETF）。
2. 计算因子：截面因子（A/B/C...）用于选股；宏观因子（F01...F06）用于仓位调节。
3. 单因子评估：算 Rank IC / ICIR。
4. 因子合成：把多个有效因子合成一个 `composite_score`。
5. 生成调仓计划：
   - 先按 `composite_score` 选 TopN；
   - 再用优化器（默认 `hybrid_cvar_rp`）给权重。
6. Regime 仓位缩放：宏观风险高时降低总仓位（保留现金）。
7. 回测与绩效输出：净值、回撤、Sharpe、Calmar 等。

这条链路的关键是：**每一步都在避免前视偏差**（只用当时能看到的数据）。

---

## 3. run_main.py 由浅入深

文件：`src/signalfive/pipelines/run_main.py`

### 3.1 它的定位

- 提交版主入口。
- 参数几乎都从 `src/signalfive/config/base.py` 读取，不走命令行覆盖。
- 目标是稳定复现一个“固定策略配置”。

### 3.2 它做的 7 个步骤

脚本中已按 Step 1~7 打印，逻辑很清晰：

1. `Step 1` 加载数据  
调用 `load_all()`，得到：
- `aligned`：量价与宏观对齐后的宽表
- `close_matrix`：`date x sec` 收盘价矩阵（回测核心输入）

2. `Step 2` 计算因子并预处理  
调用：
- `compute_factors(...)`
- `prepare_factor_matrices(..., method="rank")`

这里会做截面去极值 + 标准化，得到每个因子的矩阵。

3. `Step 3` 单因子测试  
调用 `test_all_factors(...)` 拿到每个因子的 IC 序列；再把 IC 做 `fwd_shift`（按 5 日前瞻收益位移）后筛选有效因子报告。  
注意：最终真正用于合成的是 `base.py` 里的 `DEFAULT_EFFECTIVE_FACTORS`（固定列表），不是自动筛选结果。

4. `Step 4` 因子合成  
调用 `combine_factors(...)`，默认 `COMBINE_METHOD = "icir_robust"`。  
会输出 `合成因子序列.csv`。

5. `Step 5` 宏观 Regime 仓位信号  
调用 `calc_position_scale(...)` 得到每日仓位系数 `position_scale`，再保存到 `宏观仓位系数.csv`。  
如果 `REGIME_MODE == "off"`，则系数恒为 1。

6. `Step 6` 生成调仓计划  
两套组合并行生成：
- 等权：`build_equal_weight_schedule(...)`
- 优化：`build_optimized_schedule(...)`

然后统一 `apply_position_scale(...)`，把仓位系数乘到调仓权重上。

7. `Step 7` 回测与输出  
调用 `run_backtests(...)` 跑“等权组合”和“优化后组合”，导出：
- 净值序列
- 回测图（净值/回撤）
- `绩效汇总.csv`

### 3.3 它最关键的“防作弊”设计

1. 调仓日用“前一交易日信号”选股  
在引擎里（`engine.py`）是通过 `signal_dt = previous_trading_date(rebal_date)` 实现，避免用到调仓日收盘后的信息。

2. 优化窗口收益截止到信号日  
`ret_win = returns.loc[:signal_dt].iloc[-cov_window:]`，避免把调仓日收益混进训练样本。

3. IC 序列做 forward shift  
因为 IC[t] 依赖未来收益，必须右移后才能在实时里“可见”。

### 3.4 你最应该关注的配置入口（base.py）

- `DEFAULT_EFFECTIVE_FACTORS`：固定有效因子集
- `DEFAULT_PORTFOLIO_PARAMS`：`top_n`、`cvar_alpha`、`cov_window` 等
- `COMBINE_METHOD`：默认 `icir_robust`
- `OPTIMIZER_METHOD`：默认 `hybrid_cvar_rp`
- `REGIME_*`：仓位调节参数

本项目当前 README 给出的默认最优思路也是围绕这组配置。

---

## 4. run_strict_oos_stitch.py 由浅入深

文件：`src/signalfive/pipelines/run_strict_oos_stitch.py`

### 4.1 为什么需要它

`run_main.py` 能给你一个完整回测结果，但你还会担心：
- 参数是不是“后验最优”？
- 如果每一年都重新调参，结果会不会变差？

`run_strict_oos_stitch.py` 就是为这个问题设计的：
- 外层分段做 OOS；
- 每段开跑前只用历史数据重新调参；
- 每段只用本段参数；
- 最后拼接各段净值。

这比“一次性全区间回测”更接近真实投研流程。

### 4.2 它的三层结构

1. 信号层（因子 + Regime）  
- 可 `--reuse-run-dir` 复用 `run_main` 产出的 `合成因子序列.csv` 与 `宏观仓位系数.csv`
- 或 fresh 重新计算

2. 参数层（每个 outer 段单独调参）  
- 对每个外层 OOS 段，先构建内层 WFO folds
- 在内层 folds 上用 Optuna 搜 `top_n / alpha / cov_window / lambda / beta / cvar_method`

3. 评估层（分段 OOS + 拼接）  
- 本段参数只跑本段 OOS
- 记录分段绩效
- 最后把各段净值首尾拼接为总净值

### 4.3 外层与内层分别是什么

- 外层（outer folds）：真正 OOS 测试段，决定最终拼接净值。
- 内层（inner folds）：在外层段开始之前，用历史数据做参数选择。

可以理解为：
- 内层回答“该用什么参数”；
- 外层回答“这些参数在未来是否有效”。

### 4.4 它的目标函数（Bayes 调参的核心）

每组参数在内层 folds 上算：
- `mean_sharpe`
- `std_sharpe`
- `min_sharpe`
- `avg_turnover`

然后最大化：

`objective = mean_sharpe - a*std_sharpe - b*max(0, sharpe_floor - min_sharpe) - c*avg_turnover - 缺折惩罚`

其中 `a/b/c` 对应命令行参数：
- `--obj-std-penalty`
- `--obj-worst-penalty`
- `--obj-turnover-penalty`

这体现了“稳健优先”，不是只追求单一最高 Sharpe。

### 4.5 no-bayes-first-n-outer 机制（首段稳态锚定）

参数 `--no-bayes-first-n-outer` 默认是 `1`。含义：
- 前 N 个外层段不跑 Bayes；
- 直接用 `first-fold-anchor-*` 参数（默认来自 `base.py`）。

目的：
- 避免最早阶段历史太短，Bayes 不稳定；
- 先用主流程参数作为锚，再逐段放开搜索。

### 4.6 它输出哪些关键文件

在 `outputs/strict_oos_stitch_<timestamp>/` 下：

- `STRICT_OOS_outer_folds.csv`：外层分段
- `STRICT_OOS_inner_folds_outer*.csv`：每个外层段对应内层切分
- `STRICT_OOS_trials.csv`：所有 Bayes trial 明细
- `STRICT_OOS_selected_params.csv`：每个外层段最终参数
- `STRICT_OOS_segment_performance.csv`：每段 OOS 绩效
- `严格OOS拼接_净值序列.csv`：拼接后净值
- `严格OOS拼接_绩效汇总.csv` / `严格OOS拼接_摘要.json`：总结果摘要

---

## 5. 两个脚本的关系与使用建议

### 5.1 什么时候用 run_main.py

- 日常迭代策略；
- 产出主回测结果；
- 快速看“等权 vs 优化后”对比；
- 提交版复现。

### 5.2 什么时候用 run_strict_oos_stitch.py

- 验证参数稳定性；
- 验证真实 OOS 可迁移性；
- 做更严格的报告展示（分段参数、分段绩效、拼接净值）。

### 5.3 建议流程（实践）

1. 先跑 `run_main.py`，确认主流程稳定、信号文件产出正常。  
2. 再跑 `run_strict_oos_stitch.py --reuse-run-dir <run_main输出目录>`，专注调参与 OOS 验证。  
3. 比较 `run_main` 全区间结果与 strict OOS 拼接结果差异，判断是否过拟合。

---

## 6. 你看代码时最容易卡住的点（给你一个阅读抓手）

1. 先抓“数据结构”  
- 因子矩阵：`date x sec`
- 调仓计划：`{rebal_date: pd.Series(weights)}`
- 净值：`pd.Series(date -> nav)`

2. 再抓“时间边界”  
- 选股信号日是调仓日前一交易日  
- 优化样本截止信号日  
- strict OOS 每段参数只服务该段

3. 最后抓“目标函数”  
- `run_main` 是固定参数执行器  
- `strict` 是参数稳定性评估器（带稳健惩罚）

只要这三层抓住，你会发现代码虽然长，但逻辑并不绕。

---

## 7. 快速命令示例

主流程：

```bash
python src/signalfive/pipelines/run_main.py
```

严格 OOS（复用主流程信号）：

```bash
python src/signalfive/pipelines/run_strict_oos_stitch.py \
  --reuse-run-dir outputs/<某次run_main输出目录> \
  --n-trials 40 \
  --outer-test-start 2021-01-04 \
  --outer-test-end 2025-10-30 \
  --export-backtest-plots
```

