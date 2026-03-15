# Data 文件夹整理规范

## 目录结构

```
data/
├── README.md                        # 本说明文件
├── versions/                        # 数据版本目录（核心）
│   ├── vYYYYMMDD/                   # 每个版本一个子目录
│   │   ├── price.csv                # ETF量价数据（标准化名称）
│   │   ├── macro.csv                # 宏观经济指标（标准化名称）
│   │   ├── 附件1 28只非债券ETF产品池.xlsx
│   │   └── ...                      # 其他相关文件、结果图表
│   └── vYYYYMMDD/                   # 其他版本
└── [临时文件]                        # 临时放在根目录的文件（需整理）
```

## 版本命名规范

- **格式**：`v` + 数据截止日期（YYYYMMDD）
- **示例**：`v20251030`、`v20260313`

## 文件命名规范（versions 内）

| 原始文件名 | 标准化名称 | 说明 |
|-----------|-----------|------|
| 附件2 ETF日频量价数据... | `price.csv` | 价格数据 |
| 附件3 高频经济指标... | `macro.csv` | 宏观数据 |
| 附件1 28只非债券ETF产品池.xlsx | 保持原样 | 产品池 |
| 等权_组合优化_净值曲线.png | 保持原样 | 结果图表 |
| 等权_组合优化_回撤曲线.png | 保持原样 | 结果图表 |
| 其他分析结果文件 | 保持原样 | 因子库、有效因子等 |

## 新增数据流程

当收到新的全量更新文件时：

```bash
# 1. 确定数据截止日期（从文件名或数据内容确认）
# 例如：0314文件包含数据到 2026-03-13

# 2. 创建新版本目录
mkdir data\versions\v20260313

# 3. 将文件移入并重命名
move "0314-全量更新-ETF量价数据.csv" "data\versions\v20260313\price.csv"
move "0314-全量更新-高频经济指标重命名.csv" "data\versions\v20260313\macro.csv"

# 4. （可选）将产品池也复制到新版本
copy "data\versions\v20251030\附件1 28只非债券ETF产品池.xlsx" "data\versions\v20260313\"
```

## 使用方式

```bash
# 使用 v20251030 数据回测
python -m signalfive.pipelines.run_main --data-version v20251030

# 使用 v20260313 数据回测（最新增量数据）
python -m signalfive.pipelines.run_main --data-version v20260313

# 自动选择最新版本
python -m signalfive.pipelines.run_main --data-version auto
```

## 当前版本说明

### v20251030
- **数据区间**：2019-11-01 ~ 2025-10-30
- **包含文件**：
  - `price.csv` - ETF量价数据
  - `macro.csv` - 宏观经济指标
  - `附件1 28只非债券ETF产品池.xlsx`
  - `因子库.csv`
  - `有效因子.csv`
  - `有效因子ICIR_回测前.csv`
  - `corr0.7_greedy_有效因子.csv`
  - `高相关因子对_阈值0.7.csv`

### v20260313
- **数据区间**：2019-11-01 ~ 2026-03-13
- **包含文件**：
  - `price.csv` - ETF量价数据（全量更新）
  - `macro.csv` - 宏观经济指标（全量更新）

## 注意事项

1. **结果图表**：应存放在对应版本文件夹中，与数据保持一致
2. **临时文件**：data 根目录不应长期存放数据文件，应及时整理到对应版本
3. **版本切换**：通过 `--data-version` 参数灵活切换不同版本进行回测
