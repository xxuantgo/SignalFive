# SignalFive

SignalFive 是从 `E:\USTC_Financial_Data_Inno_Comp` 迁移并整理后的版本，核心功能是 ETF 因子选股、组合优化与回测。

## 快速开始

```bash
conda env create -f environment.yml
conda activate signalfive_py311
pip install -e .
```

## 常用运行命令

```bash
# 主流程（因子 -> 选股 -> 优化 -> 回测）
python src/signalfive/pipelines/run_main.py

# WFO + Bayes 调参
python src/signalfive/pipelines/run_cvar_bayes.py --help

# 严格 OOS 流程
python src/signalfive/pipelines/run_strict_oos_stitch.py --help
```

也可以使用脚本：

```bash
# Windows
.\run_strict.bat

# Linux / macOS
bash ./run_strict.sh
```

安装为可执行命令后，也可以直接运行：

```bash
signalfive-main
signalfive-cvar-bayes --help
signalfive-strict-oos --help
```

## 目录结构

```text
SignalFive/
├── data/                     # 比赛原始数据 + 因子清单
├── docs/                     # 文档
├── src/
│   └── signalfive/               # 核心代码
├── scripts/                  # 可直接运行的批处理/脚本
├── pyproject.toml
├── requirements.txt
├── environment.yml
└── INSTALL.md
```

## 说明

- 运行输出默认写入 `outputs/`，日志写入 `logs/`。
- `.gitignore` 已默认忽略 `outputs/` 和 `logs/`，便于保持仓库干净。
- 项目根目录由 `src/signalfive/config/base.py` 自动推断，不依赖旧目录名。
