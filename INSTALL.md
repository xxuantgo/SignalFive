# 安装与环境说明

## 方式 1：requirements.txt（推荐）

```bash
conda create -n signalfive_py311 python=3.11
conda activate signalfive_py311
pip install -r requirements.txt
pip install -e .
```

## 方式 2：environment.yml

```bash
conda env create -f environment.yml
conda activate signalfive_py311
pip install -e .
```

## 验证安装

```bash
signalfive-main
```

或：

```bash
python src/signalfive/pipelines/run_main.py
```

## 常见问题

如果出现 `No module named 'signalfive'`：

1. 确认已经激活目标 conda 环境。
2. 在项目根目录执行过 `pip install -e .`。
3. 临时设置 `PYTHONPATH`（可选）：

```bash
export PYTHONPATH=/path/to/SignalFive/src:$PYTHONPATH
```

Windows PowerShell：

```powershell
$env:PYTHONPATH="E:\SignalFive\src"
```
