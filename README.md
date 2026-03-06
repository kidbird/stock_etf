# A股ETF分析回测系统

支持431只ETF的数据获取、因子分析、策略回测和投资建议生成。

## 功能特性

- **数据获取**: 支持510/512/515/516/159/588开头的ETF基金（共431只）
- **回测因子**: 20+内置因子（收益类、风险类、动量类、趋势类）
- **回测策略**: RSI、MACD、均线交叉、布林带
- **投资建议**: 综合评分模型，生成买卖建议
- **多数据源**: Yahoo Finance + 国内数据源备份

## 安装

```bash
pip install requests pandas numpy jinja2
```

## 使用示例

```python
from etf_system import ETFAnalysisSystem

system = ETFAnalysisSystem()

# 1. 列出所有ETF
etfs = system.list_all_etfs()

# 2. 获取实时行情
quote = system.data_fetcher.get_realtime_quote("510300")

# 3. 运行回测
result = system.run_backtest("510300", "rsi", {'period': 14})

# 4. 获取投资建议
advice = system.get_investment_advice("510300", "rsi", result)
```

## 命令行使用

```bash
# 列出所有ETF
python etf_system.py --list

# 查询实时行情
python etf_system.py --quote --code 510300

# 运行回测分析
python etf_system.py --code 510300 --strategy rsi
```

## 文件说明

| 文件 | 说明 |
|------|------|
| etf_data.py | ETF数据获取模块 |
| etf_factors.py | 回测因子库 |
| etf_backtest.py | 回测引擎 |
| etf_report.py | 报告生成器 |
| etf_advisor.py | 投资建议模块 |
| etf_system.py | 主程序入口 |
| all_etf_codes.py | ETF代码映射表 |

## 数据源

- Yahoo Finance (主数据源)
- 东方财富 (备用)
- 腾讯 (备用)

## License

MIT
