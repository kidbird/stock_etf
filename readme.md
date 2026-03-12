# 说明

`README.md` 是当前主文档。

本项目现已支持：

- ETF / 股票双模块
- 交易日口径历史数据
- 技术因子与基础因子分离
- 多因子组合打分择时回测
- CLI / Python / Flask API 三类入口

常用命令：

```bash
python3 etf_system.py --list-combos
python3 etf_system.py --code 510300 --combo etf_trend_size --days 500
python3 stock_system.py --list-combos
python3 stock_system.py --code 000001 --combo stock_quality --days 500
python3 etf_web.py
```
