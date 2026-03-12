# A股 ETF / 股票多因子分析回测系统

面向 A 股 ETF 和股票的研究型分析系统，支持：

- 实时行情与历史数据获取
- SQLite 本地缓存
- 技术因子与基础因子计算
- 单策略回测
- 多因子组合打分择时回测
- Flask API 与命令行入口

## 当前版本重点

- `days` 统一表示“交易日样本数”，不再表示自然日。
- ETF 与股票因子分开管理。
- 新增 ETF / 股票多因子组合模板，可直接回测。
- 相对强弱支持 `20 / 60 / 120 / 250` 日窗口。
- 股票模块增加静态代码与行业兜底，离线退化能力更强。

## 安装

```bash
pip install -r requirements.txt
```

## 项目结构

```text
etf_system.py        ETF CLI 主入口
stock_system.py      股票 CLI 主入口
etf_web.py           Flask API
etf_data.py          ETF 数据获取与缓存
stock_data.py        股票数据获取与缓存
etf_factors.py       ETF 因子
stock_factors.py     股票因子
factor_combo.py      多因子组合模板与组合回测
etf_relative_strength.py  ETF 相对强弱
design.md            技术设计文档
agent.md             使用文档
```

## 快速开始

### ETF

```bash
# 查看 ETF 列表
python3 etf_system.py --list

# 查看可用因子和组合模板
python3 etf_system.py --list-factors
python3 etf_system.py --list-combos

# 单策略分析
python3 etf_system.py --code 510300 --strategy macd --days 500

# 多因子组合分析
python3 etf_system.py --code 510300 --combo etf_trend_size --days 500

# 相对强弱
python3 etf_system.py --rotation-rank --rotation-category industry --rs-window 250
```

### 股票

```bash
# 查看股票列表
python3 stock_system.py --list

# 查看可用因子和组合模板
python3 stock_system.py --list-factors
python3 stock_system.py --list-combos

# 单策略分析
python3 stock_system.py --code 000001 --strategy stock_macd --days 500

# 多因子组合分析
python3 stock_system.py --code 000001 --combo stock_quality_industry --days 500
```

### Web API

```bash
python3 etf_web.py
```

主要接口：

- `GET /api/analysis/<code>?strategy=macd&days=500`
- `GET /api/analysis/<code>?combo=etf_trend_size&days=500`
- `GET /api/stock/analysis/<code>?combo=stock_quality&days=500`
- `GET /api/etf/combo-templates`
- `GET /api/stock/combo-templates`

## 数据口径

### 历史数据

- `days` 表示“最近 N 个交易日样本”
- 数据层会按更长的自然日窗口拉取，再裁切成最近 N 条有效交易数据
- 本地缓存与网络拉取都遵循同一口径

### 缓存

- ETF 缓存数据库：`etf_cache.db`
- 股票缓存数据库：`stock_cache.db`
- 首次下载建议使用较长样本数，后续按日增量更新

## ETF 因子

ETF 因子以趋势与轮动为核心：

- 技术面：`rsi` `macd` `adx` `supertrend` `ma_alignment` `roc`
- 风险面：`volatility` `max_drawdown` `atr` `market_fear`
- 轮动面：`relative_strength`
- 产品属性：`fund_size` `fund_size_percentile` `turnover_liquidity`

内置 ETF 组合：

- `etf_trend`
- `etf_trend_size`

## 股票因子

股票因子分为技术面与基础面：

- 技术面：`rsi` `macd` `adx` `supertrend` `ma_alignment` `roc`
- 量价面：`obv` `volume_ratio` `price_volume_correlation`
- 基础面：`market_cap` `market_cap_bucket` `roe` `industry`
- 市场状态：`market_fear`

内置股票组合：

- `stock_quality`
- `stock_quality_industry`

## Python 调用

### ETF 多因子组合

```python
from etf_system import ETFAnalysisSystem

system = ETFAnalysisSystem()
result = system.analyze_factor_combo("510300", "etf_trend_size", days=500)

print(result["composite_score"])
print(result["factor_values"])
print(result["backtest"]["total_return"])
```

### 股票多因子组合

```python
from stock_system import StockAnalysisSystem

system = StockAnalysisSystem()
result = system.analyze_factor_combo("000001", "stock_quality", days=500)

print(result["composite_score"])
print(result["factor_values"])
print(result["backtest"]["total_return"])
```

## 已知限制

- 基础因子数据优先取公开接口，缺失时会返回 `None`
- 股票 `ROE` 属于最佳努力获取，不保证所有标的都可用
- `market_fear` 为系统内部构造的大盘恐慌代理，不是交易所官方 VIX
- 组合模式当前是“单标的时序打分择时”，不是截面选股框架

## 测试

```bash
python3 -m unittest discover -s tests -q
```
