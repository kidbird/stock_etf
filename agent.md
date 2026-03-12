# 使用文档

本文档面向直接调用代码、CLI 或 API 的使用者。

## 1. ETF 使用

### 1.1 单策略

```python
from etf_system import ETFAnalysisSystem

system = ETFAnalysisSystem()
result = system.run_backtest("510300", "macd", days=500)
print(result.total_return)
```

### 1.2 多因子组合

```python
from etf_system import ETFAnalysisSystem

system = ETFAnalysisSystem()
result = system.analyze_factor_combo("510300", "etf_trend_size", days=500)

print(result["composite_score"])
print(result["factor_values"])
print(result["backtest"]["annualized_return"])
```

### 1.3 ETF 因子建议

- 趋势研究：`ma_alignment` `adx` `roc`
- 轮动研究：`relative_strength`
- 风险过滤：`market_fear`
- 产品筛选：`fund_size_percentile` `turnover_liquidity`

## 2. 股票使用

### 2.1 单策略

```python
from stock_system import StockAnalysisSystem

system = StockAnalysisSystem()
result = system.run_backtest("000001", "stock_macd", days=500)
print(result.total_return)
```

### 2.2 多因子组合

```python
from stock_system import StockAnalysisSystem

system = StockAnalysisSystem()
result = system.analyze_factor_combo("000001", "stock_quality_industry", days=500)

print(result["composite_score"])
print(result["factor_values"])
print(result["backtest"]["sharpe_ratio"])
```

### 2.3 股票因子建议

- 趋势：`ma_alignment` `adx` `roc`
- 质量：`roe`
- 规模：`market_cap` `market_cap_bucket`
- 风险：`market_fear`
- 属性：`industry`

## 3. 命令行

### ETF

```bash
python3 etf_system.py --list-factors
python3 etf_system.py --list-combos
python3 etf_system.py --code 510300 --combo etf_trend --days 500
```

### 股票

```bash
python3 stock_system.py --list-factors
python3 stock_system.py --list-combos
python3 stock_system.py --code 000001 --combo stock_quality --days 500
```

## 4. API

### ETF 多因子

```bash
GET /api/analysis/510300?combo=etf_trend_size&days=500
```

返回重点字段：

- `combo_result.composite_score`
- `combo_result.factor_values`
- `combo_result.factor_contributions`
- `combo_result.backtest`

### 股票多因子

```bash
GET /api/stock/analysis/000001?combo=stock_quality_industry&days=500
```

## 5. 结果解释

### 综合得分

- 接近 `1`：因子组合整体偏强
- 接近 `0`：信号中性
- 接近 `-1`：风险偏高或因子偏弱

### `days`

- 表示最近 N 个交易日样本
- 不是自然日

### 基础面缺失

- 市值 / ROE / ETF 规模拿不到时会返回 `None`
- 组合打分会把缺失因子按中性处理

## 6. 推荐流程

### ETF

1. 用 `relative_strength` 看轮动
2. 用 `etf_trend` 或 `etf_trend_size` 回测
3. 再看单策略表现做对照

### 股票

1. 先看 `stock_quality` 组合
2. 再叠加 `industry` 与 `roe`
3. 用 `market_fear` 判断是否需要降低风险
