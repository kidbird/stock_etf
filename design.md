# 系统设计文档

## 1. 概览

系统包含两条主线：

- ETF 分析与回测
- 股票分析与回测

两者共享以下设计原则：

- 历史数据按“交易日样本数”提供
- 优先使用本地 SQLite 缓存
- 在线数据失败时尽量降级而不是整体不可用
- 单策略回测与多因子组合回测并行存在

## 2. 架构

```text
CLI / Python / Flask API
        |
        v
业务编排层：etf_system.py / stock_system.py / etf_web.py
        |
        v
数据层：etf_data.py / stock_data.py
        |
        v
因子层：etf_factors.py / stock_factors.py
        |
        +--> 单策略层：strategies/ / stock_strategies/
        |
        +--> 组合层：factor_combo.py
        |
        v
回测与结果输出
```

## 3. 数据层设计

### 3.1 历史数据口径

- 对外 `days` 一律表示“最近 N 个交易日样本”
- 数据拉取时先扩成更长的自然日窗口
- 再按日期排序、去重、裁切成最后 N 条交易记录

这样可以保证：

- CLI、Python、Web 结果口径一致
- 文档中的样本数与实际回测长度一致
- 缓存命中与网络拉取行为一致

### 3.2 ETF 数据

ETF 数据层负责：

- 实时行情
- 历史行情
- ETF 名称与分类
- 本地缓存
- 产品属性补充信息，例如规模代理

### 3.3 股票数据

股票数据层负责：

- 实时行情
- 历史行情
- 股票名称与行业
- 基础面最佳努力获取：市值、流通市值、ROE
- 静态代码表兜底

## 4. 因子层设计

### 4.1 ETF 因子

ETF 因子偏重趋势、轮动和产品属性：

- 趋势类：`ma_alignment` `adx` `supertrend`
- 动量类：`roc` `rsi` `macd`
- 风险类：`market_fear` `volatility` `atr`
- 轮动类：`relative_strength`
- 产品类：`fund_size` `fund_size_percentile` `turnover_liquidity`

### 4.2 股票因子

股票因子分成三类：

- 技术类：`ma_alignment` `adx` `roc` `rsi` `macd`
- 量价类：`obv` `volume_ratio`
- 基础面与属性类：`market_cap` `market_cap_bucket` `roe` `industry`

### 4.3 因子缺失策略

- 公开接口拿不到的基础数据返回 `None`
- 组合评分时缺失值按 0 分处理
- API 结果原样暴露缺失值，方便前端解释

## 5. 多因子组合设计

`factor_combo.py` 负责：

- 维护 ETF / 股票组合模板
- 计算单因子时间序列
- 把因子转换成统一分数
- 汇总成 `composite_score`
- 生成买卖信号并执行组合回测

### 5.1 组合模板

ETF 默认模板：

- `etf_trend`
- `etf_trend_size`

股票默认模板：

- `stock_quality`
- `stock_quality_industry`

### 5.2 打分流程

1. 计算原始因子
2. 按因子类型转换为 `[-1, 1]` 区间分数
3. 乘以权重求和
4. 除以总权重得到综合得分
5. 综合得分高于买入阈值开仓，低于卖出阈值平仓

### 5.3 市场恐慌因子

`market_fear` 不是外部官方 VIX，而是内部风险代理：

- 波动率
- 滚动回撤
- 下跌收益强度

对于股票组合，默认使用 `510300` 的历史行情作为大盘风险代理。

## 6. 相对强弱设计

ETF 相对强弱分析使用：

- `510300`
- `510500`
- `512100`

当前支持窗口：

- `20`
- `60`
- `120`
- `250`

输出包含：

- 单基准超额收益
- 窗口汇总
- 轮动阶段
- 操作建议

## 7. 接口设计

### CLI

- ETF：`etf_system.py`
- 股票：`stock_system.py`
- 支持单策略与多因子组合两类入口

### Python

- `ETFAnalysisSystem.analyze_factor_combo`
- `StockAnalysisSystem.analyze_factor_combo`

### Flask API

- ETF 单策略：`/api/analysis/<code>`
- ETF 多因子：`/api/analysis/<code>?combo=...`
- 股票多因子：`/api/stock/analysis/<code>?combo=...`
- 组合模板：`/api/etf/combo-templates` `/api/stock/combo-templates`

## 8. 测试策略

重点覆盖：

- 交易日样本口径
- 250 日相对强弱
- 股票静态兜底
- ETF / 股票多因子组合输出
- 现有 CLI / API 回归
