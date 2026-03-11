# A股ETF中线分析回测系统

面向中线波段策略（持股周期 3 周～3 个月），支持 1394 只 ETF 的数据获取、趋势因子分析、策略回测和投资建议生成。

---

## 安装

```bash
pip install requests pandas numpy jinja2 akshare
```

---

## 关于数据下载

### 不需要提前下载到本地

本系统采用**按需实时拉取 + 自动本地缓存**模式。每次分析时自动从 akshare 获取数据并写入本地 SQLite（`etf_cache.db`），下次相同请求直接读本地，无需重复联网。

### 各场景数据获取说明

| 场景 | 首次（网络拉取） | 第二次起（本地缓存） |
|------|----------------|-------------------|
| 单只 ETF 实时行情 | < 1 秒 | < 0.1 秒 |
| 单只 ETF 历史 30 日 | 1～3 秒 | < 0.01 秒 |
| 单只 ETF 完整分析（1500 日） | 3～8 秒 | < 0.1 秒 |
| 批量下载所有宽基（1159 只 × 1500 日） | **约 2～3 小时** | 每日增量更新 < 5 分钟 |

### 首次批量下载（推荐）

第一次使用建议提前批量下载，之后只做每日增量更新：

```bash
# 下载所有宽基 ETF 的 1500 日历史数据（约 2～3 小时）
python3 etf_system.py --download --days 1500

# 之后每日运行一次，增量补全最新数据（< 5 分钟）
python3 etf_system.py --update

# 查看本地缓存情况
python3 etf_system.py --cache-stats
```

### 回测需要多长时间的历史数据？

中线策略建议：

```
--days 750    # 约 3 年，最低要求（因子预热 + 足够样本）
--days 1500   # 约 6 年，推荐（覆盖完整牛熊周期）
--days 2500   # 约 10 年，完整回测（含2015年牛市和2018年熊市）
```

> 说明：部分因子需要预热期（Ichimoku 需要 52 日、ADX/Supertrend 需要 14～30 日），
> `--days` 越短，有效回测样本越少，结果可信度越低。

---

## 快速开始

### 1. 查看 ETF 列表

```bash
# 查看静态代码库中的 ETF（80 只）
python3 etf_system.py --list

# 从 akshare 刷新全量列表并统计
python3 etf_system.py --refresh
```

输出示例：
```
akshare ETF总量: 1394 只
  宽基（510/512/56/588/515/516/159 前缀）: 1159 只
  其他: 235 只
```

### 2. 查询实时行情

```bash
python3 etf_system.py --quote --code 510300
```

输出：
```
沪深300ETF华泰柏瑞 (510300) - 最新价: 4.683, 涨跌幅: 1.17%
```

### 3. 单只 ETF 完整分析

```bash
# 默认策略（RSI），回测 500 日
python3 etf_system.py --code 510300

# 指定策略
python3 etf_system.py --code 510300 --strategy macd
python3 etf_system.py --code 510300 --strategy trend_filter_macd
python3 etf_system.py --code 510300 --strategy supertrend_follow
python3 etf_system.py --code 510300 --strategy donchian_breakout
python3 etf_system.py --code 510300 --strategy ma_cross
python3 etf_system.py --code 510300 --strategy bollinger

# 指定回测天数（推荐 1500 日以上）
python3 etf_system.py --code 510300 --strategy macd --days 1500
```

### 4. Python 代码调用

```python
from etf_system import ETFAnalysisSystem
from etf_data import ETFCodeMapper

system = ETFAnalysisSystem()

# 刷新全量 ETF 列表（可选，不调用则使用静态列表）
ETFCodeMapper.load_from_akshare()

# 完整分析（回测 + 因子 + 建议）
result = system.full_analysis("510300", strategy="macd", params={})

# 只跑回测
result = system.run_backtest("510300", "rsi", days=1500)
print(f"总收益率: {result.total_return*100:.2f}%")
print(f"年化收益率: {result.annualized_return*100:.2f}%")
print(f"夏普比率: {result.sharpe_ratio:.2f}")
print(f"最大回撤: {result.max_drawdown*100:.2f}%")

# 单独计算因子
from etf_factors import ETFFactorCalculator
from etf_data import ETFDataFetcher

fetcher = ETFDataFetcher()
df = fetcher.get_historical_data("510300", days=200)

calc = ETFFactorCalculator()
adx = calc.calculate(df, "adx", period=14)
print(f"ADX: {adx['adx'].iloc[-1]:.1f}")

st = calc.calculate(df, "supertrend", period=10, multiplier=3.0)
print(f"Supertrend 方向: {'多头' if st['trend'].iloc[-1] == 1 else '空头'}")

ma_align = calc.calculate(df, "ma_alignment", periods=[9, 21, 50])
print(f"均线排列得分: {ma_align.iloc[-1]:.2f}")  # -1(空头) ~ +1(多头)
```

---

## 回测策略说明

| 策略名 | 信号逻辑 | 推荐参数 | 适合行情 |
|--------|---------|---------|---------|
| `rsi` | RSI < 超卖阈值买入，> 超买阈值卖出 | period=21, oversold=30, overbought=70 | 震荡 + 趋势 |
| `macd` | MACD 线上穿信号线买入，下穿卖出 | fast=12, slow=26, signal=9 | 趋势行情 |
| `trend_filter_macd` | 仅在 ADX + 均线排列确认趋势后，MACD 金叉买入/死叉卖出 | fast=12, slow=26, signal=9, adx_min=20, ma_alignment_min=0.3 | 中线趋势行情 |
| `supertrend_follow` | Supertrend 翻多买入，翻空或动量转弱卖出 | period=10, multiplier=3.0, roc_period=20, roc_min=0 | 趋势延续行情 |
| `donchian_breakout` | 突破 Donchian 上轨买入，跌回中轨/下轨卖出 | window=20, adx_period=14, adx_min=20 | 突破启动行情 |
| `ma_cross` | 短均线上穿长均线买入，下穿卖出 | short_period=9, long_period=21 | 趋势行情 |
| `bollinger` | 价格触及下轨买入，触及上轨卖出 | window=20, num_std=2 | 震荡行情 |

**策略参数自定义示例：**

```python
# RSI 中线参数（21日周期）
result = system.run_backtest("510300", "rsi", {
    "period": 21,
    "oversold": 35,
    "overbought": 65,
}, days=1500)

# 均线交叉（9/21 日，对应中线用法）
result = system.run_backtest("510300", "ma_cross", {
    "short_period": 9,
    "long_period": 21,
}, days=1500)

# 趋势过滤 MACD：只有趋势成立时才允许 MACD 入场
result = system.run_backtest("510300", "trend_filter_macd", {
    "fast": 12,
    "slow": 26,
    "signal": 9,
    "adx_min": 20,
    "ma_alignment_min": 0.3,
}, days=1500)

# Supertrend 趋势跟随：适合有持续趋势的宽基 ETF
result = system.run_backtest("510300", "supertrend_follow", {
    "period": 10,
    "multiplier": 3.0,
    "roc_period": 20,
    "roc_min": 0,
}, days=1500)

# Donchian 通道突破：适合中期突破启动
result = system.run_backtest("510300", "donchian_breakout", {
    "window": 20,
    "adx_period": 14,
    "adx_min": 20,
}, days=1500)
```

**新策略使用建议：**

| 策略名 | 适合 ETF | 建议观察点 | 不适合场景 |
|--------|----------|-----------|-----------|
| `trend_filter_macd` | 沪深300、中证500、红利等宽基 | ADX 持续 > 20，均线排列转多 | 横盘震荡、频繁假突破 |
| `supertrend_follow` | 有持续上行趋势的宽基/行业 ETF | 趋势翻多后能否继续站稳 | 高位震荡、急涨急跌切换 |
| `donchian_breakout` | 中期突破型 ETF | 是否放量突破近 20 日高点 | 无趋势、低波动箱体 |

---

## 回测结果字段说明

```python
result.total_return       # 总收益率，如 0.32 = 32%
result.annualized_return  # 年化收益率（复利换算）
result.sharpe_ratio       # 夏普比率（> 1 优秀，> 2 极好，< 0 不可用）
result.max_drawdown       # 最大回撤（负数，如 -0.18 = 最深亏损 18%）
result.win_rate           # 胜率（完成的交易中盈利比例）
result.profit_factor      # 盈亏比（盈利总额 / 亏损总额，> 1.5 较好）
result.total_trades       # 总交易次数
result.winning_trades     # 盈利次数
result.losing_trades      # 亏损次数
result.equity_curve       # 资金曲线 DataFrame（date, equity 两列）
result.trades             # 每笔交易记录列表
```

**如何评价回测结果（中线策略参考标准）：**

| 指标 | 差 | 一般 | 好 | 优秀 |
|------|----|----|----|----|
| 年化收益率 | < 5% | 5～10% | 10～20% | > 20% |
| 夏普比率 | < 0.5 | 0.5～1.0 | 1.0～2.0 | > 2.0 |
| 最大回撤 | > 30% | 20～30% | 10～20% | < 10% |
| 胜率 | < 40% | 40～50% | 50～60% | > 60% |
| 盈亏比 | < 1.0 | 1.0～1.5 | 1.5～2.5 | > 2.5 |

> 注意：高胜率 + 低盈亏比 ≠ 好策略；中线策略更关注盈亏比，胜率 45% + 盈亏比 2.5 优于胜率 60% + 盈亏比 1.1。

---

## 趋势因子说明（中线核心）

| 因子 | 调用方式 | 输出 | 中线解读 |
|------|---------|------|---------|
| `adx` | `calc.calculate(df, 'adx', period=14)` | DataFrame: adx, di_plus, di_minus | ADX > 25 趋势成立；< 20 震荡，慎用趋势信号 |
| `supertrend` | `calc.calculate(df, 'supertrend', period=10, multiplier=3.0)` | DataFrame: supertrend, trend(+1/-1) | trend=+1 持有多头；trend=-1 离场观望 |
| `ma_alignment` | `calc.calculate(df, 'ma_alignment', periods=[9,21,50])` | Series: -1.0 ～ +1.0 | ≥ 0.8 完全多头排列；≤ -0.8 完全空头排列 |
| `ichimoku` | `calc.calculate(df, 'ichimoku')` | DataFrame: 5 条线 + price_vs_cloud | price_vs_cloud=+1 在云上（强势）；-1 在云下（弱势） |
| `donchian_channel` | `calc.calculate(df, 'donchian_channel', window=20)` | DataFrame: upper, lower, middle, position | position > 0.8 突破上轨（趋势启动）；< 0.2 触及下轨 |
| `aroon` | `calc.calculate(df, 'aroon', period=25)` | DataFrame: aroon_up, aroon_down, oscillator | Up > 70 & Down < 30：上升趋势确立 |
| `linear_regression_slope` | `calc.calculate(df, 'linear_regression_slope', window=21)` | Series | > 0 向上趋势；< 0 向下；绝对值越大趋势越陡 |
| `roc` | `calc.calculate(df, 'roc', period=20)` | Series | 20日变化率 > 0 中期动量向上；60日变化率更稳 |

---

## 投资建议输出说明

```
## 🚀 强烈买入            ← 信号：强烈买入/买入/持有/卖出/强烈卖出
综合评分: 87.3/100        ← 0～100 分，基于回测结果 + 5 个实时因子
置信度: 86.7%             ← 基于评分高低换算
目标价: 5.15元            ← 当前价 × 1.1（仅在总收益率 > 0 时给出）
止损价: 4.22元            ← 当前价 × 0.9

### 建议理由
- 策略历史收益表现良好，总收益率为 23.40%
- 策略胜率 62.5% 超过 50%
- 9/21/50 均线完全多头排列，中线趋势明确向上
- ADX=31.2 趋势强度充分，中线信号可信度高
- Supertrend 指向多头，建议持有或做多
- 价格高于21日均线 6.0%，短期趋势向上

### 风险提示
- 市场有风险，投资需谨慎
- 过往业绩不代表未来表现
```

**评分构成（满分 100）：**

| 评分项 | 分值范围 | 说明 |
|--------|---------|------|
| 基础分 | 50 | 起始分 |
| 总收益率 | ±20 | 正收益最多加 20 分 |
| 夏普比率 | ±10 | > 1 加 10，< 0 扣 10 |
| 最大回撤 | ±10 | < 10% 加 10，> 25% 扣 10 |
| 胜率 | ±5 | > 50% 加 5 |
| RSI(21) | ±8 | 超卖加分，超买扣分 |
| 趋势强度 | ±5 | 偏离均线幅度 |
| 均线排列 | ±10 | 9/21/50 多头/空头排列 |
| ADX | ±5 | 趋势强时加分，震荡扣分 |
| Supertrend | ±8 | 多头方向 +8，空头方向 -8 |

---

## 文件结构

```
etf_system.py       主入口（CLI + ETFAnalysisSystem 类）
etf_data.py         数据获取（akshare → Yahoo → 东方财富 → 腾讯）
etf_factors.py      24 个因子（收益/风险/动量/趋势）
etf_backtest.py     回测引擎（4 种策略，含手续费 0.03% + 滑点 0.01%）
etf_advisor.py      投资建议（9 项评分模型）
all_etf_codes.py    静态 ETF 代码字典（80 只，akshare 未加载时兜底）
```

---

## 可用因子完整列表（24 个）

```python
from etf_factors import ETFFactorCalculator
print(ETFFactorCalculator().get_available_factors())

# 收益类（4）
['cumulative_return', 'annualized_return', 'monthly_return', 'roc']

# 风险类（4）
['volatility', 'max_drawdown', 'sharpe_ratio', 'atr']

# 动量类（3）
['rsi', 'macd', 'bollinger_bands']

# 趋势类（13，中线核心）
['ma', 'ema', 'dema', 'ma_cross', 'ma_alignment',
 'trend_strength', 'linear_regression_slope',
 'adx', 'supertrend', 'donchian_channel',
 'keltner_channel', 'aroon', 'ichimoku']
```
