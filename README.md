# A股主升浪起涨点科学识别系统

五维共振（5-dimension resonance）量化评分引擎，用于识别 A 股个股「主升浪」起涨点。
评分引擎与数据源解耦，可在**完全离线**下运行回测与单元测试；接入 AkShare 后可跑实盘数据。

> ⚠️ 本项目为方法论的工程化实现，文中阈值为基于历史样本的经验值，**不构成任何投资建议**。

## 五维模型（spec 4.1）

| 维度 | 含义 | 权重 |
|---|---|---|
| d1 | 月线级别长期底部盘整 | 25% |
| d2 | 月线级别突破平台 | 20% |
| d3 | 日线级别走出上涨趋势 | 15% |
| d4 | 主力行为（4A 公开数据 12% + 4B 量价 13%） | 25% |
| d5 | 复合热点风口共振 | 15% |

- 每个子指标按 `0/3/6/10` 四档打分；维度得分 = 子指标加权平均（0~10）。
- `维度贡献 = 维度得分 × 权重 × 10`；`总分 = Σ维度贡献`，区间 `[0, 100]`。
- 分级：`A+ ≥85`、`A ≥75`、`B ≥60`、`C ≥45`、`D <45`。
- **七大一票否决**（任一触发 → 总分清零，归 D 级）：
  1. 月线盘整 < 6 个月
  2. 月线突破当月量能 < 盘整月均量 ×1.5
  3. 日线跌破 MA20 且 3 日未收复 / 日线 MACD 死叉
  4. ST/\*ST/退市风险/立案调查
  5. 所属板块 20 日涨幅后 1/3 且无政策催化
  6. 主力行为得分 < 满分 30%
  7. 突破后连续 3 日未创新高且量能萎缩 50%+

## 安装

```bash
pip install -r requirements.txt          # 核心引擎：pandas / numpy（+ pytest 用于测试）
pip install -U akshare                    # 可选：接入实盘数据（AkShareProvider）
```

> AkShare 的 `jsonpath` 依赖在部分 Python 3.11 镜像上无法编译；引擎不依赖它即可离线运行，
> 实盘数据按需安装即可。

## 架构

```
ashare_main_wave/
  config.py       集中管理所有权重/阈值/风控参数（spec 3,4,7；建议季度校准）
  utils.py        0/3/6/10 计分阶梯、加权、日期工具
  indicators.py   MA / MACD / 摆动结构 / 抬高高低点
  data.py         DataProvider 抽象 + AkShareProvider（懒加载）+ 列名归一化
  dimensions/     d1~d5 五个维度评分器（纯函数，易测）
  scoring.py      综合评分 + 分级 + 七大一票否决（修正了 spec 伪代码 rule6 的笔误）
  risk.py         仓位映射 / 三段止损 / 分段移动止盈 / 板块涨跌幅适配（spec 7）
  screener.py     周末全市场筛选漏斗（spec 9.1）
  cli.py          命令行入口
```

数据访问统一走 `DataProvider`，评分逻辑只消费归一化的 pandas 帧
（列：`date/open/close/high/low/volume/amount/turnover`），因此可用合成数据完整单测。

## 用法

### Python API

```python
from ashare_main_wave import score_stock, AkShareProvider

provider = AkShareProvider()                       # 需已安装 akshare
res = score_stock("300308", "20230430", provider, catalysts=2)
print(res.grade, res.total)
print(res.as_dict())
```

离线 / 回测：实现自定义 `DataProvider`（或子类化覆盖 `kline` 等方法）喂入历史帧即可。

### 命令行

```bash
python -m ashare_main_wave score 300308 --date 20230430 --catalysts 2
python -m ashare_main_wave screen --date 20230430 --universe 300308,688256,600150
```

### 风控

```python
from ashare_main_wave import position_plan, stop_loss_price, take_profit_action
position_plan("A+")                                  # 初始/上限仓位 + 加仓金字塔
stop_loss_price(100.0, cap_size="small", ma20=95.0)  # 三选一取最高（最紧）止损
take_profit_action(0.65, below_ma10=True)            # 分段移动止盈决策
```

## 测试

```bash
python -m pytest -q
python examples/demo.py        # 离线合成数据演示，打印评分与风控示例
```

测试全部使用合成数据，无需网络、无需 akshare。
