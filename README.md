# worldcup-forecast

**简体中文** | [English](README-EN.md)

一个面向 FIFA 世界杯的结构化 + 贝叶斯预测器；同样重要的是，它也提供了一份
**诚实、经过严格验证**的说明：到底哪些方法真的能改进国家队比赛预测。

*本项目受 Joachim Klement 的结构化世界杯模型启发。*

![Joachim Klement——连续三届精准预测世界杯冠军（2014 德国、2018 法国、2022 阿根廷）](joachim-klement.jpg)

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![tests](https://img.shields.io/badge/tests-passing-brightgreen)

大多数世界杯模型会发布一份冠军榜和一个看起来很漂亮的准确率数字。这个项目则提供了
**滚动向前验证框架**和一份**发现台账**，用显著性检验说明哪些想法有效、哪些无效，
包括那些不太令人舒服的结论。核心结论本身也保持诚实：经过穷尽式搜索后，模型已经位于
*结构效率前沿*，而博彩市场非常难以击败。

---

## 2026 世界杯比赛预测结果（持续更新）

按双轨预测做的演示性下注记录：每场投注 $10,000，依模型预测概率分配到胜/平/负三个结果
（最小单位 $100），并以该场实际结果对应的赔率结算。

| 对阵 | 模型 胜/平/负 % | 分配 胜/平/负 | 实际 | 赔率 | 盈亏 |
|---|---|---|---|---|---|
| 墨西哥–南非 | 63/24/12 | 6400/2400/1200 | 胜 | 1.44 | −784 |
| 韩国–捷克 | 39/32/29 | 3900/3200/2900 | 胜 | 2.67 | +413 |
| 加拿大–波黑 | 54/28/18 | 5400/2800/1800 | 平 | 3.64 | +192 |
| 美国–巴拉圭 | 61/25/14 | 6100/2500/1400 | 胜 | 2.15 | +3,115 |
| 卡塔尔–瑞士 | 13/26/61 | 1300/2600/6100 | 平 | 8.00 | +10,800 |
| 巴西–摩洛哥 | 42/31/28 | 4100/3100/2800 | 平 | 3.92 | +2,152 |
| 海地–苏格兰 | 19/30/51 | 1900/3000/5100 | 客胜 | 1.55 | −2,095 |
| 澳大利亚–土耳其 | 26/31/43 | 2600/3100/4300 | 胜 | 5.71 | +4,846 |
| **合计（8 场，本金 $80,000）** | | | | | **+18,639** |

> ⚠️ 目前仅 8 场样本，收益高度集中于卡塔尔–瑞士一场平局冷门（@8.00，+$10,800，约占六成）；
> 对着庄家约 5.9% 的水位，该策略整体期望可能为负。

## 它能做什么

- **球队强度**：基于冻结的赛前 FIFA 快照、Transfermarkt 阵容身价、无泄漏 Elo，
  并用 Klement 风格的结构先验包裹起来（GDP、人口、气候、主场优势、足球文化）。
- **比赛模型**：带部分池化的分层贝叶斯 Poisson 模型（PyMC）——数据丰富的球队更多由
  比赛结果牵引，数据稀缺的球队则收缩到结构先验。
- **赛事模拟**：对官方 2026 赛制进行 Monte-Carlo 模拟（12 个小组、8 个成绩最好的
  小组第三、73-104 淘汰赛映射），输出冠军和阶段概率。
- **验证**：样本外滚动向前评分（log-loss / RPS / Brier），并使用配对 bootstrap
  显著性检验；从不只用少量比赛来判断模型。
- **市场基准**：Polymarket 和传统博彩公司赔率仅用于比较。根据设计中的独立性原则，
  **市场数据绝不会作为模型输入**。

## 快速开始

```bash
git clone <repo-url> && cd worldcup-forecast
pip install -e .                 # 或者：make install

wcforecast forecast              # 2026 冠军概率（双轨）
wcforecast predict Brazil Morocco
wcforecast validate              # 带显著性检验的样本外评分表
ODDS_API_KEY=xxxx wcforecast odds   # 实时博彩公司共识（可选，需要免费 key）
```

需要 Python >= 3.10。首次运行会自动下载公开比赛和排名数据集；小型的冻结 2026 输入
已随项目放在 `data/snapshots/` 中。

## 示例输出

```
2026 World Cup — champion probability (Monte Carlo)

  team                      accuracy %  independent %
  France                          13.8           13.3
  Spain                           12.9           11.4
  England                         11.4           11.3
  Argentina                        8.6            9.9
  United States                    6.7            7.7
  ...
```

项目会并排报告两条轨道：
- **accuracy**：完整模型（FIFA + 阵容身价锚点 + 比赛数据）；
- **independent**：仅结构化（“Klement”）先验，不含市场数据，且可解释。

二者之间的差异是*需要解释的特征*，不是需要隐藏的错误。

## 诚实发现：这个项目最独特的部分

每个候选想法都在一个**锁定的 305 场比赛窗口（2024-2026）**上做过样本外测试，
并使用配对 bootstrap 检验显著性。完整台账见：
[`docs/FINDINGS.md`](docs/FINDINGS.md)。

| 想法 | 结论 |
|---|---|
| 无泄漏 Elo（K=40）+ 结构先验 | ✅ 稳健基线 |
| 温度校准 + 中立场平局提升 | ✅ 小幅但显著 |
| 模型平均（Bayesian + Elo-logit） | ✅ 降低方差 |
| 阵容身价作为强度锚点 | ✅ 有助于 2026 预测（在强弱悬殊比赛上，与敏锐市场的距离约缩小 30%） |
| FIFA vs Elo 锚点 | ➖ 平手（FIFA 的价值在于数据质量，而不是额外信号） |
| 额外特征（状态、休息、洲际足联） | ➖ 没有显著收益 |
| 历史交手记录 | ❌ 控制强度后几乎为零 |
| Dixon-Coles 低比分修正 | ❌ 在这份数据上 rho 约为 0 |
| 梯度提升（LightGBM）集成与 isotonic 校准 | ❌ 对稀疏数据过拟合，显著更差 |

**一句话结论：** 简单、低方差的改动有帮助；灵活、复杂的方法容易在国家队数据上过拟合。
模型已经到达结构前沿——单场世界杯比赛结果大约有一半来自运气，而市场很难击败。
这个结论本身就是产品的一部分。

## 如何借鉴 Klement 的结构化思路

Joachim Klement 的核心观点是：国家队实力主要是**结构性**的——不靠球星名气，而是看缓慢
变化的国家变量、赛事自身的对阵路径，再加上一定的运气。本项目正是把这些输入原封不动地
作为一份**无泄漏的结构先验**（即 `independent` 轨道），再让比赛结果通过部分池化去修正
它。他模型所看的每一项数据，都对应到 `ratings.structural_index` / `simulate.py` 里的一个
具体项：

| Klement 看的数据 | 本项目的实现方式 |
|---|---|
| FIFA 排名 | 标准化后的 FIFA 积分作为当前实力**锚点**，赛前冻结（2026 赛事再与 Transfermarkt 阵容身价混合）。 |
| 人口规模 | `log10(人口)` 的「人才库规模」项，外加人口 × 足球文化的交互项——人口红利只在真正踢球的国家才生效。 |
| 人均 GDP | `log10(GDP)` 的**倒 U 形**（经济发展有帮助，但到一定程度后趋平）；系数靠先验推理设定，而非拟合。 |
| 足球在社会中的重要程度 | 一个无泄漏的**足球文化**代理：一半看历届世界杯出场、一半看长期 Elo。 |
| 气候和比赛环境 | 年均气温的**倒 U 形**（偏离温带最优值则扣分），外加东道主加成。 |
| 分组及淘汰赛路径 | 对**官方 2026 赛制**做 Monte-Carlo——真实抽签：12 个小组 → 前两名 + 8 个最佳小组第三（按资格匹配）→ 32 强……决赛。 |
| 一定程度的随机因素 | 每场比赛的 Poisson 进球噪声、淘汰赛里约 ⅓ 强度的加时与点球掷硬币，以及每次模拟都抽取一组全新的后验样本。 |

**在 Klement 之上更进一步。** 这份结构先验是一个分层贝叶斯 Poisson 模型的**收缩目标**——
数据丰富的球队向自己的比赛结果靠拢，数据稀缺的球队则留在先验附近——并额外引入无泄漏
Elo 与阵容身价作为锚点。所有系数都由先验推理设定，绝不在仅约 36 行的世界杯历史上拟合
（那样会得出诸如「GDP 系数为负」之类的荒谬结果）。而且每一项改动都在样本外用显著性检验
验证过——见上文的「诚实发现」。

## 工作原理

完整方法论和设计取舍见 [`docs/DESIGN.md`](docs/DESIGN.md)
（为什么使用部分池化、为什么市场只作为基准、如何避免数据泄漏）。

## 项目结构

```
worldcup-forecast/
├── src/wcforecast/
│   ├── teams.py        # 48 支球队、官方抽签、洲际足联
│   ├── data.py         # 无泄漏加载器（martj42、FIFA、阵容、World Bank）
│   ├── ratings.py      # Elo + 结构化（Klement）先验指数
│   ├── model.py        # 分层贝叶斯 Poisson（PyMC）
│   ├── predict.py      # 比分网格 1X2 + 校准
│   ├── simulate.py     # 官方赛程结构 + Monte Carlo 冠军赔率
│   ├── validate.py     # 滚动向前样本外框架 + 指标 + 显著性
│   ├── markets.py      # Polymarket + 博彩公司赔率（仅作基准）
│   └── cli.py          # `wcforecast` 命令行接口
├── tests/              # 快速、离线的单元测试
├── docs/               # DESIGN.md（方法论）+ FINDINGS.md（已验证台账）
└── data/               # 冻结的 2026 快照（已提交）+ 缓存（git 忽略）
```

## 作为库使用

```python
from wcforecast import data, ratings, model, simulate, predict

results = data.load_results()
s = ratings.structural_index("2026-06-11", results,
                             fifa=data.load_fifa_snapshot(),
                             squad=data.load_squad_values())
matches = data.load_matches(start="2006-01-01", cutoff="2026-06-11")
m = model.fit(matches, s, weights=ratings.recency_weights(matches, "2026-06-11"),
              dixon_coles=True)

simulate.champion_probabilities(m, s, n_sims=20000).head()
predict.calibrate(m.match_probs("Brazil", "Morocco", home_advantage=0.0))
```

## 设计原则：独立于市场

模型从不读取博彩赔率（无论是作为特征、校准目标，还是集成成员）。它的目的在于提供一个
*独立、可解释的结构化信号*；市场只用于基准比较。击败市场**不是**目标——验证结果也确认，
这件事本来就非常困难。

## 局限

- 单场比赛预测的上限较低（很多结果来自运气）；详见 FINDINGS。
- 阵容身价和 FIFA 快照都是单一的 2026 快照——用于向前预测，而不是历史回测
  （因为没有可用历史）。
- 免费博彩公司赔率层级只覆盖实时赔率，因此用于*历史*验证的市场基准只能比较结果，
  不能比较历史价格。

## 许可证

MIT — 见 [`LICENSE`](LICENSE)。

## 致谢

受 Joachim Klement 的结构化世界杯模型启发。数据来源：martj42/international_results、
Dato-Futbol/fifa-ranking、FIFA、Transfermarkt、World Bank、Polymarket、The Odds API。
