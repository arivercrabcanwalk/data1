# July Knowledge Layer 数据字典

## `market_daily.csv`
仅由 GitHub `2026-07` 一分钟行情和当日状态表重建。包含全市场上涨/下跌家数、红盘率、中位收益、±5%家数、成交额、非ST收盘涨停、触及涨停、炸板代理、收盘跌停等。它是市场事实主表。

## `cycle_daily.csv`
只根据截至当日收盘的市场事实，以冻结规则给出 `冰点 / 修复 / 升温 / 高潮 / 分歧/震荡 / 退潮`。任何 T 日状态最早 T+1 使用。

## `timestamped_theme_evidence.csv`
23个交易日逐日、带公开发布时间的当日复盘/新闻证据。保存原始题材标签、规范化题材、来源、URL、发布时间、当日已观察到的市场语境。只提取当日已发生事实，不提取文章中的明日预测/荐股。

## `theme_membership.csv`
历史涨停榜的逐日股票-题材映射，必须保存6位股票代码、原始题材、规范化题材、来源封板时间和风险标记。该表是结构性历史归因，不单独证明信息在当日盘中已发布；进入交易候选前必须由点时题材证据和 GitHub 行情共同确认。

## `board_ladder.csv`
逐日连板梯队，记录代码、名称、板数和来源时间。用于梯队/高度事实，不直接等同于“总龙头”。

## `theme_rank_daily.csv`
题材原始指标与冻结后的 ThemeScore。核心字段：成员数量、红盘率、中位收益、10:00强度、成交额占比、最高板、梯队层数、早盘触板比例、3/5日持续性、昨日同题材跟随表现、点时语义确认、来源票数、题材分数和题材排名。

## `leader_role_candidates.csv`
个股地位候选。包含题材排名、板数、首次实际触板时间、成交额、当日收益、5/10/20分钟后同题材跟随数量、LeadershipScore、弹性标志，以及候选角色：`theme_leader / capacity_core / elastic_core / supplement / switch_pioneer / follower`。另有 `total_leader_candidate`，但它只是系统候选，不是事后封神标签。

## `profit_loss_effect.csv`
昨日涨停/连板人群在今日的平均收益、胜率、大亏比例等，用于衡量短线赚钱与亏钱效应。

## `catalyst_timeline.csv`
题材催化与当日公开语境，保留来源和发布时间；无法证明点时可用的催化不得进入当日交易。

## `daily_review_book.csv`
每天收盘后的结构化复盘卡：市场周期、广度、中位收益、前三高置信题材、总龙候选、昨日强势群体赚钱效应、当日结论。字段 `formal_trade_decision=false` 表示知识层阶段不产生收益交易。

## `expectation_book.csv`
T 收盘写、T+1生效的预期卡：市场周期、前三题材、核心候选、允许研究的Playbook、确认条件、失效条件。必须满足 `effective_date > asof_date` 且 `uses_future_data=false`。

## `playbook_catalog.csv`
正式回测前冻结的 P1-P6 模式定义：前置环境、触发、禁止、失效。

## `rule_change_log.csv`
因果规则升级日志。任何未来迭代必须写 `decision_date / effective_date / evidence_before_change / old_rule / new_rule / reason / applies_retroactively`。`applies_retroactively` 永远必须为 false。

## `source_audit.csv`
逐日来源覆盖与冲突审计。不同网站的涨停/题材口径不做静默合并；GitHub负责价格事实，网页负责语义证据。

## `knowledge_manifest_v2.json`
最终准入门禁。20项规划和反未来测试全部通过后，`ready_for_formal_backtest=true` 才允许创建正式收益回测。