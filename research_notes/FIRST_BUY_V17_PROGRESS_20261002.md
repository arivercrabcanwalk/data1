# V17 研究进度｜2026-10-02

## 已完成

- 建立独立分支 `research/first-buy-v17-structural-executor-20261002`。
- 重建 2025-09-01 至 2026-09-30 canonical 日线。
- 修复“amount 缺失导致涨停/连板链错误中断”的语义漏洞。
- 重新生成 PRIMARY + 30日 RECYCLE 生命周期。
- 将 Day1 拆成 FAILED_OPEN / PANIC / TOUCH_BREAK / NORMAL_RED / POSITIVE_BREAK。
- 建立 HOLD1 / IMPULSE3 / RECLAIM1 / REPAIR_HOLD3 / REPAIR_HOLD5 因果 Day2 执行器。
- 所有确认均使用完整分钟，成交统一使用下一分钟 open。
- 将 round-trip 研究成本提高到约 0.52%。
- 建立显式 PRICE_ONLY 对照，不在 amount/VWAP 缺失时偷偷补值。
- 建立最新 Day1 实时计划输出和自动一致性测试。

## 数据诊断

V17 canonical：
- 789669 行；
- 3016 只主板研究股票；
- 263 个交易日；
- 最新 2026-09-30。

2026-09-29 有 3010 行日线 amount 缺失，原始 1 分钟 amount 同样缺失。
因此 V17 保留价格/涨停/连板语义，但 VWAP 相关字段在真实缺失时保持缺失。

该修复纠正了雪龙集团：
9/23 一板 -> 9/24 二板 -> 9/28 三板 -> 9/29 四板 -> 9/30 Day1。
旧逻辑会因为 9/29 amount 缺失而错误把连板链打断。

## 当前样本空间

2026 年 lifecycle Day1 共 500 个：
- B3：232；
- B4P：180；
- R2：88。

Day1 分支：
- POSITIVE_BREAK：200；
- NORMAL_RED：157；
- PANIC：101；
- FAILED_OPEN：24；
- TOUCH_BREAK：18。

有完整 Day2 分钟覆盖的事件 424 个。

## 关键研究结论

generic repair 没有优势：
- RECLAIM1 全体胜率约 47%；
- HOLD1 全体约 50%；
- REPAIR_HOLD3/5 全体均不足 50%。

因此“只要重新站回昨收/VWAP就买”继续否定。

真正出现差异的是“Day1 分支 × 连板高度 × Day2 修复方式”。

当前 CORE 两条历史合计 19 笔：
- 胜率 78.95%；
- 平均 +6.20%；
- 最差 -2.64%；
- 历史亏损超过 3% 的比例 0%。

CORE 只包括：
1. 4+ NORMAL_RED + 09:45 前五分钟 Repair Hold；
2. 3板 PANIC + Day1 >=10亿成交额 + 09:45 前 Reclaim。

## 仍未通过的部分

自动 walk-forward 动态 selector 仍不合格。
这证明目前不能把 V17 描述成“机器学习已经泛化成功”。

Experimental 三条虽然历史统计较好，
但样本量或时间稳定性不足，继续等待真正新样本：
- TOUCH_3_REPAIR45；
- POS_4P_ABS_RECLAIM；
- FAILED_3_RECLAIM。

R2 二板再起继续进入生命周期观察池，但暂不自动买入。

## 最新 Day1

2026-09-30：
- 中新赛克：R2 recycle PANIC -> OBSERVE_ONLY；
- 雪龙集团：4板后 PANIC，但不属于当前 V17 已验证 PANIC 通道 -> PASS_V17。

这两个判断均来自当前冻结规则，不因看到后续结果修改。

[executed on device: zhanghaodeMacBook-Air-3.local (656495e0-4b4b-466c-a782-26cb2f66ad3e)]