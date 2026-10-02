# 第一买点 V17 Structural Executor（2026-10-02）

> 状态：research candidate，尚未升级为生产规则。
> 边界：只研究一买。严禁混入 R50、龙头二买、二波项目。
> 所有 V17 胜率均为 2026 历史研究统计，不能当作未来胜率。

## 1. 为什么从 V16 升级

V16 的 30 日生命周期状态机保留，但原统一分类器废弃。
原因不是样本不足，而是把不同 Day1 / Day2 结构混成一个模型后，伪样本外表现接近无优势。

V17 将系统拆成四层：
1. Canonical 数据层；
2. 30 日生命周期 / Day1 事件层；
3. Day1 结构分类层；
4. Day2 因果执行器。

## 2. Canonical 数据修复

当前重建范围：2025-09-01 至 2026-09-30。
主板研究代码 3016 只，263 个交易日，789669 行。

发现 2026-09-29 原始日线有 3010 行 amount 缺失，原 1 分钟 amount 同样缺失。
V17 不伪造 amount/VWAP，也不再因为 amount 缺失把价格有效性判成失效。

因此：
- 涨停、连板、Day1 语义只依赖可验证价格与交易日连续性；
- VWAP 只在真实 amount 可用时计算；
- 缺失 VWAP 时只能走显式 PRICE_ONLY 研究路径，不能静默替代。

## 3. 生命周期

初始 >=3 连板进入 PRIMARY_ARMED。
第一次非涨停日 = primary Day1。
Day2 是唯一买入决策日。

从 Day2 起保留 30 个自然日观察：
- 30 日内重新 >=2 连板 -> RECYCLE_ARMED；
- 下一次断板生成 recycled Day1；
- 每个新 Day1 再开启自己的 30 日窗口；
- 30 日没有再武装则该生命周期失效。

R2 二板再起目前继续观察，但不属于 V17 买入通道。

## 4. Day1 分类器

五类互斥结构：
- FAILED_OPEN：Day1 开盘在涨停附近，最终收绿；
- PANIC：Day1 收跌 <= -8%；
- TOUCH_BREAK：盘中触及涨停、最终收绿；
- NORMAL_RED：普通收绿断板；
- POSITIVE_BREAK：断板但收红。

质量辅助标签：
- ABSORBED：close_loc >= 0.65 且 close/VWAP >= 0.99；
- MIXED：close_loc >= 0.35 且 close/VWAP >= 0.97；
- WEAK：其余。

高度桶：
- B4P：4 板及以上；
- B3：3 板；
- R2：生命周期里的二板再起。

## 5. Day2 因果执行

所有信号都必须先等当前分钟收盘完成判断，最早按下一分钟 open 成交。
这条规则禁止“用本分钟收盘确认，同时假设成交在本分钟收盘”。

主要动作：
- HOLD1：第一分钟完成强度确认，下一分钟开盘执行；
- IMPULSE3：深低开后前三分钟形成真实脉冲，第四分钟开盘执行；
- RECLAIM1：首次收复 Day1 close + Day2 open + cumulative VWAP，下一分钟执行；
- REPAIR_HOLD3：连续 3 个完整分钟保持修复，下一分钟执行；
- REPAIR_HOLD5：连续 5 个完整分钟保持修复，下一分钟执行。

当前完整回测统一使用约 0.52% round-trip 摩擦成本，
用于覆盖佣金、税费及约双边 20bp 级滑点假设。

## 6. V17 当前 CORE

CORE_RED_4P_REPAIR45：
- Day1 = NORMAL_RED；
- prior_streak >= 4；
- Day2 连续 5 个完整分钟保持在 Day1 close、Day2 open、累计 VWAP 之上；
- 修复必须在 09:45 前完成；
- 下一分钟 open 执行。

PANIC_3_LIQ_RECLAIM45：
- Day1 = PANIC；
- prior_streak = 3；
- Day1 成交额 >= 10 亿元；
- Day2 在 09:45 前首次完整收复 Day1 close、Day2 open、累计 VWAP；
- 下一分钟 open 执行。

## 7. Experimental 通道

以下保留研究，不与 CORE 等级混淆：
- TOUCH_3_REPAIR45；
- POS_4P_ABS_RECLAIM；
- FAILED_3_RECLAIM。

当前明确 PASS：
- R2 recycle 自动买入；
- 普通 3 板首阴统一买入；
- 所有 POSITIVE_BREAK 统一买入；
- 单纯“重新站回昨收/VWAP”且不看 Day1 分支与高度的 generic repair。

## 8. 当前历史统计

在 0.52% round-trip 成本后：
- CORE：19 笔，历史胜率 78.95%，均值 +6.20%，最差 -2.64%；
- Experimental：26 笔，历史胜率 69.23%，均值 +3.18%；
- 合并：45 笔，历史胜率 73.33%，均值 +4.46%，中位数 +3.46%，最差 -7.45%。

双倍摩擦压力下，合并均值仍约 +3.94%。
删除历史贡献最大的两只股票后，剩余 43 笔胜率约 72.09%，均值约 +3.88%。

这些结果仍然属于规则发现后的历史回放，不是独立未来样本。

## 9. 稳健性现状

通用 walk-forward 动态选动作仍未通过：
历史动态 selector 在真实后月只有接近零或负期望。
这意味着 V17 不能被包装成“机器已经自动学会”。

当前更值得继续验证的是透明的结构通道，
尤其是 CORE 两条；Experimental 必须等待新样本转正。

## 10. 生产晋级条件

V17 在满足以下条件前不得覆盖旧生产规则：
1. 新交易日完全未参与规则设计；
2. CORE / Experimental 分开统计；
3. 至少完成新的跨月样本；
4. 因果执行、涨跌停可成交性、T+1 约束全部通过；
5. 题材 PIT 字段验证完成；
6. 新样本不能靠新增股票代码、月份或个案阈值修规则。

[executed on device: zhanghaodeMacBook-Air-3.local (656495e0-4b4b-466c-a782-26cb2f66ad3e)]