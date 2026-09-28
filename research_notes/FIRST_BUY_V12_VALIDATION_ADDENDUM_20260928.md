# V12 验证补充（2026-09-28）

## 1. 规则模块已经代码化
新增：
- backtest/first_buy_v12_overlay.py
- backtest/test_first_buy_v12_overlay.py

overlay 只实现 V12 的“新增通道”，不复制或替代 V10/V11 主干。这样主干可以继续冻结，新通道可以单独验证和回滚。

总语义保险：
- prior_streak 必须 >=4；
- 必须是 is_first_break；
- 当前日不能仍是涨停；
- is_secondary_break 必须为 false。

## 2. GitHub Actions 第三轮
run: 36383765167

结果：
- checkout / Python 环境 / syntax checks 全部通过；
- V12 overlay 边界测试 8/8 通过；
- focused refinement 再跑成功。

测试覆盖：
- 非第一断板拒绝；
- prior_streak<4 拒绝；
- HIGH_GAP_HEIGHT_5PLUS；
- 高开一字不可交易拒绝；
- HIGH_GAP_THEME_4 必须题材核心 + 触板；
- SUPER_HIGH_ABSORPTION；
- PANIC_REPAIR 必须 60 分钟内因果确认；
- TRUE_RED_6_7 的 D+1 低开拒绝；
- QUALIFIED_PROBE_ADD 只能给 V11 已合格 probe 加仓，不能生成新候选。

## 3. 第六通道探索：强收 + 次日低开
为了继续寻找 4–6 月漏票，额外审计了“首次断板日强收、D+1 中低开”结构。

正例与反例：
- 合锻智能 06-02：历史 H3 约 +18.9%，但题材 rank9，不是绝对核心；
- 株冶集团 04-22：prior leader / rank1，但 H3 只有约 +1.7%；
- 东方锆业 06-25：leader/rank1，H3 只有约 +1.3%；
- 通鼎互联 05-13：相似价格结构，H3 约 -9.4%；
- 福达合金 05-12：leader/rank1、followers9、ret20很高，仍然 H3 约 -3.1%；
- 百花医药 08-13：7板后正收益分歧，D+1低开，H3约 -0.8%。

结论：
- “观察日强收 + 次日低开”本身没有足够辨识力；
- 即使再叠加 leader/rank1，也不能稳定提纯；
- 因此不新增第六条通道，不为了月度数量牺牲胜率。

## 4. 当前收敛结论
V12 的合理升级边界暂时收敛为：
1. V10/V11 主干；
2. HIGH_GAP_HEIGHT_5PLUS；
3. HIGH_GAP_THEME_4；
4. SUPER_HIGH_ABSORPTION（experimental）；
5. PANIC_REPAIR_5PLUS（experimental）；
6. TRUE_RED_6_7_HEIGHT_OVERRIDE（experimental）；
7. QUALIFIED_PROBE_ADD（仅仓位升级，不生成候选）。

继续扩容的下一步不应再靠修改历史阈值，而应拿 2026-09-24 之后的全新交易日做真正未见验证，再决定 experimental 通道是否转正。
