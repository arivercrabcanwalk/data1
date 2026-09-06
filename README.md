# A 股全市场 1 分钟数据（2026 年 3–8 月）

本仓库按自然月整理本地全市场股票的 1 分钟数据，每个子包最多 12 个交易日。

每天只有一个精简 Parquet 文件 `minute1.parquet`，保留：

`code`、`datetime`、`open`、`high`、`low`、`close`、`volume`、`amount`

分钟线文件本身已经包含成交量和成交额，因此没有重复上传 `volume/minute1` 镜像层。

## 目录结构

```text
2026-03/part-01_12trading_days/date=YYYY-MM-DD/minute1.parquet
2026-03/part-02_10trading_days/date=YYYY-MM-DD/minute1.parquet
2026-04/part-01_12trading_days/date=YYYY-MM-DD/minute1.parquet
2026-04/part-02_09trading_days/date=YYYY-MM-DD/minute1.parquet
2026-05/part-01_12trading_days/date=YYYY-MM-DD/minute1.parquet
2026-05/part-02_06trading_days/date=YYYY-MM-DD/minute1.parquet
2026-06/part-01_12trading_days/date=YYYY-MM-DD/minute1.parquet
2026-06/part-02_09trading_days/date=YYYY-MM-DD/minute1.parquet
2026-07/part-01_12trading_days/date=YYYY-MM-DD/minute1.parquet
2026-07/part-02_11trading_days/date=YYYY-MM-DD/minute1.parquet
2026-08/part-01_12trading_days/date=YYYY-MM-DD/minute1.parquet
2026-08/part-02_09trading_days/date=YYYY-MM-DD/minute1.parquet
```

每个 part 目录下有 `daily_summary.csv` 和 `manifest.json`。每个日期文件均低于 50 MB。

## 每日股票状态与涨跌幅规则

每个日期目录还包含 `daily_stock_status.parquet`，与同目录的 `minute1.parquet` 通过 `code + date` 关联。它记录：

`market_board`、`security_status`、`is_st`、`is_star_st`、`is_new_listing_initial`、`is_suspended`、`suspension_status`、`price_limit_up_pct`、`price_limit_down_pct`、`price_limit_rule`。

状态表按股票日存储，不重复写入每一分钟；完整字段说明见 [`STATUS_FIELDS.md`](STATUS_FIELDS.md)，逐日索引见 [`status_manifest.json`](status_manifest.json)。`is_suspended` 为 `null` 时表示本地证据不足，不能当作未停牌。
