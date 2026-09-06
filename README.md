# A 股全市场 1 分钟数据（2026 年 6–8 月）

本仓库按自然月整理本地全市场股票的 1 分钟数据，每个子包最多 12 个交易日。

## 字段

每天只有一个精简 Parquet 文件 `minute1.parquet`，保留：

`code`、`datetime`、`open`、`high`、`low`、`close`、`volume`、`amount`

由于分钟线文件本身已经包含成交量和成交额，没有重复上传 `volume/minute1` 镜像层，以减少冗余和仓库体积。

## 目录

```text
2026-06/part-01_12trading_days/date=YYYY-MM-DD/minute1.parquet
2026-06/part-02_09trading_days/date=YYYY-MM-DD/minute1.parquet
2026-07/part-01_12trading_days/date=YYYY-MM-DD/minute1.parquet
2026-07/part-02_11trading_days/date=YYYY-MM-DD/minute1.parquet
2026-08/part-01_12trading_days/date=YYYY-MM-DD/minute1.parquet
2026-08/part-02_09trading_days/date=YYYY-MM-DD/minute1.parquet
```

每个 part 目录下有 `daily_summary.csv` 和 `manifest.json`，记录行数、文件大小和字段。
