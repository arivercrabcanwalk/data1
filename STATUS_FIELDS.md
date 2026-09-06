# 每日股票状态表

`daily_stock_status.parquet` 与同日期的 `minute1.parquet` 并列。
每行是一个 `code + date`，因此不会把板块、ST、停牌等重复写入每一分钟。

- `market_board`：沪市主板、深市主板、创业板、科创板、北交所。
- `security_status`：*ST、ST、新股上市初期、正常股票、状态未知。
- `is_st` / `is_star_st` / `is_new_listing_initial`：可直接筛选的布尔字段。
- `is_suspended`：True=发现停牌事件，False=未发现，null=证据不足；同时看 `suspension_status`。
- `price_limit_up_pct` / `price_limit_down_pct`：日涨跌幅参考限制；新股无日涨跌幅时为空。
- `status_source`、`status_confidence`、`status_notes`：记录历史底稿与近似推断边界。

注意：本地保留的历史底稿在 2026-07-30 前可提供逐日 `is_st`，但没有保留当日原始名称，故 ST 与 *ST 的历史细分会明确标注为当前名称近似；2026-07-31 之后同样采用近似。停牌源覆盖到 2026-08-07，之后无分钟线且无停牌事件的股票保留为 unknown。
