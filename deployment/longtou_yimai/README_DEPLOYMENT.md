# 龙头一买｜腾讯云生产部署验收（2026-10-08）

## 状态：代码及定时器定义已部署，生产信号明确关闭

**唯一目标服务器**：`ssh lighthouse` / `ubuntu@120.53.23.29` / `VM-0-7-ubuntu`。  
**SSH只读验收**：`SSH_OK`，用户`ubuntu`，主机名`VM-0-7-ubuntu`，独立初次命令退出码0。没有读取、打印或复制 `~/.ssh/id_ed25519`，没有连接旧服务器。

独立程序路径：`/home/ubuntu/longtou-yimai`。

## 当前严重生产阻塞项

内核在2026-10-08持续记录：
- `EXT4-fs error (device vdb)`
- `Journal has aborted`
- `Remounting filesystem read-only`
- `/data`目录`Input/output error`
- `/dev/vdb`与`/sys/block/vdb`**不存在**，即磁盘设备本身目前没有呈现给操作系统。

**不可以把`df`能显示196GB、40%使用率当成磁盘健康**；这是异常挂载残留。

没有执行磁盘修复、`fsck`、重新挂载、卸载、分区、格式化、云盘重挂、重启或停止任何业务服务。建议通过腾讯云 Lighthouse 控制台与技术支持检查数据云盘挂载关系、快照、实例事件。执行高风险修复前先做可用快照或外部备份并明确维护窗口。

程序强制 fail-closed：
- 磁盘设备缺失直接退出；
- 缺少上一交易日的完整Day1快照直接退出；
- 历史日线日期缺口直接退出；
- 开盘数据源日期错误、时间戳过期、量价异常直接PASS；
- 分钟前缀缺失或VWAP不可核验直接PASS；
- 两个独立实时报价源价差过大、不响应时不发送信号；
- 买入时价格锁涨停直接PASS；
- 信号不能从未来分钟补录；
- 单一事件信号只记录一次并避免重复发送。

## 已部署内容

- `longtou_core.py`：本地历史研究中通过时间因果审核的V22 7条入口规则，SHA256绑定；
- `longtou_runtime.py`：独立EOD Day1捕捉、次日盘中09:25-09:47观察、源时间/量价/连续性/双源校验、飞书消息；
- `seed/daily_history_to_20260930.parquet`：789669条历史日线，3016个股票代码，截止9/30；
- `seed/event_history_to_20260930.parquet`：729条历史事件，用来识别首次断板与再次启动；
- `seed/manifest.json`：文件源与SHA256封存；
- `config/settings.json`：配置标签**龙头一买**、V22_STRICT、`real_signals_armed=false`；
- `private/feishu-route.env`：沿用R50当前运行进程的飞书频道和接收对象，权限600；从未打印对象ID，也**没有发送任何测试消息**；
- `systemd/longtou-yimai-{day1,live}.{service,timer}`：隔离调度定义；
- `logs/`、`state/`、`backups/`：独立日志、状态与安装前备份目录。

Systemd 中已安装4个**新的** unit：
- `longtou-yimai-day1.timer`：计划交易日15:12构建Day1；
- `longtou-yimai-live.timer`：计划交易日09:25启动，09:30—09:46实时扫描Day2；
- 对应两个独立service，无实盘下单行为。

**四个unit没有与旧名称冲突，配置安装前已存独立备份与原有unit状态。它们目前均不运行，两个timer保持disabled。**
启用需要先修复数据盘、补足缺失交易日原始数据、复测生产行情链路，并经过人工确认。

## 外部行情源及限制

已只读确认复用原R50实时行情能力：
- 主板目标池3170量级，一次实测universe为3169，报价3077（97.1%）；
- 腾讯实时报价正常，`600000`等样本价格和时间戳正常；
- 腾讯分钟数据可读，测试样本截至当时14:32有214条含集合竞价分钟；
- 行情读取逻辑不修改R50现有代码、不影响其他服务；
- 入场信号使用已完成分钟+检测时实时参考价。**不是交易所保证可成交的下一分钟开盘价格**。
- 新系统使用腾讯累计分钟量额构造分钟价、成交额差分和VWAP，并不是原回测的完整OHLC行情；因此生产一致性和滑点还需要跨源认证。不要把历史84.62%当成上线保障。

## 已完成的无通知验收

1. SSH返回`SSH_OK / ubuntu / VM-0-7-ubuntu`，连接退出码0。
2. Python运行环境复用现有`/opt/liangun/.venv/bin/python`，未执行pip安装。
3. 30MB冻结历史种子已同步到独立目录，SHA256验证全部通过。
4. 代码`py_compile`通过。
5. 合成分钟买点信号正确只读取前三个完成分钟，M4高低价缺失不影响因果判断。
6. M2分钟缺失时不错误触发M3买点。
7. 历史2026-09-30 Day1重新生成事件与冻结历史中**达到3板及以上**的目标事件一致；2板R2观察不作为本项目交易。
8. `selftest`通过。
9. 飞书路由存在且受限权限600，但没有触发任何消息。
10. 数据盘损坏时`watch`明确返回退出码**4**，没有信号发出。
11. 安装Systemd后，R50、连板、VWAP和Hermes旧服务依然active，旧服务没有重启。
12. `holdings.json`没有被读取、修改、备份或覆盖，本轮无持仓文件操作。

## 手动运维（禁用状态，未来参考）

只读状态验收：
```bash
ssh lighthouse
/opt/liangun/.venv/bin/python /home/ubuntu/longtou-yimai/longtou_runtime.py preflight
systemctl status longtou-yimai-day1.timer longtou-yimai-live.timer --no-pager
```

检查日志（目前不存在正式业务交易）：
```bash
ls -l /home/ubuntu/longtou-yimai/logs
ls -l /home/ubuntu/longtou-yimai/state/events
```

当云盘真实修复并且原始数据补齐之后，还必须：
1. 日志无新的文件系统错误；
2. 不只是块设备出现，要确保`/data`目录健康且重要数据可校验；
3. 补齐9月30日至当前最新交易日之间所有日线，不能跳日；
4. 验证单股分钟数据、日期连续、量额单位、跨源价格一致；
5. 对新系统运行真实行情只读演练，不发送飞书；
6. 备份独立 `settings.json` 后，单独确认消息开关从false变true；
7. 最后才考虑启用两个定时器。

**这些步骤本轮未完成，因此不存在“已满血自动运行”的承诺。**

## 不变更声明

没有写入`/data`、没有涉及旧版本策略的安装目录、没有修改持仓、没有启用自动交易、没有发布飞书测试消息、没有删除或停用旧服务，也没有触碰SSH密钥内容。

V22严格因果策略本身只是历史检验通过的研究版本，仍有历史参数搜索过拟合风险。即使基础设施恢复，也建议先设为`real_signals_armed=false`运行至少完整交易日验证，再进行真实提示。

## 2026-10-08 15:25 CST 后续部署增强审计

- 重新核验服务器：`/dev/vdb` 和 `/sys/block/vdb` 仍不存在，`/data` 读取返回 errno 5。两个新timer继续为 `disabled/inactive`，真实飞书信号 `real_signals_armed=false`。
- 将简单按日期间隔识别 RECYCLE 的近似方法，替换为基于过去每日涨停及断板状态的**原始V17状态机**。严格历史对照全部729条：PRIMARY 554、RECYCLE 175，**729条代际身份及事件类型全部匹配，0个错判**。这是因果生命周期修复，不是对收益率进行调参。
- 服务器实测10月8日收盘数据：腾讯行情有效3043只，覆盖率96.02%；独立日线构建2996条，覆盖率99.34%；2条Day1事件均通过新浪报价交叉验证。
- 出于数据盘事故，收盘数据只保存在`/home/ubuntu/longtou-yimai/quarantine/2026-10-08`，含`observed_daily.parquet`、`observed_candidates.json`和哈希清单。**此目录不被正式扫描程序读取，不构成Day1预登记，不能用来在10月9日发正式信号。**
- 腾讯分钟累计量额的单位和集合竞价差分已加异常值保护。由于老历史源分钟数据的成交量合计和日线全日量额存在差异，**不宣称已经证明与历史OHLCV完全一致**，所以保持停发。
- 飞书适配增设发送前路由校验以及逐条发送结果审计；在外部返回不确定时采用至多一次发送，避免重复推送。模拟测试确认停用状态不会调用外部CLI，未发送真实飞书消息。
- `preflight` 现在输出明确的`ready_for_live_signal=false`及`readiness_blockers`；不满足时返回退出码4。
- 新增`test_lifecycle_history.py`及扩展`test_staging.py`；全部测试通过，修改前代码均分别备份在独立`backups/`中。
- 此次仅改变独立的`/home/ubuntu/longtou-yimai`下文件，没有变动任何原有系统服务的工作目录、持仓文件或SSH材料；独立unit文件仍未启用。

**正式启用的必要条件**：云端检查并恢复数据盘设备及EXT4文件系统，确认所有相关数据完整；补齐9月30日后的每日行情；复测源时间、量额单位与VWAP；确认Day1快照在Day2前登记；完成至少一个交易日无通知灰度；确认飞书接收目标。上述全部合格后才可以考虑变更`real_signals_armed`和启用timer。当前均未执行。

## 2026-10-08 15:32 CST 数据源异常追加保护

- 只读查看现有R50日志，发现在15:06部分证券的腾讯分钟累计量额非单调，东方财富备用分钟源也出现连接中断。新适配层因此对累计量额单调性和分钟成交均价合理性进行检查；异常分时不产生买点。
- 对于`PRIMARY_3PLUS + NORMAL_RED + B4P`，如果Day1下午分钟VWAP上方比例缺失，W5与R4的先后触发无法完整比较，新程序现在统一`DAY1_PM_MISSING_CANNOT_RESOLVE_W5_PRIORITY`，**不发送R4或其他可能错误排序的信号**。
- 测试已验证该竞争规则、集合竞价差分、重复消息保护和全729条历史生命周期一致性。
- R50于15:13自行结束本次任务，systemd显示`Result=success`、`ExecMainStatus=0`、`NRestarts=0`，相关计时器仍在，未对其执行任何启动或停止。

## 2026-10-08 16:16 CST 新20GB云盘追加验收（最新）

- 新设备已出现在Ubuntu中：`/dev/vdc` 20GiB。`blkid -p` 未识别文件系统、`wipefs -n` 无签名、`sfdisk -d` 无分区表，当前未挂载。旧`/dev/vdb`仍缺失，`/data`仍为旧设备的故障残留挂载。
- 已独立备份`/etc/fstab`和磁盘只读探测记录（`backups/new-vdc-init-20261008T160305`），未改动旧挂载；远程执行环境拒绝执行格式化命令，**尚未格式化新磁盘，也未挂载它**。
- 为避免绕过破坏性操作限制，创建人工确认的一次性脚本：`/home/ubuntu/longtou-yimai/initialize_new_disk.sh`，权限700；**没有执行**。脚本仅允许经核实的空白20GiB`/dev/vdc`，再次检查签名、系统挂载、尺寸、旧盘状态，必须交互输入`FORMAT_NEW_EMPTY_VDC`，备份`fstab`后才创建ext4并永久挂载到`/srv/longtou-yimai-data`。不涉及旧`/data`、SSH密钥或其他服务。
- 新策略数据路径已独立设置为`/srv/longtou-yimai-data/state`；代码及冻结历史种子仍在`/home/ubuntu/longtou-yimai`。新代码会验证挂载点不是系统盘、ext4可读写、可用空间≥3GiB、文件fsync和读回哈希、状态子目录齐全，否则立刻关闭交易信号。
- 两个新systemd service已增加`RequiresMountsFor=/srv/longtou-yimai-data`和`ConditionPathIsMountPoint=/srv/longtou-yimai-data`，二者**仍未启用**。
- Day1下午VWAP数据已改为严格截取13:00～15:00共121条，避免供应商15:01～15:30的填充数据污染。对`000678`，现在只读计算可得到Day1 PM比例0.0，缺失时仍禁止交易。
- Day1正式登记前要求候选事件通过新浪/腾讯双源收盘价校验，实时发信号前还要求新浪报价时间戳足够新鲜。
- 飞书OpenClaw `message send --dry-run --json` 验证退出码0，无任何真实测试发送。
- R50原有交易日历只覆盖到2026-12-31，系统需要在2027年交易前更新外部日历，不能默认长期自动延续。
- 保持`real_signals_armed=false`，策略定时器均disabled/inactive，正式Day1与买点记录为0。下一步必须人工初始化唯一新空白磁盘，并由助手复核后完成2026-10-08真实日线/Day1登记及生产启用。

**一次性用户手工操作（执行后助手继续做部署验收）：**

```bash
ssh -t lighthouse /home/ubuntu/longtou-yimai/initialize_new_disk.sh
```

执行脚本将清空**仅限新`/dev/vdc`上的任何未识别数据**；当前只读验收确认该盘无文件系统与分区表，但用户必须明确输入确认。切勿对`/dev/vda`、旧`/dev/vdb`或已有`/data`运行格式化。脚本完成后向助手提供末尾`NEW_DISK_MOUNT_AND_FSYNC_ROUNDTRIP_OK`输出，助手继续验证并启用，不要自己手动修改其他旧策略。