# 龙头一买｜生产上线验收报告

**验收日期：2026-10-08，中国标准时间**  
**当前状态：生产提醒已开启；等待2026-10-09首个实际盘中交易时段验证。**  
**交易策略：V22_STRICT（龙头第一次分歧后的Day2第一买点；不混入R50二买）。**  
**备注：历史研究胜率84.62%来自反复筛选历史样本，非经过独立实盘验证的未来胜率。**

## 唯一服务器与SSH

- 快捷连接 `ssh lighthouse`
- 服务器：`ubuntu@120.53.23.29`，`VM-0-7-ubuntu`
- 2026-10-08 16:41：原始SSH验证成功，退出码0。
- 未读取、上传、修改任何SSH私钥，未连接旧服务器。

## 新数据盘已上线

- **唯一专用新数据盘**：`/dev/vdc`、20GB、EXT4、挂载于`/srv/longtou-yimai-data`。
- `/etc/fstab`使用UUID `08db1955-1389-4f31-b23d-5418cb1c648b`，配有`nofail`和设备超时，供下次启动自动挂载。
- `longtou_runtime.py`仅把`/srv/longtou-yimai-data/state`作为正式交易日数据与信号存储。
- 健康检查通过：设备挂载、独立于根盘、文件系统读写、fsync、数据回读、剩余空间等。
- **旧200GB盘 /dev/vdb 没有恢复，/data 的旧挂载记录仍有IO错误。** 不再是龙头一买依赖；为保护其他策略及原有系统，未对旧盘执行fsck、umount、清理fstab或其它可能破坏操作。

## 冻结策略和数据

- 程序：`/home/ubuntu/longtou-yimai/longtou_runtime.py`
- V22冻结核心：`/home/ubuntu/longtou-yimai/longtou_core.py`，与配置中SHA256相符。
- 历史种子：截至2026-09-30的789669条日线与729条事件记录。
- 2026-10-08收盘：3043只实时报价、2996条有效日线（覆盖99.34%）、2个提前登记Day1事件。
- 真实Day1于**2026-10-08 16:43:09 CST**登记并保存到新盘，早于10月9日09:25竞价截止时间。
- 观察事件：
  - `000011 深物业A`：3板后触板分歧，当前冻结V22主入口不覆盖此分支，仅作为观察记录；
  - `000678 襄阳轴承`：4板后的NORMAL_RED主策略Day1，下午VWAP比例0.0，翌日视实时规则判断是否触发。
- Day1快照与对应日线SHA256分别为：
  - `b4745fd695e80975b1982647af2edeb5ec5ea438bf04a5234098b1a6143210fe`
  - `9f2a4d03a7e499415cf930473269869091f63adebc19714e351d6a89ca48e895`
- 两份原始文件完整拷贝到了专用盘`archive/day1-source-2026-10-08`，比较字节一致；源文件权限600。
- 在真实交易结果出现前，**正式前瞻完成交易0笔，收益率和胜率不可计算。**

## 自动调度现已正式启用

```text
longtou-yimai-live.timer   enabled/active
    next: Fri 2026-10-09 09:25 CST
    actual watch: 09:30-09:47

longtou-yimai-day1.timer   enabled/active
    next: Fri 2026-10-09 15:12 CST
    postclose candidate registration
```

systemd服务独立使用ubuntu用户、受限CPU/内存，不依赖旧/data，`RequiresMountsFor`指向新盘。只进行**买点提醒，绝不自动下单、买卖持仓或修改holdings.json**。

## 飞书可用性

- 从原R50运行环境复用飞书发送目标到独立权限600的路由文件。
- 真实信号开关为`real_signals_armed=true`；配置更新前已在`backups/BEFORE_REAL_SIGNAL_ARMING_*`独立备份。
- `openclaw message send --channel feishu --dry-run --json`在用户环境和**systemd隔离环境**均通过（退出码0）。
- 模拟代码单测验证真实提醒路径、成交参考价格、来源时间、去重及发送结果独立存档。
- **截至本验收，没有发送真实飞书测试消息，也没有发生真实策略买点提醒。** 第一次实际交付能力需要在以后出现真实信号时才能证实。
- 不触发规则时不会发送买点；数据源延迟、价格锁涨停、分钟缺失、双源价差、竞价前Day1未登记或存储错误均拒绝发送。

## 核心测试/限制

- 729/729条历史First/Recycle生命周期与冻结结果一致，错判0。
- Day1当日两条真实样本通过双源收盘价交叉验证。
- 对Day2分钟序列实行因果前缀，禁止读取未完成分钟价格或事后卖出结果决定入场。
- 正式源采用腾讯累计分钟量额换算；与历史分钟OHLC完整逐笔一致性仍需要实际交易日继续核验，**不能声称已完成真实成交级一致性验证**。
- 襄阳轴承真实Day1的分支在纯函数构造的5分钟弱势序列上能触发W5信号；该测试不构成真实2026-10-09买点。
- 隔离测试没有发送真实飞书消息。
- 本方法仍有历史参数选择的过拟合风险，不能保证未来实际胜率、收益率或报价即为成交价。
- 原R50当前的交易日历覆盖到2026-12-31，其自有日历更新服务存在，若不更新则2027交易日前需续期。

## 变更及隔离声明

- 本轮安装/启用的只有`longtou-yimai-live.timer`与`longtou-yimai-day1.timer`。
- 代码、数据与新盘完全隔离；未重启、停止或删除任何旧业务服务。
- 原有R50在收盘后正常结束（`Result=success`），其他策略服务验收时保持active。
- 未修改持仓、SSH、公钥、旧的数据挂载条目，未对坏盘采取破坏动作。
- 生产通知配置`/home/ubuntu/longtou-yimai/config/settings.json`已设置为true，代码仓库中的默认模板继续保留false，防止克隆后意外发送信号。

## 日常只读巡检

```bash
ssh lighthouse
systemctl list-timers --all | grep longtou-yimai
systemctl status longtou-yimai-live.timer longtou-yimai-day1.timer --no-pager
set -a; . /home/ubuntu/longtou-yimai/private/feishu-route.env; set +a
/opt/liangun/.venv/bin/python /home/ubuntu/longtou-yimai/longtou_runtime.py preflight
tail -n 60 /home/ubuntu/longtou-yimai/logs/day2-live.log
```

10月9日开盘第一轮信号是否成功需要在实际交易中验证。若无触发或数据校验拒绝，则不会发送消息；遇到正式系统故障请以日志中的`FAIL_CLOSED`、`STALE`或`SOURCE_ERROR`作为排障证据。