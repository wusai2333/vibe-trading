# Vibe-Trading 会话交接文档（2026-08-18 更新）

## 零、新会话快速启动（先读这节）

**项目**：A 股量化因子研究 + 每日选股系统（沪深 300）。生产模型 **stable-7+EP**（2026-08-21 用户批准加固定 10% EP，勿改）：
gtja191_171 / alpha101_083 / alpha101_042 / qlib158_klow / alpha101_060 / limit_dist / vol_ivol60，
滚动 252 日 IR 权重、63 日重训、min_periods=6 平滑、Top-15/5 日调仓/10bps。
约束口径 OOS 31.9%/年、Sharpe 1.13（2019-2026，干净面板）。08-18 做过 13 变体审计：**无更优组合，别动**。

**每日三件套**（收盘 15:00 后按顺序，工作目录 `~/Vibe-Trading`）：
```bash
python reports/data/build_csi300_panel.py    # ~10 分钟，盯 DATA QUALITY 输出
# 若报 RECONCILE_RECOMMENDED：
python reports/data/panel_reconcile.py       # ~10 分钟（内置 baostock 登录重试）
python reports/data/panel_despike.py         # 跑 2-3 遍直到收敛
python reports/data/stable5_screener.py      # 个股 Top20 + 板块（内部已是 stable-7）
python reports/data/holdings_tracker.py      # 持仓打分 + 追加历史
```
然后手算上期 Top20 盈亏（vs 全池等权），写当日报告 `reports/YYYY-MM-DD_稳定7因子推荐.md`。

**用户持仓**（2026-08-18）：华能 25 / 中韩半导体ETF(513310) 10 / 紫金 10 / 亨通光电 10 /
生益科技 10 / 国电电力 10 / 雅克科技 9 / 中金黄金 8（92% 仓位）。持续问题：华能+国电 35%
压在弱信号上（仓位与信号倒挂），每次报告要提。

**铁律**：
1. 不主动推送 git——用户明说"推送"才用 `reports/scripts/push_snapshot.sh`
2. 报告只留本地；`reports/data/*.pkl` 勿提交
3. 生产模型改动需用户拍板；新因子必须先过 strict 随机对照 + 约束引擎增量检验
4. 盘中（9:30-15:00）可拉实时价做执行建议，但因子信号只用收盘数据
5. baostock 登录会瞬时失败（脚本已内置重试）；东财接口基本不可用；北向个股数据 2024-08 起停发

**当前待办**：每日三件套（收盘后；08-19 已完成——新信号对科技仅有限减仓 14→12 席，主线未变，
见 `reports/2026-08-19_稳定7因子推荐.md`）；定时任务自动化（用户未拍板）；推送（本地领先远端，等用户开口）。
**CSI500 试点三轮已全部收尾**：基本面线、session 增补、空白基线建模均关闭/打平，
结论：CSI500 对已测因子族无免费午餐，无新因子源前暂停（见 08-19 三份报告与第八节 #15-17）。
详细历史与所有坑见下文各节。

## 一、项目是什么

在 `~/Vibe-Trading`（HKUDS/Vibe-Trading 的本地 clone）上搭建的 **A 股量化因子研究 + 每日选股系统**。核心产出：样本外验证过的 stable-7 选股模型 + 每日三件套工具链 + 473 因子库的完整验证体系。

**GitHub**：`wusai2333/vibe-trading`（快照式推送，远端最新 `5572379` = 08-13 快照）。
**推送规则（2026-08-14 用户更新）**：默认**不再主动推送**。仅当用户明确说"推送/我想知道改了啥"时，才用 `reports/scripts/push_snapshot.sh` 推送。报告始终只留本地。

## 二、架构（三层）

```
数据层  agent/backtest/loaders/ashare_loader.py（自研，已并入源码）
        Sina+腾讯双源前复权日线 → reports/data/csi300_panel.pkl（288 只 × 8 年缓存）
引擎层  vibe-trading 原生因子注册表（462 因子）+ bench_runner / bench_runner_strict
应用层  reports/data/*.py 工具脚本 → reports/*.md 报告
```

**注意**：vibe-trading 内置 agent 聊天/TUI/Web UI 此前因 A 股 identity_conflict 门禁 bug 不可用——**已修复**（`3f5e730`）：根因是 search_symbol 未把 Yahoo 的 `.SS/.SZ` 后缀转成项目约定，同一 A 股以 `600519.SH`+`600519.SS` 双候选返回 → 门禁判 ambiguous → 全链路 identity_conflict。现可正常使用（A 股解析/取数已实测通过）。本项目自研脚本链仍是主力。

## 三、生产模型（stable-7+EP+hq20 暴露闸，勿随意改；EP 2026-08-21、hq20 2026-08-24 用户拍板）

- **因子（7 个 IR 加权 + 1 个固定权重）**：gtja191_171、alpha101_083、alpha101_042、alpha101_060、qlib158_klow、**limit_dist**（距涨停距离，confirmed_alive，IC ~100 倍随机）、**vol_ivol60**（特质波动率，reversed，负权重使用）；**fund_earnings_yield（EP）固定 10%**（2026-08-21 用户拍板方案 A）
- **EP 混合公式**：signal = 0.9 × IR加权(stable-7) + 0.1 × z(EP)。EP 是 IC 0.02 门槛的第一个反例（IC 0.0163 却过闸 3）；验证：固定 10% → 31.9%/Sharpe 1.17 vs 29.4%/1.04，剔 2020 改善仍在，**剔 2024-25 牛市后 Sharpe 持平——定性为牛市增强器**。EP 数据来自 fund_cache.pkl（公告日 PIT），screener 内 ffill 至面板末端（季频值在下次公告前不变，PIT 安全）
- **权重**：滚动 252 日窗口拟合 IR，每 63 日重训（无未来信息）
- **平滑**：`rolling(10, min_periods=6)`——**min_periods=6 是验证过的口径**（min_periods=10 会丢一半股票）
- **回测参数**：Top-15 等权，5 日调仓，单边成本 10bps
- **hq20 暴露闸**（2026-08-24 上线，S10）：沪深300 指数 Hurst（方差标度 k∈{2,4,8,16}、120d 窗、5d 平滑）落近 252 日最低 20 分位 → **建议半仓**，否则满仓。screener JSON 的 `hgate` 字段输出当日状态（H/阈值/暴露/人话）。指数缓存自维护：sina 全史覆写 + 滞后时腾讯实时补当日（前收吻合 <1e-4 才补）。验证：hq10~30 分位网格全 PASS（非参数运气），hq20 Sharpe 1.45→1.51、MaxDD -22→-15.1%、8月26 亏损 -4.86→-2.46% 砍半。**坑：H 估计下偏（中位 0.474），只用相对分位阈，勿用绝对阈 0.5**；caveat：单段历史，overlay 提 Sharpe 样本外常衰减
- **样本外成绩（约束口径，干净面板）**：**stable-7 +30.3%/年，Sharpe 1.08，回撤 -34.5%**（2019-2026）；stable-5 对照 +25.3%/Sharpe 0.99。2026YTD：stable-7 +19.0% vs stable-5 -2.1%。保留意见：ivol 权重符号有翻转、增益集中在 2021/2026。详见 `reports/2026-08-17_因子挖掘_涨跌停与波动率.md`
- **历史口径**：stable-5 干净面板 +25.9%/年（无约束）/+25.4%（约束）；旧 +50.8% 是受污染面板口径（见「已知坑」#11）
- **逐年差异极大**：2021 好、2022-2023 平淡——好年份坏年份差距大
- **模型切换点**：08-17 收盘起三件套用 stable-7；`holdings_history.csv` 08-11~08-14 是 stable-5 口径，**08-17 起的分位与之前不可直接比较**（报告需注明）。**08-21 起信号含 10% EP**（screener JSON 带 `ep_fixed_weight: 0.1`），08-21 前后的 score 亦不可直接比较。**08-24 起 screener JSON 带 `hgate` 暴露闸状态**，每日报告需播报闸门状态（半仓/满仓）
- **stable-7 组合审计**（2026-08-18，`csi300_stable7_audit.py`，13 变体约束口径）：7 因子消融——083/ivol60 贡献最大（去掉 -5.7/-6.5pp），171/limit_dist 中等（-2.3/-1.8pp），042 中性、klow 与 060 最弱（去掉 +0.7/+2.1pp 但逐年看是年份再分配：去 060 在 2019-2021 赢、2022/2026 输且回撤恶化到 -39.2%，不构成稳健改进）；5 个 top-IR 候选增补全部变差（与现有因子相关 0.5-0.76，冗余）。**结论：维持 stable-7 不变**

**关键验证结论**（别推翻）：
1. qlib158_klow 在 strict 模式下判 reversed，但**不能剔除**——滚动 blend 在平滑空间里它 IR 为正，剔除后成绩明显下降（方案 A 已验证；注意该验证也在脏面板上，方向仍成立、幅度待复核）
2. 4/5 因子过 bench_runner_strict 随机对照（随机 IC ≈0，真 IC 高 30-40 倍）
3. 等权基准被幸存者偏差污染（当前名单回测），**任何"跑赢基准"的对比都不可信**；且 2026-08-15 发现该基准在脏面板上 +34.5%、干净面板 +20.2%——**历史所有绝对收益数字都偏高**
4. **2026-08-15 数据清洗**：策略对干净面板的真实超额约 **+5.7pp/年**（25.9% vs 基准 20.2%），仍是可观的正 alpha，但远小于旧口径的表观超额

## 四、每日三件套（收盘后按顺序跑）

```bash
cd ~/Vibe-Trading
python reports/data/build_csi300_panel.py   # 1. 数据更新，~10 分钟
python reports/data/stable5_screener.py     # 2. 个股 Top20 + 板块排名，~2 秒
python reports/data/holdings_tracker.py     # 3. 持仓打分 + 追加历史，~5 秒
```

第 5 步（08-26 起）：`python reports/data/paper_tracker.py`——因子模拟盘跟踪
（paper_portfolio.json 定义，paper_history.csv 累积；建仓价自动锁定，见 #65）。

build 结束会打印 `DATA QUALITY: ... -> PASS/RECONCILE_RECOMMENDED`：PASS 正常；
RECONCILE_RECOMMENDED 说明双源拼接又出了新错价，跑 `python reports/data/panel_reconcile.py`（~10 分钟），
再跑 `python reports/data/panel_despike.py` 2-3 遍（剥洋葱式清残留尖峰，每遍 1-5 分钟），然后重跑三件套。

可选第 4 步（2026-08-24 起）：`python reports/data/news_briefing.py`——Top10 推荐+持仓的
东财个股新闻（近 3 天）→ LLM 逐股评估 → `news_briefing_latest.md`。**仅供人读参考，
绝不回流因子信号**。LLM 挂掉时降级为纯新闻列表。新闻当日缓存 `news_cache_*.json`，重跑不重复抓。

输出：
- `reports/data/stable5_screener_latest.json`（当日信号）
- `reports/data/holdings_history.csv`（持仓分位轨迹，逐日累积；08-11~14 为 stable-5 口径，08-17 起 stable-7 口径+新 3 股组合，两处断点不可直接比较）
- 上期推荐盈亏：手动用面板算（Top20 昨收→今收 vs 全池等权）

**近期回测记录**（Top20 vs 全池等权，超额；08-14→08-17 起为 stable-5 名单的收尾记账）：
08-07→08-11 -1.38%；08-11→08-12 +0.41%；08-12→08-13 **+1.89%**；08-13→08-14 +0.84%；08-14→08-17 **+2.78%**；08-17→08-18 -1.16%（半导体链获利回吐，4 连正后首负）；08-18→08-19 **-6.32%**（科技/半导体反转日重挫，跟踪以来最差单日超额；动量模型固有暴露，非因子失效）；08-20→08-21 +0.79%；08-21→08-24 **-3.29%**（光模块/电信链重挫，跟踪以来第二差；同属动量回撤日）；08-24→08-25 -0.10%；08-25→08-26 **-1.41%**（连续第 3 日负超额；池子 +0.90% 但 Top20 -0.51%，航运/能源拖累）；08-24→08-25 -0.10%（近平，黄金股回吐：山金 -5.0%/山东黄金 -4.2%；英维克 +10% 独亮）；08-26→08-27 **+2.84%**（3 连负后大翻身：池子 +0.94%、Top20 +3.78%，澜起 +10.1%/生益 +8.4%/中天 +8.0%/英维克 +6.3%；4 日累计 -1.96pp，动量信号在轮动市的高波动形态再次验证）

## 五、用户持仓（跟踪对象）

**2026-08-17 用户调仓后**：华能国际 25%、中金黄金 10%、国电电力 8%，其余 **57% 空仓**（紫金矿业/中钨高新/藏格矿业/中信金属已清仓）。`holdings_tracker.py` 的 HOLDINGS 已同步。
~~08-18 调仓~~ → **08-25 调仓（以此为准）**：紫金 20.8 / 兆易 20 / 平安 12.5 / 华能 11.7 /
中金黄金(600489) 8.3 / 国电 8.3 / 宝丰 7.5（约 89% 仓位；用户原始权重/1.2）。
清仓 中韩ETF/亨通/生益/雅克。tracker HOLDINGS 已同步，08-24 面板口径加权分位 62.8。
**历史坑**：旧 HOLDINGS 曾把 000975 标作"中金黄金"——000975 实为山金国际（现名），
600489 才是中金黄金；08-25 用户确认持仓是 600489。
**注意**：国电已不在 CSI300 成分内，面板不含，tracker 单独拉取评分。
stable-7 口径（08-14）分位：中金黄金 83.8%（强）、华能国际 51.0%（中位）、国电电力 25.2%（弱）——倒挂问题部分解决（最强的中金已加仓），但国电仍弱于其仓位应得。
历史持仓（08-17 前）：华能 25%、紫金 20%、中钨 15%、国电 10%、藏格 8%、中金 8%、中信金属 7%；旧结论"仓位与信号倒挂"在 08-11~08-14 报告中留档。

## 六、已知坑（踩过的，别重踩）

1. **腾讯接口传显式 end_date 会丢最新一根 bar**——ashare_loader 已修（end 是今天时留空），提交 `3b0096f`
2. **面板重建后必须检查最后一日 NaN 数**——build 脚本已内置自动修补，但数据源没更新时（盘中/刚收盘）会补不上，需截断残缺行
3. **csindex 板块接口限流**——连续请求 ~8 次后挂；板块映射有 7 天缓存（`stock2sector_cache.json`）
4. **东财接口在这个网络环境基本不可用**（频繁断连）；Tushare 免费 token 无 index_weight/adj_factor 权限
5. **orphan 快照推送流程会误删未跟踪文件**——`git add -A` + `checkout -f` 组合曾删掉 6 份报告（08-11/12/13）。已修复：新增 `reports/scripts/push_snapshot.sh`，推送时 `git rm --cached reports .github/workflows` 排除本地文件，且要求工作区无未提交 tracked 改动。**以后推送必须用这个脚本，别再用裸 `git add -A`**
6. **误删的报告可从 git 悬空对象恢复**——`git fsck --dangling` 找 dangling commit，`git show <hash>:reports/xxx.md` 提取。08-14 已用此法恢复 6 份
7. **推送极慢**（网络问题）：正常 1-15 分钟，最差 92 分钟；`send-pack: unexpected disconnect` 是常态，重试即可
8. **中钨高新、中金黄金等会停牌**——因子可算但价格 NaN，跟踪脚本已容错
9. **推送脚本的 pathspec 排除法是坏的**——`git checkout --orphan` 保留 main 的索引，`git add -A -- . ':(exclude)reports'` 不会移除索引里已跟踪的 reports 文件，08-14 两次快照尝试各泄漏 34 个文件。已修（`a48e72e`）：先 `git add -A` 再 `git rm -rq --cached reports .github/workflows`，并加 ERR trap——推送失败自动回 main、删 orphan 分支，不再把仓库滞留在 push-snapshot 上
10. **国电电力、中信金属不在沪深 300 面板里**——holdings_tracker 会即时用 a_share loader 拉取；离线重建历史时也要走 loader，别只查面板
11. **双源前复权拼接有大面积错价（2026-08-15 发现并修复）**——腾讯/新浪各自返回的 qfq 序列都有整段坏区（偏离真实 10-50 倍）与拼接跳变（单日 +54%～+5200%），曾使回测虚增约一倍（50.8%→25.9%）。**新浪不是干净参照**（它有自己的坏段）。已建清洗管道：`panel_scrub.py`（水平错位掩码，已接入 build 脚本）+ `panel_reconcile.py`（用 **Baostock** 做独立仲裁的深度重建，约 10 分钟，全量重建面板后应跑一次）。**2025-2026 数据未被清洗改动，实盘信号不受影响**；脏面板备份 `/tmp/csi300_panel.pkl.bak-20260814`
12. **回测必须带收益护栏**——主板 >12%、创/科 >22% 的非复牌单日收益物理不可能，P&L 归零（`csi300_constrained_backtest.py` 已内置；盐湖股份 2021-08-10 +306% 是真实复牌事件，长停牌豁免保留）
13. **baostock 已装**（`.venv/bin/pip install baostock`，免费无 token）——qfq 复权链与面板近端比值 0.998-1.003，可作第三数据源/仲裁；东财依旧不可用
14. **跨源符号后缀约定差异会触发 identity_conflict**——Yahoo 用 `.SS/.SZ`、项目用 `.SH/.SZ`，同一标的双候选会让 grounding 门禁判 ambiguous 并锁死全链路（已修，`3f5e730`）。以后接新数据源先核对符号约定，并跑 search_symbol → GroundingLedger 联动验证
15. **方向涉及"追涨停"的因子，无约束回测必虚增**——limit_dist 单独加入时无约束 +3.7pp、约束后只剩 +1.1pp（涨停挡买从 10 次飙到 56 次：无约束引擎能"买到"实盘买不到的封板股）。**新因子增量检验一律先过约束引擎**（`csi300_limit_vol_test.py` 双引擎对照模板）
16. **个股层面北向持股数据已死**——东财 2024-08-16 后停发日频个股北向持股（监管披露调整），akshare 的 individual/hold_stock/statistics 系接口要么数据止于该日、要么直接报错；只剩聚合流向（stock_hsgt_hist_em）。涉及北向的第三方策略/研报数据都止步于此，别再投入
17. **全量重取必带回拼接错价，reconcile 后仍有历史残差**——08-17 实测：build 后质量门报 286 个非法收益，reconcile 修掉大部分但剩 610 个，形态是**整段假斜坡**（错复权时代，如 002532 连续 5 天 +13%——reconcile 因"面板≈新浪"而整段采纳）。**已修**：`panel_despike.py` 后处理（不可能收益→按 run 用 Baostock 锚点重缩放替换，迭代剥洋葱 610→38→21→14）。剩余 14 个是地板：601088 2018 四个（两源共有错误，无第三源可裁）、688256 上市首周（合法无涨跌幅）、002466 短停牌复牌、几个 2018 错时代边缘 + 2 个 2025 Baostock 也确认的残差。**实盘窗口（2025+）干净，今日信号验证零漂移**。reconcile 原依赖 /tmp 原始备份（已被系统清理，脚本已改为回退用当前 scrub 后面板）
18. **baostock 会瞬时登录失败**——08-17/18 出现四次。已根治：panel_reconcile.py / panel_despike.py 内置 5 次重试（间隔 5 秒）
19. **收益护栏 shift 错位 bug（2026-08-20 发现并全库修复）**——护栏掩码抄成了 anomalous.shift(1)，
    屏蔽的是非法收益的"后一天"，非法收益本身直通 P&L（正确：shift(-1)，因 fwd[t]=ret[t+1]）。
    CSI300 干净面板上影响仅 -0.4pp（30.1→29.7，噪声级，历史结论在误差内成立）；CSI1000 脏面板上
    造出 2023 +243% 假年份（000422.SZ +1207% 脏格被持仓命中）。15 个回测脚本已统一 sed 修复。
    教训：好得离谱的年份先怀疑数据管道。审计脚本 csi300_guardfix_check.py


## 七、关键文件索引

| 文件 | 作用 |
|---|---|
| `reports/data/csi300_panel.pkl` | 数据面板缓存（26MB，勿提交 git，已 gitignore） |
| `reports/data/stable5_screener.py` | 主力筛选器（个股+板块）；**08-17 起内部为 stable-7**，文件名保留以不动三件套命令 |
| `reports/data/holdings_tracker.py` | 持仓跟踪（**08-17 起 stable-7 口径**） |
| `reports/data/build_csi300_panel.py` | 面板构建（含最后一日自动修补 + panel_scrub 水平错位掩码） |
| `reports/data/csi300_planA.py` | 方案 A 验证（4 vs 5 因子对照） |
| `reports/data/csi300_strict_compare.py` | strict 严格验证 + 两两对比 |
| `reports/data/csi300_constrained_backtest.py` | 滚动权重 OOS 重建 + 涨跌停/停牌约束回测（原 csi300_rolling_weights.py 已丢失，此为权威替代，含收益护栏） |
| `reports/data/csi300_session_blend_test.py` | session 因子增量 OOS 对比（stable-5 vs +onin20） |
| `agent/src/factors/zoo/session/` | session 因子家族（隔夜/日内分解 4 因子；判定见 08-17 报告，未入生产） |
| `agent/src/factors/zoo/limit/` | limit 因子家族（涨跌停微观结构 3 因子；limit_dist 为 stable-7 候选） |
| `agent/src/factors/zoo/vol/` | vol 因子家族（rvol20/ivol60；ivol60 为 stable-7 候选） |
| `reports/data/csi300_limit_vol_test.py` | limit/vol 候选双引擎增量检验（无约束 vs 约束对照模板） |
| `reports/data/build_fund_panel.py` | baostock 基本面 PIT 面板构建（可续跑，输出 fund_cache.pkl） |
| `reports/data/csi300_fund_bench.py` | 基本面因子 strict 检验 |
| `reports/2026-08-17_因子挖掘_基本面因子.md` | 基本面 6 因子全 noise 的检验记录 |
| `reports/2026-08-17_因子挖掘_隔夜因子试点.md` | 干净面板复核 + session 因子试点全记录 |
| `reports/2026-08-17_因子挖掘_涨跌停与波动率.md` | limit/vol 候选检验 + stable-7 升级提案 |
| `reports/data/panel_scrub.py` | 面板水平错位检测（build 时自动跑） |
| `reports/data/panel_reconcile.py` | 面板深度重建（Sina+Baostock 仲裁，全量重建后跑） |
| `reports/data/panel_despike.py` | reconcile 后残留尖峰修复（Baostock 锚点替换，幂等，跑 2-3 遍） |
| `reports/2026-08-11_严格验证与因子对比.md` | 模型定型的依据报告 |
| `reports/2026-08-11_方案A验证.md` | 为什么保留 5 因子 |
| `reports/2026-08-15_约束回测与数据清洗.md` | 约束回测结果 + 数据污染发现与修复全记录 |
| `reports/scripts/push_snapshot.sh` | 安全推送脚本（排除 reports，防误删，失败自动回 main） |
| `reports/data/rebuild_holdings_history.py` | 08-11/12/13 持仓历史重建脚本（一次性，留档） |
| `reports/data/csi500_cons.json` | 中证 500 成分（baostock query_zz500_stocks，2026-08-17） |
| `reports/data/build_csi500_panel.py` | CSI500 价格面板构建（scrub+质量门，同 CSI300 链） |
| `reports/data/panel_reconcile_csi500.py` / `panel_despike_csi500.py` | CSI500 清洗链（reconcile 无备份时回退 scrub 后面板） |
| `reports/data/csi500_panel.pkl` | CSI500 清洗后价格面板（勿提交 git） |
| `reports/data/build_fund_panel_csi500.py` | CSI500 baostock PIT 基本面面板 |
| `reports/data/fund_cache_csi500.pkl` | CSI500 基本面缓存（勿提交 git） |
| `reports/data/csi500_fund_bench.py` / `csi500_fund_strict.json` | CSI500 基本面 strict 检验脚本与结果 |
| `reports/data/csi500_ep_test.py` / `csi500_ep_test.json` | EP 双引擎增量检验（负增量，关线依据） |
| `reports/data/csi500_zoo_bench.py` / `csi500_zoo_strict.json` | limit/vol/session 三族 strict 检验 |
| `reports/data/csi500_factor_cache.pkl` | 10 因子缓存（勿提交 git） |
| `reports/data/csi500_session_corr.py` | 向量化截面 Spearman 相关性闸 |
| `reports/data/csi500_session_test.py` / `csi500_session_test.json` | session 因子双引擎增量检验 |
| `reports/data/csi500_blank_baseline.py` / `csi500_blank_baseline.json` | 空白基线三候选池回测（打平无超额） |
| `reports/data/stable7_crash_diagnose.py` | 08-19 崩盘日诊断（近期 IC/尾部日分布/集中度，只读） |
| `reports/data/csi300_sector_cap_test.py` / `csi300_sector_cap_test.json` | 行业上限增量检验（cap8/6/5 双引擎，结论：保险非 alpha） |
| `reports/2026-08-19_因子挖掘_行业上限检验.md` | 行业上限检验全记录 |
| `reports/data/csi300_tree_blend_test.py` / `csi300_tree_blend_test.json` | 树模型组合检验（HistGBR d2/d3 + 随机对照，结论：负面） |
| `reports/data/csi300_xgb_blend_test.py` / `csi300_xgb_blend_test.json` | XGBoost 对照（同口径，同样负面；xgboost 3.4.1 已装） |
| `reports/2026-08-19_因子挖掘_树模型组合.md` | 树模型检验全记录 |
| `agent/src/factors/zoo/pool2/` | 扩池第一轮 10 因子（过门禁，0 存活，留档复现用） |
| `reports/data/csi300_pool2_strict.py` / `csi300_pool2_strict.json` | pool2 strict 检验 |
| `reports/data/csi300_rev5_test.py` / `csi300_vpcorr20_test.py`（+json） | rev5/vpcorr20 三道闸记录 |
| `reports/2026-08-19_因子挖掘_因子池扩展第一轮.md` | 扩池第一轮全记录 |
| `reports/data/build_margin_panel.py` / `margin_panel.pkl` / `margin_cache_parts/` | 两融 PIT 面板构建与缓存（pkl 勿提交） |
| `agent/src/factors/zoo/margin/` | 两融 4 因子（fchg5/fchg20/fbuy5/schg5，全 reversed_strict） |
| `reports/data/csi300_margin_bench.py` / `csi300_fchg20_test.py`（+json） | 两融 strict 与增量检验 |
| `reports/2026-08-19_因子挖掘_两融因子.md` | 两融因子全记录 |
| `reports/data/csi300_sector_weights_test.py` / `csi300_sector_weights_test.json` | 板块级权重检验（sector_w/shrink，结论：年份再分配） |
| `reports/2026-08-19_因子挖掘_板块权重检验.md` | 板块权重检验全记录 |
| `reports/data/build_csi1000_panel.py` / `csi1000_panel.pkl` / `csi1000_cons.json` | CSI1000 面板构建（分块可续跑；pkl 勿提交） |
| `reports/data/panel_reconcile_csi1000.py` / `panel_despike_csi1000.py` | CSI1000 清洗链 |
| `reports/data/csi1000_port_test.py` / `csi1000_port_test.json` | CSI1000 搬入回测（修正护栏后：负超额） |
| `reports/data/csi300_guardfix_check.json` | 护栏 bug 影响审计（legacy vs 修正） |
| `reports/2026-08-20_因子挖掘_CSI1000试点与护栏bug.md` | CSI1000 试点 + 护栏 bug 全记录 |
| `agent/src/factors/zoo/pool3/` | modsearch 扩池 8 因子（vmax20 最优但增量负） |
| `reports/data/csi300_pool3_strict.py` / `csi300_pool3_strict.json` | pool3 strict 检验 |
| `reports/data/csi300_vmax20_test.py` / `csi300_vmax20_test.json` | vmax20 三道闸记录 |
| `reports/2026-08-20_因子挖掘_modsearch扩池.md` | modsearch 扩池全记录 |
| `agent/src/factors/zoo/lit/` | 顶刊族 6 因子（coskew/dnbeta/tail/resmom/psliq/trend，0 alive 留档） |
| `reports/data/csi300_lit_strict.py` / `csi300_lit_strict.json` | lit strict 检验 |
| `reports/data/csi300_monthly_test.py` / `csi300_monthly_validate.py`（+json） | 月度赛道回测与 B 验证 |
| `reports/2026-08-20_因子挖掘_文献族终局.md` / `_月度赛道试点.md` | 文献族终局 + 月度赛道全记录 |
| `reports/data/earnings_fetch.py` / `earnings_cache/` / `event_panel.pkl` | 业绩预告/快报拉数与 PIT 事件面板（pkl 勿提交） |
| `reports/data/csi300_regime_overlay_test.py`（+json）/ `csi300_index_daily.csv` | regime 覆盖层检验（全灭留档）；指数日线=新浪源 2002-2026 |
| `FinRL-Trading/`（根目录，已 gitignore） | AI4Finance FinRL-X 参考仓库，仅 regime 阈值有借鉴价值 |
| `reports/data/csi300_finrl_ml_test.py` / `_validate.py`（+json×2, finrl_ml_score.pkl） | FinRL 式 ML 筛选与季频袖验证（关线留档） |
| `reports/data/csi1000_monthly_features.py` / `csi1000_lgbm_monthly.py` / `_recompute.py`（+pkl×2, json×2） | 月频赛道 v2：431 因子月末特征库 + LGBM 走查 + 修正 metrics（LS 袖留档） |
| `astock-alpha-factor-lab/`（根目录，已 gitignore） | A 股因子实验室参考仓库，ml_lgbm/portfolio_opt 两个规格有借鉴价值 |
| `reports/data/csi300_cst_practice.py` / `csi300_ep_validate.py`（+json×2） | CST 五论断检验 + EP 验证（待拍板） |
| `Cheap-Stable-Trending-quant/`（根目录，已 gitignore） | CST 深度价值项目参考仓库，研究流程文档有方法论价值 |
| `reports/2026-08-21_因子挖掘_CST实践检验与EP发现.md` | 闸的自我检验 + EP 牛市增强器定性 |
| `agent/src/factors/zoo/event/` | 事件 5 因子（surp/type/jor/mom/kb，全灭留档） |
| `reports/data/csi300_event_strict.py` / `csi300_event_strict.json` | 事件 strict 检验 |
| `reports/2026-08-20_因子挖掘_业绩预告事件.md` | 业绩预告事件全记录 |
| `reports/data/csi300_lr_test.py` / `csi300_lr_test.json` | Ridge/LR 联合训练检验 |
| `reports/2026-08-20_因子挖掘_训练式线性模型.md` | 训练式线性模型全记录 |
| `reports/2026-08-19_因子挖掘_CSI500微结构与隔夜因子.md` | CSI500 三族因子检验 + 空白基线全记录 |
| `reports/2026-08-19_因子挖掘_CSI500基本面试点.md` | CSI500 试点全记录（EP confirmed_alive） |
| `reports/HANDOFF.md` | 本文档 |

## 八、未完成 / 可选下一步

1. ~~08-14 三件套~~ **已完成**：面板更新 + 筛选 + 持仓 + 上期盈亏，报告 `reports/2026-08-14_稳定5因子推荐.md`
2. **定时任务**：每日收盘自动跑三件套 + 推送结果（用户多次被推荐，未拍板）
3. ~~回测加涨跌停/停牌约束~~ **已完成**（2026-08-15）：约束成本仅 ~0.5pp/年，报告 `reports/2026-08-15_约束回测与数据清洗.md`。过程中发现并修复面板数据污染，基线修正为 +25.9%/年
4. ~~内置 agent identity_conflict bug~~ **已修复**（2026-08-17，`3f5e730`）：Yahoo `.SS/.SZ` 后缀未转项目约定致双候选 ambiguous。工具层转换 + 门禁层回归测试 4 个，全量测试 2171 过（test_agent_goal_context 的 sqlite 失败是沙箱环境问题，与本改动无关）。Web UI/TUI 现可用于 A 股
5. ~~fundamental 因子库~~ **已完成**（2026-08-17）：Tushare 权限墙用 **baostock** 绕开——新建 `build_fund_panel.py`（288 只 × 8.5 年日频估值+季频盈利，pubDate 严格 PIT 对齐，31 分钟可续跑，输出 `fund_cache.pkl`）。fundamental zoo 扩到 6 因子（新增 fund_bp、fund_epsmom），strict 检验 **6/6 noise**（最强的 EP 也只有 IR 0.069）——沪深 300 日频基本面信号不可用，与文献一致（价值/质量溢价在小微盘+长周期）。管道留作基础设施，池子扩到中证 500/1000 可复用。报告 `reports/2026-08-17_因子挖掘_基本面因子.md`
6. ~~推送待办~~ **已完成**（2026-08-27 用户"提交推送远程"）：本地先提交 `56b20ad2`（hq20 闸门+模拟盘工具+zoo 新因子池+gitignore 补 *.pkl/*.log/缓存规则），再跑 push_snapshot.sh，远端 `5572379`→`aea8a335`。**同日设计变更（用户拍板）**：研究脚本+HANDOFF 也推远端——push_snapshot.sh 改为排除 reports 后回补 `reports/data/*.py`+`reports/scripts/`+`HANDOFF.md`，第二次推送 `aea8a335`→`40027af9`（验证：91 个研究脚本+HANDOFF 在远端，0 个报告 md/json/csv/pkl 泄漏，本地盘完好）。安全前提：reports/ 文件已全部提交进 main，checkout 回程只恢复不删除；**以后推前必须先提交 reports 新文件**。仍不推：每日报告/策略库等 md、结果 json/csv、数据缓存。历史记录：远端曾长期停在 08-13 快照 `5572379`。08-14 本地做过三次快照尝试（`865eb28`/`830fcfd`/`b7ab7e4`，均未推送，前两次还泄漏了 reports；`9775b01` 是悬空提交，其修复已并入 `a48e72e`）。脚本已修好并端到端验证（本地 bare remote 测试：快照 0 reports/0 workflows、推送失败自动回 main）。**用户说"推送"时用 `reports/scripts/push_snapshot.sh` 推即可**
7. ~~08-14 晚间 git/历史修复~~ **已完成**（本次会话）：a) 修 push_snapshot.sh 的 pathspec 泄漏 + 失败滞留（`a48e72e`，已提交未推送）；b) 重建 `holdings_history.csv` 的 08-11/12/13 历史行（分位转录自旧报告，价格/动量由面板与 loader 重算，国电电力 08-13 重算值 +1.0%/+8.6% 与 08-13 报告完全吻合）；c) 更正本文档 git 状态描述
8. ~~干净面板上的复核~~ **已完成**（2026-08-15）：方案 A（5 因子 27.0% > 4 因子 24.4%，qlib158_klow 仍不可剔除）、strict 随机对照（4/5 confirmed_alive，真 IC 是随机 30-70 倍，qlib158_klow 仍 reversed）、min_periods=6（10 会丢 ~65 只/日）——**三条定性结论在干净面板上全部复现**，生产模型无需改动，仅绝对收益下修。详见 `reports/2026-08-15_约束回测与数据清洗.md` 第四节
9. **ashare_loader 源头修复**（可选）：错价根因在双源拼接，清洗是下游兜底；长期可在 loader 里加源间一致性校验（当前用下游 scrub+reconcile 已够）
10. ~~干净面板复跑 zoo bench + 隔夜因子试点~~ **已完成**（2026-08-17）：a) 462 因子干净面板复核，stable-5 排名结构不变（第三次验证）；b) 新建 session zoo（隔夜/日内收益分解 4 因子），strict 判 onin20 confirmed_alive、in20 reversed_strict，但 onin20 与 qlib158_klow 相关 -0.77、与 in20 相关 -0.97，OOS 增量仅 +0.2pp 且回撤恶化、2026YTD 大幅跑输——**不并入生产**（清晰的阴性结果，两道闸门工作正常）。报告 `reports/2026-08-17_因子挖掘_隔夜因子试点.md`
11. ~~因子挖掘：涨跌停微观结构 + IVOL/低波~~ **已完成**（2026-08-17）：新建 limit/vol 两个 zoo（5 因子）。limit_dist confirmed_alive（IR +0.155，IC ~100 倍随机，与 stable-5 相关 ≤0.20）、vol_ivol60 reversed_strict（IR -0.092）；rvol20/upcnt20/dncnt20 因冗余或太弱弃。双引擎增量检验：stable-7（+limit_dist+ivol60）约束口径 30.3%/年（+5.0pp）、Sharpe 1.08、2026YTD +19.0%——**用户已批准，08-17 起上生产**（见「生产模型」节）。报告 `reports/2026-08-17_因子挖掘_涨跌停与波动率.md`
12. ~~因子挖掘剩余候选~~ **全部收尾**（2026-08-17）：a) 基本面——baostock 解锁，6 因子全 noise（见 #5）；b) **北向资金流——方向关闭**：东财个股北向持股数据（akshare stock_hsgt_individual_em）**止于 2024-08-16**（2024-08 起交易所停发个股层面日频披露，持股排行类接口全部失效），仅剩聚合流向（hsgt_hist_em 更新正常）但无横截面信息，做不了选股因子。若未来恢复季度披露可再评估。**至此因子挖掘四条线 + 基本面 + 北向全部完成，生产模型 stable-7 定型**
13. ~~08-17 三件套（stable-7 首次实盘）~~ **已完成**：质量门首次实战报警→reconcile 修复→筛选+持仓+上期盈亏（+2.78%，连续 4 日正超额），报告 `reports/2026-08-17_稳定7因子推荐.md`。用户当日调仓（见第五节）
14. ~~reconcile 精细化~~ **已完成**（2026-08-17）：`panel_despike.py` 后处理——残留 610 个非法收益（整段假斜坡形态）迭代修复到 14 个地板（详见坑 #17），今日信号验证零漂移。日常流程：质量门报警 → reconcile → despike 2-3 遍 → 三件套
15. ~~CSI500 基本面试点~~ **已完成并关线**（2026-08-19）：全套管道搬到中证 500（成分 baostock，面板 500×2093 天，清洗后残差 65、2025+ 仅 1、2026 零）。strict：**fund_earnings_yield confirmed_alive（IR 0.077，IC 1134×随机）——本项目第一个过 strict 的基本面因子**；bp 信号全在 2022 后判 noise；epsmom train_only。增量检验（用户批准后）：相关性闸通过（max |ρ| 0.253），但双引擎约束回测 **stable-7+EP 约束 37.1% vs stable-7 42.2%（-5.1pp），Sharpe 1.32→1.19，EP 滚动权重 mean -0.008——负增量，基本面线在中盘同样关闭**。报告 `reports/2026-08-19_因子挖掘_CSI500基本面试点.md`
16. ~~CSI500 微结构/波动率/隔夜因子~~ **已完成**（2026-08-19）：limit/vol/session 9 因子 strict——**session_onin20 全场最强（IR 0.200，t 6.7/6.1，IC 308×随机）**；limit_dist/ivol60 跨池稳健；on20/on5 alive，四个 reversed。相关性闸全过（onin20 vs klow 仅 -0.23，CSI300 上的 -0.77 致死冗余在中盘消失）。增量检验：**单加 onin20 负增量（-3.1pp）；成对 +onin20+on20 +1.9pp 但是年份再分配（2021 -23.2pp）——按判例不构成稳健改进，作为 stable-7 增补关闭**。报告 `reports/2026-08-19_因子挖掘_CSI500微结构与隔夜因子.md`
17. ~~CSI500 空白基线建模~~ **已完成：打平无超额，CSI500 线暂停**（2026-08-19）：预注册三候选池同一机制回测——pool_top3（onin20+limit_dist+ivol60）约束 27.5%/Sharpe 0.78（涨停挡买 263 次，最小推荐集不成立）；pool_alive（5）38.1%/1.16；**pool_all（10 因子全池）42.7%/Sharpe 1.26/MaxDD -36.3%，与搬来的 stable-7（42.2%/1.32/-38.5%）打平但无超越**，逐年是不同分布非改进。实现权重 mean |w| ≤0.15（池稀释+窗口 IR 不稳）。**结论：若无新因子源/新机制（非线性、换频、扩池到 1000），CSI500 线暂停；基础设施全部保留可复用**。报告同 #16 文件（空白基线节）

18. **08-19 崩盘日诊断结论（stable7_crash_diagnose.py）**：08-18→08-19 Top20 超额 -6.32%（跟踪以来最差）。
    因子未失效（崩盘前两天 blend IC +0.44/+0.57，尾部日后 IC 1-3 天恢复），但重建 7.5 年 OOS 发现
    **全部 7 个单日超额 ≤-4% 日都在 2026**（05-29/07-09/07-14/07-16/07-27/07-29/08-18），机制全是
    科技集中仓撞反转；14/20 IT+电信处历史 97.2% 分位。**风控缺口 = 模型无行业上限**。
    **检验已完成（csi300_sector_cap_test.py，用户批准）**：cap6 把 ≤-4% 尾部日从 10 砍到 5（2026 9→4），
    但 Sharpe 持平（1.06）、全 OOS -0.8pp/年、代价集中在 2026（YTD 13.7%→8.4%）——按判例不构成稳健改进，
    **建议不并入生产，stable-7 维持不变；用户 08-19 拍板选 A（维持 baseline，不上上限）**。极端日板块内部 IC 也翻转（-0.71），
    上限降频率不消事件（cap6 下 08-19 仍 -5.7%）。详见 reports/2026-08-19_因子挖掘_行业上限检验.md

19. ~~树模型组合试点~~ **已完成：负面结果，关线**（2026-08-19，用户批准）：同 7 因子、同滚动口径，
    HistGBR d2/d3 约束口径 22.4%/20.4% vs 线性 30.1%，Sharpe 0.68/0.62 vs 1.06，MaxDD 恶化 11pp；
    随机特征对照 IC≈0（管道无泄漏）；逐年看树在 regime 翻转年（2022 -31.9%、2026 -10.8%）高确信度站错队。
    追加 XGBoost(hist) 对照同样全败（d2 25.9%/Sharpe 0.79/MaxDD -49.0% vs 线性 30.1%/1.06/-34.8%，
    随机对照 IC≈0）。再追加 422 全因子树：CAGR 32.2%/IC 0.0216 小赢，但 Sharpe 1.03↓/MaxDD -41.4%↓，
    2022/2026 翻转年 -17.5%/-6.6%（线性 +11.6%/+13.7%）——彩票式方差，判例拒绝；先筛后喂的 55 因子版
    反而 15.2%（双重窥视教学案例）。结论：日频截面 R² ~0.1% 的信噪比下低方差线性估计就是最优族，**stable-7 维持不变**；
    "为树重新筛因子"一并关闭（机制两连败，前提不成立）。未扫参（判例：重复窥视 OOS）。
    报告 reports/2026-08-19_因子挖掘_树模型组合.md
20. ~~因子池扩展第一轮~~ **已完成：0 存活，关线**（2026-08-19，用户指示扩池）：预注册 10 个文献因子
    （rev5/max20/momacc/clv20/vpcorr20/dnvol60/rng20/updays20/volratio/obv20），新 zoo `pool2/` 过门禁。
    strict：仅 rev5 confirmed_alive（IR 0.114）；相关性闸：rev5 与 limit_dist 相关 0.834 阵亡（换皮），
    max20/rng20/dnvol60 撞 ivol60（0.70-0.87）；最强过关者 vpcorr20（max|ρ| 0.127）增量检验 **-2.5pp**
    （权重符号翻转，IC regime 漂移）。结论：stable-7 已占满 OHLCV 日频主要信号方向，局部最优假说再强化。
    web_search 不可用（缺 DEEPSEEK_API_KEY），候选来自文献预注册。报告 reports/2026-08-19_因子挖掘_因子池扩展第一轮.md
21. ~~两融因子试点~~ **已完成：关线，但质量是扩池以来最高**（2026-08-19，用户批准）：交易所官网个股两融
    面板 2018 起（build_margin_panel.py，分块+原子分片缓存，PIT shift(1)）。预注册 4 因子 strict **4/4 reversed_strict**
    （融资追涨是反向指标：fchg20 IR -0.101/t -3.98 最强）；相关性闸 3/4 过（fchg20 max|ρ| 仅 0.115，前所未见的正交）；
    增量检验 fchg20 **-5.2pp** 阵亡（IC 绝对值太小，blend 权重噪声成本>信号）。**stable-7 维持不变**。
    教训：连续第三个"过两闸死三闸"——blend 增量门槛极高，日频 IC<~0.02 进不去。基建小改：registry 白名单加 margin: 命名空间（2 行）。
    报告 reports/2026-08-19_因子挖掘_两融因子.md
22. ~~板块级因子权重~~ **已完成：关线**（2026-08-19，用户假设"每板块一个最优组合"）：sector_w（10 板块各自
    滚动 IR 权重）Sharpe 1.14 首超 baseline 1.06，但逐年拆开是年份再分配（2024/2026 让 23.8pp 换 2023/2025 的
    22.2pp），CAGR 29.3↓、2026YTD 2.0 vs 13.7、尾部日 11 vs 10；sector_shrink（50/50 收缩）全面更差——
    板块特异性分量大部分是噪声（能源 8 只/公用 11 只的截面拟合 IR 无意义）。按判例不构成稳健改进，
    **stable-7 维持不变**（今日第七次局部最优验证）。报告 reports/2026-08-19_因子挖掘_板块权重检验.md

23. ~~CSI1000 扩池试点~~ **已完成：负超额，关线**（2026-08-20，用户批准）：1000 只面板新建
    （分块拉数+17 遍 despike，2025+ 残留 3 个 baostock 确认地板）。stable-7 原样搬入 15bps 口径
    **7.3%/Sharpe 0.23/日超额 -0.037%**——判据双败（预注册 Sharpe≥1.0+正超额）。初版 45.3% 是护栏
    bug 假象（见坑 #19）。扩池三级闭环：300 生产→500 打平→1000 负超额，因子族优势在大中盘。
    报告 reports/2026-08-20_因子挖掘_CSI1000试点与护栏bug.md

24. ~~modsearch 因子扩池（pool3）~~ **已完成：0 存活，关线**（2026-08-20，用户指示）：网络检索预注册 8 候选
    （聪明钱日频近似/MAR动量/量能斜率/峰度/量能尖峰/量加权动量/振幅压缩/杠杆波动率）。
    strict 6 reversed_strict + 2 noise；相关性闸 vmax20 max|ρ| 0.094 极正交过闸；
    增量检验 vmax20 **-2.3pp** 阵亡（权重符号翻转，与 vpcorr20/fchg20 同一死法）。
    三连规律：日频 IC 0.011-0.018 的因子进 blend 必死，增量门槛实测约 IC 0.02+。候选池 495 个。
    报告 reports/2026-08-20_因子挖掘_modsearch扩池.md

25. ~~业绩预告事件因子~~ **已完成：第一道闸团灭，关线**（2026-08-20，用户指示 modsearch 验证）：
    东财业绩接口实测可用（资金流挂但业绩系列活着）。2018Q1-2026Q2 预告 5380 条/快报 1330 条，
    PIT 事件面板（公告日对齐）。预注册 5 因子（surp/type/jor/mom/kb，JOR 出自天风净利润断层研报）：
    4 noise + event_type train_only（2022 后消失）。结论：大盘股盈余惊喜定价太快，免费基本面信息
    在沪深300 日频横截面（水平+事件两种结构）均无 alpha。管道保留（中盘池/公告时点数据可复用）。
    基建小改：registry 白名单加 event: 命名空间。报告 reports/2026-08-20_因子挖掘_业绩预告事件.md

26. ~~训练式线性模型（Ridge/LR）~~ **已完成：风险口径输，关线**（2026-08-20，用户追问"哪怕 LR"）：
    生产 IR 加权是单变量统计非联合训练——真实缺口，补测。ridge7 CAGR 32.7%（今日最高）但
    Sharpe 0.98↓/MaxDD -37.8%↓/尾部日 18 vs 10/2026 -4.7% vs +13.7%，判例拒绝；
    ridge_all（435 因子）崩到 11.5%/0.39（高维联合拟合噪声）。结论：监督学习两大族（线性+树）
    风险调整后全输 IR 加权——"不学习联合噪声"在 R²~0.1% 信噪比下就是最优正则。
    报告 reports/2026-08-20_因子挖掘_训练式线性模型.md

27. ~~月度赛道试点~~ **已关闭**（2026-08-20，用户选 B 后验证反转、复选 A）：长线因子换月频尺子全部复活
    （dnbeta120 月频 IC 0.034/carhart_mom 0.032/EP 0.024/mom252 0.021）。Top-30/20日调仓：
    等权零参数 **31.0%/Sharpe 0.94/年超额 +9.1pp**（判据线 1.0 惜败）；月频 IR 拟合 22.3%/0.74——
    拟合不如不拟合（第二次验证）。过程中抓到第二个未来函数（月频 IR 窗口泄漏 18 天，虚增 13pp，已修）。
    用户先选 B，验证（csi300_monthly_validate.py）反转：剔除 2020 后 Sharpe 1.03→**0.69**（单年依赖）、
    与 stable-7 日收益相关性 **0.785**（非低相关，是同一动量押注的慢速版）——两卖点证伪，
    用户复选 **A 关闭**。报告 reports/2026-08-20_因子挖掘_月度赛道试点.md（含验证节）

28. ~~文献因子族终局（lit zoo）~~ **已完成：0 alive，关线**（2026-08-20，用户质疑"就 7 个因子"后全网检索）：
    modsearch 锁定 6 个池内没有、纯价量可算的顶刊族：coskew60（Harvey-Siddique）/dnbeta120（Ang）/
    tail120（Kelly-Jiang）/resmom20（Blitz）/psliq60（Pastor-Stambaugh）/trend（Liu-Zhou）。
    strict：0 confirmed_alive，仅 resmom20/coskew60 弱 reversed（IR -0.058/-0.053，低于增量门槛不配进闸 2）。
    至此日频因子线收官：候选池 506，生产 stable-7 不变。但引出关键发现——长线因子是 horizon mismatch
    （见 #27 月度赛道）。报告 reports/2026-08-20_因子挖掘_文献族终局.md

29. ~~Regime 覆盖层（组合层择时）~~ **已关闭：全灭**（2026-08-21，FinRL-Trading 仓库借阈值）：
    FinRL-X（arXiv 2603.21330）adaptive_rotation 的 regime 阈值原样照搬（不调参），映射到沪深300：
    慢闸=指数<130日MA / 65日回撤≤-10% / 波动率robust-Z≥3 → 仓位1.0/0.7/0.5（持续10日换档）；
    快闸=3日跌≥3%或波动率冲击 → 仓位0.3十日。结果 V1慢闸 23.0%/0.89、V2慢+快 15.7%/0.78、
    V3二值 26.3%/0.97，全部劣于基线 29.4%/1.04，且 V1/V3 的 MaxDD 纹丝不动（-34.5%）。
    死因：stable-7 的 DD 是个股/alpha 驱动，与指数 regime 正交；截面 alpha 自带熊市防御
    （2022 基线 +11.9%，降仓反砍半）；A 股 V 型反弹惩罚慢滤波器。至此组合层择时线亦关闭，
    12 条研究线全灭，stable-7 不变。报告 reports/2026-08-21_因子挖掘_regime覆盖层检验.md

30. ~~FinRL 式 ML 因子筛选~~ **已关闭：日频全灭，季频袖验证后关线**（2026-08-21，用户点名 FinRL）：
    FinRL-Trading 用例 2 配方忠实映射：9 特征（基本面水平 4 + 基本面动量 2 + 价格动量 3，
    fund_cache 公告日 PIT）→ RF(100/d6) 滚动 63 日重训预测 63 日前向收益，训练窗 horizon 后移。
    日频 IC 0.0082（<0.02 门槛）；ML 单用日频 0.91、混信号 0.84 均死。
    FinRL 原版季频 top25 表面 Sharpe 1.15 > 基线 1.04，但**剔 2020 后 0.82 < 0.84**
    （优势全来自 2020 单年 +71.7%），与 stable-7 相关 0.752，2022 熊市 -4.9%——
    与月度赛道（#27）同一死法，同判关线。第 13 条线。stable-7 不变。
    报告 reports/2026-08-21_因子挖掘_FinRL式ML筛选.md

31. ~~月频赛道 v2（astock-lab 方法重启）~~ **已关闭：二次关线**（2026-08-21，用户重启并指定方法/池子）：
    astock-alpha-factor-lab 的 ml_lgbm 规格原样照搬：zoo 全量 431 因子（75 缺列跳过）月末截面，
    LGBM 走查 36训/12测/步12、验证期 IC 早停+网格，中证1000，20 组。61 个 OOS 月（2021-06 起）。
    信号层真实：月度 IC 0.0756（t=5.5），全周最强 OOS 信号。但可执行口径（纯多头 net）
    20.6%/Sharpe 0.80，**剔 2024-25 牛市后 Sharpe 0.13**，6 年 4 亏——小盘 beta 择时袖非 alpha，
    与 #27/#30 同一死法。LS 袖 42%/1.77（剔牛市 1.55）信号真但需做空 ZZ1000，不可执行，
    记为信号上限留档（csi1000_lgbm_monthly.pkl 含预测持仓，得券源可复活）。
    对 stable-7 同窗 27.3%/0.94 全面落败。stable-7 不变。
    报告 reports/2026-08-21_因子挖掘_月频赛道v2.md

32. **astock-lab 因子库融入 zoo：506 → 510**（2026-08-21，用户指定）：逐式 diff 确认他们的
    Alpha158/GTJA191 与我们 qlib158/gtja191 同源——GTJA 我们 191/191 全覆盖（他们仅 186），
    Alpha158 重合 154/158（含 CORD/IMXD 命名一致）。移植唯一缺口 4 个价格归一化因子：
    qlib158_open0/high0/low0/vwap0（open|high|low|vwap ÷ close）。门禁 1019 全过，
    实算中位数物理正确（high0 1.012 / low0 0.991）。仅为候选池扩充，进生产仍需过三道闸。

33. **CST 实践检验 + EP 发现（闸的自我检验）**（2026-08-21，用户质疑闸的正确性后）：
    Cheap-Stable-Trending-quant 五论断上我们的数据：动量 40d 窗口论断反转（mom40 IC 0.0016
    垫底）、LowVol 是 vol_ivol60 克隆（ρ0.881，闸 2 拦对）、ROE 死（两实验室一致）、
    三因子袖劣于 stable-7。**EP 是全周唯一闸 3 正增量**：IC 0.0163 低于 0.02 门槛却活——
    经验定律第一个反例（价值族日频 IC 系统性低估）。验证：固定 10%/20% EP 均有效
    （Sharpe 1.17/1.23，剂量单调），剔 2020 改善仍在，但**剔 2024-25 牛市后 Sharpe 持平**
    ——定性为牛市增强器。风险调节器通道实测=卖波动非 alpha（50/50 配比 Sharpe 持平换
    MaxDD -13pp）。**待用户拍板**：A 固定 10% EP（推荐）/ B IR 混 EP / C 不动。
    报告 reports/2026-08-21_因子挖掘_CST实践检验与EP发现.md。
    附：mom40（61-21 跳窗动量）按 lit zoo 先例留档入库（zoo/momentum/mom40.py，
    门禁 1021 全过），**zoo 总数 511**（今日 +5：qlib158_open0/high0/low0/vwap0 + mom40）

34. ~~Yisee multifactor_strategy 验证~~ **已关闭：框架与因子双灭**（2026-08-21）：
    教学级框架（3 因子固定权重/月度 top10/3 年回测/无 PIT 纪律），框架零可取。
    因子实测：momentum(20/60跳5) IC -0.0045、quality(ROE+毛利率) 0.0031、
    growth(净利YoY) 0.0029、合成 -0.0021——全 noise。袖回测 MaxDD -50%+
    （beta 彩票簿）。ROE 第三个实验室确认死亡。stable-7+EP 不变。
    报告 reports/2026-08-21_因子挖掘_Yisee多因子验证.md

35. **easonZC factorlab 诊断武器：四件收编 + 抓出 EP 软肋**（2026-08-21）：
    框架代码不搬（理念同源），收编四个诊断指标为标准分析模式（脚本
    csi300_factorlab_diagnostics.py）：分位数单调性 / FM+NW-t / rank 自相关 /
    IC 衰减曲线。在已知生死谱系上实测：**EP 被抓出非单调性**——IC +0.021 与
    极端价差 -0.17%/5d 并存（倒微笑，独立双法复算确认），IC 完全不可见，
    正是 CST F5 死法；klow/limit_dist 同样非单调（印证 strict 怪异判级）；
    mom40 IC 衰减 h21 翻负（动量→反转轨迹）；FM 全员 |t|<1.2（全是 blend-only
    alpha，与三道闸互证）。**生产监控项**：EP 10% 袖若退化，先查分位数轮廓。
    未来任何进生产的因子必跑单调性 + IC 衰减。报告 reports/2026-08-21_因子挖掘_easonZC诊断武器实测.md

36. **策略库建立**（2026-08-21，用户指示：建模策略至少作为策略库候选）：
    reports/策略库.md——所有实测策略/袖登记在案：S1 stable-7+EP 在产；
    S2 ZZ1000 LGBM LS（得券源复活，全周唯一剔牛市仍强的袖）、S3 CST 三因子季频
    （MaxDD -16.7% 全场最低，低波配置候选）、S4 MVO 权重层（待实验）、
    S5 50/50 低波配比（卖波动换回撤）、S6 FinRL RF 空头侧（待补测）为候选；
    D1-D9 已否决条目附死因，防重复检验。新策略入库规则见库内第四节。
    建模融合实验（CSI300 上 LGBM×RF 集成）进行中，结果入库

37. ~~kailiu0712 回测工具包~~ **已评估：无可合并项**（2026-08-21）：719 行日频回测工具。
    机制与我们约束引擎同构（weights×next_ret、换手/2、费用×换手、不可交易持仓结转），
    但净值用**简单收益累加**（1+cumsum，非复利，长回测系统性失真）、年化用算术均值——
    均劣于我们的 cumprod+几何 CAGR。numba 加权内核我们不需要（单次回测 ~1 分钟），
    IR2=(mu-mdd/4)/vol 是 ad-hoc 指标（我们有标准的 Calmar）。我们是严格超集：
    另有收益护栏/面板清洗/PIT/板块涨跌停差异。不合并

38. ~~majiajue ML 策略~~ **已评估：零可合并，连验证实验都省了**（2026-08-21）：
    atrader 平台 HS300 XGBoost 月频分类（2016-2019，宣称 11.54%/-17.91%）。
    因子全族已覆盖且已判死：PE/PB（价值族，EP 之外全 noise）、MktValue/NegMktValue
    （size 风格暴露，CST 独立证明 size proxy 中性化后死）、LFLO（流动性族已有）、
    NIAP（成长 YoY 实测 noise）、MA10（短窗动量=反转区）。模型=固定 3% 阈值二分类，
    丢排序信息，严格劣于我们的回归+走查；风控=HS300 波动率阈值，属已三杀择时类。
    训练好的 pickle 是 2016-2019 厂商特征，不可迁移。策略库 D 条目已覆盖全部死因

39. **majiajue 复刻实验（用户点名重跑）**：忠实复刻上我们的台面——6 因子（ep/bp/对数市值/
    净利YoY/Amihud/MA10 距离）XGBoost 月频走查 36训/6验，双变体：clf（二分类≥3%，
    原配方）与 reg（z 回归，我们的方式）。**双灭**：clf 月度 IC -0.0127（负）、
    13.8%/0.60/MaxDD -44%；reg IC≈0、11.4%/0.50/-41%；剔牛市 0.17/0.13。
    教训：高概率分类≠排序能力（clf IC 为负）；标签形式不是救命稻草，特征死则模型死。
    其宣称 -17.9% 回撤依赖已三杀的波动率择时。入策略库 D11，不进生产（用户规则：
    一切实验结果只作策略库候选）

40. ~~微盘股实盘策略~~ **已评估：零可借鉴，但实盘日志是活教材**（2026-08-21）：
    纯市值过滤（20-30 亿）+3 只持仓+5 日全换仓，无排序无信号。实盘日志解码
    （GBK Trade_Log.csv）：2024-11→2025-08 九个月 **+123.8%/Sharpe 3.25/MaxDD -22%**，
    利润 80% 来自 2025-04~06 三个月（+102 万/100 万本金）——与本周核心发现完全一致：
    2024-25 小盘牛市扛着一切。日志恰好停在 2025-08，2026 小盘退潮后无下文。
    执行层小教训：科创板必须限价单（其实盘市价单被拒）。入策略库 D12

41. ~~沪市多因子实证~~ **已评估：课程作业级，不跑实验**（2026-08-21）：净利增速 0.6+ROE 0.3+RSI 0.1
    固定权重，2020-01~06 仅五个月，零成本零滑点，宣称的期货对冲是空函数。三因子全在
    死亡名单（成长 noise/ROE 三杀/RSI 反转区），策略库 D 条目已覆盖，重测无意义。入 D13

42. **Introduction-to-Quantitative-Finance = 弹药库型资产**（2026-08-21）：非策略仓库，
    是 A 股因子研究正统文献库——华泰多因子 13 篇 + 华泰 AI 41 篇 + 海通选股因子 93 篇
    + 另类策略 32 篇（PDF 可文本提取，pypdf 已装）。**首个已兑现价值**：海通《选股因子
    空头收益的转化》逆向剔除法实测——方法成立（剔 LGBM 底部 200 广持，剔牛市 0.33 >
    基准 0.07）但成品太弱，且**对集中 top-N 选股是数学 no-op**（sanity 验证），不适用于
    stable-7 形态。入 D15。仓库留作未来因子挖掘的种子库（尤其华泰单因子测试系列 +
    海通高频/Level2 因子系列）。路径 Introduction-to-Quantitative-Finance/资料/卖方金工研报/

43. ~~finhack~~ **已评估：零可合并，GPL 一票否决**（2026-08-21）：全栈量化框架
    （采集/因子/回测/ML/实盘），但 (1) **GPL-3.0 双许可**——复制代码即许可证污染，
    我们 PyPI/前端分发形态碰不得；(2) README 自认"大改重构中，代码跑不了"；
    (3) 重叠能力我们全更严（因子引擎 vs 511 zoo+门禁，采集 vs 双源清洗，回测 vs
    护栏+约束引擎）。**唯一未来价值**：miniqmt 实盘接入（~1900 行，迅投 QMT gRPC）——
    若将来要从手动交易转自动执行，miniQMT 是 A 股标准通道，但只能按公开的 xtquant
    API 净室重写，不能抄 GPL 代码。当前手动持仓工作流不需要

44. **akquant（akshare 官方）已评估：代码零融合，两个战略价值**（2026-08-21）：
    Rust 事件驱动引擎 + Python 绑定，MIT 许可。与我们是不同范式（事件驱动 vs
    向量化截面研究），因子表达式引擎撞上冻结的 zoo 契约，研究引擎无我们的护栏/
    PIT 积累——**今天无代码可融**。两个价值：(1) 自带 15 章教科书（docs/zh/textbook/，
    A 股微观结构建模：T+1 双态持仓/涨跌停/集合竞价/tick 校验 + 陷阱附录），参考资产；
    (2) **未来执行层首选候选**——MIT + akshare 生态（数据层我们已依赖）+ OMS/RMS
    架构 + broker 注册扩展机制，但其文档自认 MiniQMT/PTrade 适配器是占位骨架，
    实盘路径仍不成熟。执行层结论不变：当下不做，真要做时 akquant > finhack

45. ~~PyBroker 因子挖掘~~ **已关闭：闸 3 双杀**（2026-08-21，用户点名挖因子）：
    edtechre/pybroker（MIT）vect.py 28 指标中挖 5 个新概念入 zoo/pybroker/
    （pvfit20 量价回归斜率、pvi20/nvi20 放缩量收益累积、qtrend20 二次曲率、
    ltrend20 线性趋势；剔 aroon≡IMAX/IMIN、laguerre_rsi 振荡器）。闸 1：3 死
    （ltrend/nvi noise，qtrend 不一致），**pvi20/pvfit20 reversed_strict**
    （IC -0.019/-0.017，t≈-3）。闸 2 双过且非克隆（与 vol_ivol60 仅 -0.16/-0.32）。
    **闸 3 双杀**：增量全维度劣化（-2.5pp CAGR、Sharpe -0.09、回撤恶化）。
    经验定律再验证：IC<0.02 过闸 1-2 必死闸 3（第 4-5 案例）。zoo 总数 516。
    报告 reports/2026-08-21_因子挖掘_PyBroker移植.md（含建模补篇：用户假设"过不了闸的
    因子喂模型会活"被证伪——Ridge 0.04/LGBM 0.44/12 因子元模型 0.59，全部远不如
    IR 加权基线 1.54；树日志大量 no positive gain，**瓶颈在因子强度不在模型表达力**）
46. **daily_stock_analysis（ZhuLinsen）已评估：零融合价值**（2026-08-21）：
    MIT、314k 行 Python，但是 LLM 股评推送产品（拉数据→LLM 生成决策报告→
    推微信/飞书/TG），与我们的量化路线哲学相反。15 个 strategies/*.yaml 全是
    LLM 提示词剧本（缠论/波浪/龙头/情绪周期），不是因子；core/backtest_engine.py
    820 行只是关键词解析 LLM 建议（买入/加仓字样→多头模拟→数方向对错），
    不是策略回测。无 IC/rank/因子代码。推送、WebUI、多源路由对我们均无需求。
    已 gitignore，留作 LLM 股评产品形态参考（次日即借鉴其形态做了 #47）
47. **新闻面简报工具已上线**（2026-08-24，用户需求）：`reports/data/news_briefing.py`
    ——Top10 推荐+持仓（ETF 跳过）→ 东财个股新闻近 3 天（当日缓存 news_cache_*.json）
    → LLM 逐股评估（-2~+2 分+一句话+风险）→ `news_briefing_latest.md`+日期副本。
    推送通道用户选「先不推送，本地看」，webhook 后补。铁律：仅供人读，绝不回流因子信号。
    LLM 走 **dsh 的 IdeaLab 网关**（内部免费，key 在 ~/.zshenv 的 IDEALAB_API_KEY，
    模型 Peach-07-17-DogFooding），token plan 仅作回退。2026-08-24 实测 17 只全绿
49. **八月因子审计 + 降险层实测**（2026-08-24，用户质疑"8月因子与表现相反"）：
    审计（csi300_august_factor_audit.py）：8 月 blend 日 IC -0.0996，**7 年半样本第二差月**
    （分位 1.1%，仅次于 2024-02 量化踩踏）；5/8 因子翻转；2025-08/2026-08 连进最差前五。
    机制：7 月 -7.5% → 8 月 V 型反弹 08-17 见顶 → 08-19 暴跌，动量因子被 whipsaw。
    降险层网格（csi300_vol_overlay_test.py，预注册判据）：**vt15 唯一 PASS**
    （21.4%/1.41/-17.5%，exBull 0.93；8月亏损 -4.86→-3.10%；代价 CAGR -7.1pp 保险费）；
    ma20/ma60/disp80 择时层全破闸（D16）。入策略库 S9，上生产待拍板
75. ~~andywarui/xaubot~~ **已评估：零融合（2026-08-27，用户要求评估）**：XAUUSD 黄金
    MT5 bot，LightGBM 68 特征 + SMC（FVG/订单块/流动性扫荡）+ ONNX 部署，README 宣称
    7 个月 3780% 收益/66% 胜率（典型过拟合营销口径）。市场（黄金外汇 intraday）、
    执行（MT5）、方法论（SMC 订单流形态——与 D17 起涨点同类的高知名度公开形态，实证上大概率已被套利）三重不搭；
    LGBM 线我们早已测关。工程红旗：仓库连 venv 都提交了（tarball 133M，真实代码仅 85
    个 .py）。git 协议克隆两次断线，改 codeload tarball 拉取。已 gitignore，建议即删
74. ~~ml4t/engineer~~ **已评估：代码零融合，但这是迄今质量最高的工具库，记为军火库
    （2026-08-27，用户要求评估）**：16M/356 文件，2026-08 活跃，MIT，PyPI 可装
    （ml4t-engineer）——《Machine Learning for Trading》官方生态六库之一。内容：120 个
    Polars+Numba 特征（11 类，60 个对 TA-Lib 1e-6 校验）+ **López de Prado 全套标签法
    （三栅栏/ATR/分位/trend-scanning/meta-labeling）**+ 替代采样 bar（volume/dollar/tick
    imbalance）+ 防泄漏数据集构建。不融原因：PyPI 可装无需 vendor；zoo 契约冻结（AST 注册
    表，外部特征须重写为 zoo 模块才能进）；日频因子线已收官（506 候选）。**两条未来线
    的工具箱**：a) 若重开 ML 线（LGBM v2/GP），三栅栏/trend-scanning 标签是标准实现，
    pip install 即用；b) 日频微观结构定义现成——Amihud 非流动性/Roll 价差/Kyle Lambda
    都可从日频 OHLCV 算，将来测流动性因子直接引定义。**2026-08-27 用户拍板保留，定为未来 ML 线工具箱**（目录改名 ml4t-engineer/，已 gitignore，未来清理勿删）
73. ~~PVinh-Quant/Kairos-v2~~ **已评估：零融合（2026-08-27，用户要求评估）**：9.5M/118 文件，
    2026-08 活跃，越南币圈全栈平台（CCXT+Polars+PyTorch+DuckDB+PyQt6）：多时间框架对齐防
    lookahead、bar-by-bar+向量化双回测、Optuna 贝叶斯调参+Walk-Forward、8-regime PyTorch
    分类器路由策略、实盘/模拟执行。市场不搭（币圈多时框架 vs 我们 A 股日频单时框架）；
    regime 路由我们已两度测死（#29 FinRL 阈值全灭、D16 择时层全破闸）。唯一可记：
    **Deflated Sharpe Ratio**——对多重检验的正式校正，与我们 506 候选池的 strict 随机
    对照是同一问题的两种解法，他们的是公式化版本；若将来要发正式回测报告可引用。
    已 gitignore，随时可删
72. ~~asavinov/intelligent-trading-bot~~ **已评估：零融合（2026-08-27，用户要求评估）**：
    1.1M/9k 行，2026-08 活跃。加密货币（Binance 1min/1h）ML 信号 bot：download→merge→
    features→labels→train(GB/LC/NN/SVC)→predict→signals→Telegram，另有 MT5 外汇配置。
    市场/频率/形态三重不搭（我们是 A 股日频截面研究，无在线交易模式）。它主打的
    offline/online 特征一致性保障对我们无场景。唯一可记：topbot 标签的极值定义
    （level=左右两侧最小跳幅 + tolerance=极值邻域宽度）比我们在起涨点挖掘（D17）里用的
    更干净——但 D17 结论是个股层顶底信号已被套利掉，定义再好也翻不了案；若未来在
    指数 regime 层（hq20 闸门域）做顶底检测可参考此定义。已 gitignore，随时可删
71. ~~nagulapatisaiashwin-lab/AlphaLens~~ **已评估：零融合（2026-08-27，用户要求评估）**：
    1.4M，2026-08 活跃，MIT。NAV tearsheet 仪表盘（FastAPI+Next.js）：吃收益曲线 CSV →
    出机构风格报告。指标全是标准件（Sharpe/Sortino/Calmar/偏度/CAPM-FF-Carhart 归因/
    跟踪误差/IR），我们回测链全有；FF 归因是美股学术因子口径，与 A 股日频因子体系不搭。
    模拟盘才 1 个数据点，tearsheet 无米下锅。唯一可记：将来模拟盘满 3 个月想要总结
    报告时，Sortino/Calmar/偏度这几项可加进 paper_tracker（几行代码）。已 gitignore，随时可删
70. ~~Crises-Strity/NLP-Sentiment-Factor-Construction-...A-Shares~~ **已评估：代码零融合，
    但带来一个现成的阴性实验（2026-08-27，用户要求评估）**：1.7M，2026-08 活跃，工程规范
    全场最佳（src/tests/artifacts/docs + uv + 43 测试）。内容：东财研报标题 FinBERT 情绪+
    评级变化+覆盖度 → 中证500 月频因子，90 个月 4.5 万行 PIT 面板。**结论已替我们做完：
    no_robust_incremental_alpha**——frozen combo 月 IC 0.049 但 Newey-West 95% CI
    [-0.001,0.100] 含零，残差化后仍含零，分位不单调。挖矿价值：(1) 研报情绪线不用自己
    再测一遍；(2) 数据通路已验证——akshare 东财研报接口可拉 13 万条（标题/评级/机构/日期），
    将来若做情绪因子数据源现成；(3) 两个方法论提醒：long-only 归因必须对全样本基准
    （legacy 版正是掉进信号可得子集基准的选择偏差陷阱才虚高）、月 IC 序列要 HAC 推断。
    我们三闸门（日频 IC+strict 随机对照）比它更严，无需改闸门。已 gitignore，可随时删
69. **GP 符号回归 = 挖矿路线第三候选**（2026-08-27，用户拍板记录）：与 AlphaAgent
    （LLM 挖矿）、QuantaAlpha（数据挖掘）并列。真要做 = pip gplearn + 我们沪深300面板
    + 自写 IC/IR fitness(~30 行)，不需要 vendor 任何外部库。同日清理 22 个已评估仓库
    （~1.6G）：零融合/价值已提取者全删，仅留 AlphaAgent、QuantaAlpha（挖矿试点）、
    qlib-repo（QuantaAlpha 工具链）、akquant（执行层候选+教科书）、
    Introduction-to-Quantitative-Finance（研报种子库）。gitignore 同步收敛到 5 条
68. ~~Morgansy/Genetic-Alpha~~ **已评估：零融合，残缺死库（2026-08-27，用户要求评估）**：
    ~3200 行 GP 符号回归挖 alpha（2020-12 停更）。genetic.py=gplearn fork（要 GP 直接 pip
    装原版，活跃维护）；functions.py=标准 WorldQuant 算子且有重复定义（两个 _correlation/
    _ts_argmax，质量差）；demo.py 依赖 config/pick_alpha/utilities/calculate_alpha 四个模块
    根本不在仓库里，跑不起来。可记一笔：GP 是第三条挖矿路线（vs AlphaAgent LLM、
    QuantaAlpha 数据挖掘），真要做 = gplearn + 我们面板 + 自写 IC/IR fitness(~30 行)。
    已 gitignore
67. ~~bintoo/AlphaFactory~~ **已评估：零融合，参考件（2026-08-27，用户要求评估）**：
    ~1800 行 LangGraph 多智能体（Architect/Developer/Inspector/Scientist）：读论文 PDF →
    生成 QuantConnect LEAN 算法，带基因库 JSON。输出目标 LEAN 与我们栈（日频面板+
    agent/backtest china_a）完全不搭；Inspector 的 AST 检查全是 LEAN 专属（OnData/
    cold-start/zombie state），我们 zoo 已有 AST 注册表+纯度契约+CI 门。论文→策略思路
    与 AlphaAgent 试点重叠。唯一可记的思路：Developer↔Inspector 自愈环——我们三闸门
    已用实证方式覆盖 lookahead，无需引入。已 gitignore
66. ~~ppoak/BearAlpha~~ **已评估：零融合，死库（2026-08-27，用户要求评估）**：
    ~4800 行 pandas accessor 工具包，最后提交 2022-08。quool=backtrader 包装+Sharpe/OLS/
    Excel 等标准件（我们全有等价物）；oxygene=akshare 包装+东财数据中心接口（龙虎榜/机构
    交易/回购/股吧）+微博/知网爬虫。唯一亮点是东财机构流数据端点，但真要时直接用 akshare
    同名接口（活跃维护）即可，无需 vendor 死库。已 gitignore，留作参考
63. **起涨点挖掘 + 高位降权：双负结果，闭环"滞后"之问**（2026-08-26，用户连续追问）：
    a) 起涨点 8 因子（launch_factor_mine.py）：无可用买入信号——F4 安静底座/F6 量价点燃
    是 1 日反转本尊（与 -ret 相关 -0.86/-0.98），F2/F8 是动量克隆，F7 跳空 IC 0.017
    未过闸（分年 8/8 正但 OOS t 仅 2.5）。结论：起涨形态是广为人知的公开形态，已被市场套利掉（拒绝理由是实证 IC 不过闸，与使用者身份无关——我们自己就是散户）。
    b) 高位降权闸 3（ext_penalty_gate3.py）：对 extension_z>1.28 候选扣分 λ∈{0.25,0.5,1.0}，
    **最惨 5 日超额 -10.21% 四变体分毫未动**，Sharpe 反微降。结论：崩盘是市场级 regime
    事件非个股高位反转，个股层挡不住——崩盘保护的杠杆在指数闸门（hq20），不在因子层。
    方法论收获：用户"因子在顶部打高分"直觉正确，但修复位置在 regime 层非个股层
65. **因子模拟盘开仓**（2026-08-26，用户确认"组合 ok、尊重闸门"）：
    50% 仓位（尊重 hq20 半仓）+ 50% 现金：山东黄金 10 / 中微公司 9 / 英维克 8 /
    中远海能 8 / 招商轮船 8 / 沪硅产业 7。建仓价=08-26 收盘（paper_tracker 自动锁定），
    08-27 起每日随三件套出盈亏（vs 沪深300 + 全池等权）。观察仓：天孚通信（企稳再进）。
    v1 固定持仓，调仓/闸门变动手动处理；定义在 paper_portfolio.json
64. **新增工具 rising_watchlist.py**（2026-08-26，用户需求"更早看到进入中的股"）：
    官方 Top20 不变，另出 Top50 扩展榜 + 爬升观察榜（现排名 21-80、近 5 日排名改善最快者），
    输出 rising_watchlist_latest.json。诚实边界：给的是更早的可见性，不消除滞后（排名仍是
    滞后因子）；从 #200+ 暴跳百名的条目警惕数据缺口，优先看从 60-120 名爬进 20-50 名的。
    工程要点：blend 须与 screener 逐字一致（Pearson IC + 默认 pct_change fill，勿用 Spearman
    否则整榜错位）
62. **QuantaAlpha 已评估：挖矿赛道第二候选（优先级最高）**（2026-08-25）：
    arXiv 2602.07085 官方代码，**MIT**，活跃维护（2026-06）。轨迹自进化挖矿（多样化规划
    初始化+轨迹级进化+假设-代码约束）。论文 CSI300 2022-25：**IC 0.0472/RankIC 0.0459**
    （我们 stable-7 blend 的 ~2 倍）、IR 0.65/MDD 11.8%，零样本迁移 CSI500 累计超额 40.3%。
    **工程咬合度极高**：qlib 原生（provider_uri 直指我们 csi300_panel 转储）、LLM 客户端
    与 AlphaAgent 同源 RD-Agent 系（IdeaLab 接线法已知）、官方仓库无 fork 漂移。
    试点方案：复用 AlphaAgent 数据管线，跑 1-2 条轨迹，产出过三关。风险：论文用 GPT-5.2，
    Peach 效果未知；测试期 2022-25 含牛市段，剔牛市复核必须。待 AlphaAgent 盘点后启动
61. ~~Yan1015/Multifactor-Model-Strategy-based-on-data-mining~~ **已评估：代码零融合，一个可测假设**（2026-08-25）：
    Yan & Zheng《Fundamental Analysis and Cross-Section: A Data-Mining Approach》复刻，
    971 行，**无 LICENSE**、硬编码 CSMAR 商业库个人路径、2007-2016 旧数据，代码不可用。
    价值在方法+假设：(1) 排列信号宇宙——65 会计变量×13 基准×5 变换=4290 基本面信号
    （X/Y、Δ、%Δ、%Δ-%Δbase、ΔX/lagY），与我们闸 1 的数据挖掘纠偏思想同源；
    (2) **具体假设：财务费用/管理费用相关信号在 A 股最强**——可测（akshare 财报接口
    免 token），若用户点头可做 mini 试点（2-3 个费用因子过 IC 快筛）
60. ~~mfrdixon/ML_Finance_Codes~~ **已评估：零融合，教科书参考件**（2026-08-25）：
    Springer《Machine Learning in Finance》官方代码（Dixon/Halperin/Bilokon），MIT，47 个
    notebook（概率模型/GP/BNN/CNN-RNN/RL/逆 RL），2020 年停更、TF 旧时代依赖。纯教学、
    玩具数据，无因子无 A 股内容；建模技巧恰是已证不产生 alpha 的维度（四轮 ML 实验）。
    与 Intro-to-Quant-Finance 同类：留作学习资料，不入管线
59. ~~piaopiao9393/stockquant~~ **已评估：零融合，设计旁证**（2026-08-25）：
    5.4k 行散户向 A 股多因子系统（akshare+streamlit+邮件推送），内含 1.3k 行 AlphaAgent
    轻量复刻（三智能体+AST 相似度原创检测 0.85 阈+复杂度约束 50 节点/深度 10+LLM 一致性）。
    **无 LICENSE 声明（保留所有权利，代码不可抄）**。无验证严谨性（无随机对照/walk-forward
    闸/防未来函数体系）。价值=独立实现印证我们正在跑的 AlphaAgent 架构方向，无新增信息
58. **AlphaAgent（hongha5192-bit fork）已评估：首个值得立项的仓库**（2026-08-24）：
    KDD'25 LLM 挖因子系统（RD-Agent 谱系，**MIT**），三智能体闭环（Idea 假设→Factor
    AST 公式→Eval 回测反馈）+ 三正则（**AST 原创性对照 alpha zoo**、假设一致性、
    复杂度惩罚）——专治 alpha 衰减/拥挤。论文 CSI500 成绩：IC 0.0212/ICIR 0.19/IR 1.49
    （2021-24 OOS，Qlib+LGBM 日频，过我们 0.02 闸线）。fork 主实战派：22+ 轮越南市场
    挖掘记录、bugfix/复现报告齐全（o4-mini/qwen 都跑过）。**定位：不是融合件，是独立
    挖矿机**——产出仍须过我们三关才入 zoo。我们 516 zoo 天然是它的原创性对照库。
    **试点已开工（2026-08-24 用户"跑"）**：pyqlib 0.9.7 + qlib CN 数据 510M
    （~/.qlib/qlib_data/cn_data，qlib-repo/scripts/get_data.py 下载）；.env 接 IdeaLab
    （Peach-07-17-DogFooding，JSON 冒烟通过）；fork 缺 alphaagent/log 模块已写 loguru shim；
    factor_template/conf.yaml 已换成 conf_csi300_pilot.yaml（csi300/SH000300，KDD 口径
    train 2015-19/valid 2020/test 2021-24，topk50/ndrop5）；依赖只补缺不降级（requirements
    钉死 numpy1.23/pandas1.5 会炸 3.12，勿整装）。入口：python -m alphaagent.app.cli mine
    --path ./log/pilot_csi300 --step_n N --direction '...'。已修 fork 三坑：cli.py 无
    __main__（写了 run_mine.py 启动器）、load() 续跑路径 STOP_EVENT 未定义（补模块级
    None）、factor_data_template/generate.py 是越南版（改 cn_data/csi300/去外资字段，
    产出 daily_pv_all.h5 15.4 万行）。会话在 log_trace/__session__/，续跑
    --path log_trace/__session__/0/<最新步骤文件> --step_n N（step_n 计内层步，5 步=1 轮）。
    首轮实况：假设 47s 生成（5日反转×相对量能），CoSTEER 产出表达式
    RANK(-TS_SUM($return,5)*TS_MEAN($volume,5)/TS_MEAN($volume,20))。
    **08-25 续修四坑**：(1) 官方 qlib cn_data 只到 2020-09（免费包停更）→ 用我们干净面板
    转储 ~/.qlib/qlib_data/csi300_panel（dump_bin.py dump_all --data_path，286 股+SH000300，
    2018-01→2026-08，instruments 需手工从 all.txt 生成 csi300.txt）；(2) factor_data_template
    的 daily_pv h5 有缓存检测，换数据源后须删旧 h5 才会重新生成（昨晚因子算在 2020 旧数据上
    致崩）；(3) factor.py 子进程需 PYTHONPATH=AlphaAgent 根（env 继承链 subprocess.check_output
    无 env 参数→继承父进程）；(4) evolving_strategy.py:366 只捕 JSONDecodeError，LLM 回包缺
    expr 键的 KeyError 会杀整轮 → 已加 KeyError。配置双文件结构：conf.yaml=裸基线
    （QlibDataLoader 4 特征），组合实验走 conf_csi300_pilot.yaml（NestedDataLoader+combined
    _factors_df.pkl），runner 行 121 已指向后者。基线擂主（我们数据）：2023-26 年化 34.6%/
    IR 1.69/MaxDD -23.4%（含成本 30.1%/1.47）。启动模板：export PATH=venv/bin + PYTHONPATH
    + MLFLOW_ALLOW_FILE_STORE=true，run_mine.py mine --path <session步骤文件> --step_n N
    （step 计内层步，5 步=1 轮；会话 log_trace/__session__/0/）。**08-25 首轮全链路闭环**：
    假设=放量 5 日反转；CoSTEER 实现 2 个变体（Current/Smoothed Volume Ratio Reversal
    _5D_20D）；结果（循环口径，毛成本）：IC 0.00802→0.00884（+10.2%）、IR 1.685→1.761
    （+4.5%）、MaxDD -23.4→-22.2%，年化 34.6→30.8%（-10.9%）——Eval 判"假设部分成立：
    预测力与风险效率改善，原始收益未改善，需细化构造"。当晚挂 5 轮过夜
    （/tmp/alphaagent_overnight.log，从 4_feedback 续 step_n=25）。注意：实验工作区 conf
    是建区时模板快照，改模板后须同步存量工作区（cp 两个 conf 过去）
57. ~~Wrigggy/quant-factor-mining~~ **已评估：零融合**（2026-08-24）：
    三经典因子（动量 252/21、反转 21d、低波 63d）+ IC 加权 walk-forward——因子全是 zoo
    已覆盖族（mom/反转/vol_ivol60），方法学是我们三关的严格子集（有 walk-forward/holdout/
    无未来函数测试，无随机对照/相关闸/约束增量）。无 LICENSE 文件（代码不可抄）。
    research/ 深度调研文档是 WorldQuant/Renaissance 方法综述，无新信息
56. ~~ricequant/rqalpha~~ **已评估：零融合，一条细节备忘**（2026-08-24）：
    A 股原生事件驱动框架（回测+实盘），活跃维护（2026-07）。许可=非商用 Apache 2.0、
    商用需米筐授权（同 vectorbt 档：个人自用合法，产品融不得）。内容盘点：撮合器
    涨跌停 clamp=我们约束引擎的持仓级闸门的订单级版本（等价）；risk validators=
    订单级检查（产品已有 fail-closed 门）；数据层=米筐私有 bundle（无服务无用）。
    **唯一细节**：交易成本 mod 有 PIT 印花税（2023-08-28 前 0.1% 后 0.05%，卖方）——
    我们 flat 10bps 单边全程保守覆盖（真实约 15→10bps 单边往返），改它只抬绝对数
    不改任何排序/结论，不动。执行层候选序不变：akquant（MIT）> rqalpha（许可出局）
55. ~~σ=1.0 终审：配对决斗败诉，ML 赛道定谳~~（2026-08-24，用户问"只用 σ=1.0 呢"）：
    σ=1.0 有先验资格（1 倍月波动=障碍宽度自然值，非纯事后挑选），故许其再审：
    csi300_lgbm_l2_duel.py——L2_s1.0 vs L0 同折同种子配对 ×8 种子（含 5 个全新种子）。
    判决：Sharpe 赢 **4/8**（需 6）、ex-bull 赢 **5/8**（需 6）、均值差 +0.086（过线）、
    最低 Sharpe 0.55（<0.6 灾难线）——**四判三败，D17 维持**。画像：L2 均值略优但
    方差更大（0.55~1.21 vs L0 0.64~0.94），赢得少输得多=抽奖袖，不可部署。
    赛道关闭为定谳：四轮实验（模型×3+标签）全方向撞墙，瓶颈=因子强度
54. ~~ML 赛道再关闭：L2 稳健性轮判死~~（2026-08-24，用户拍板"跑"）：
    csi300_lgbm_l2_robust.py——σ∈{0.75,1.0,1.5}×种子{7,13,21}+L0 三种子对照，预注册四判：
    a) L2 九跑 Sharpe 均值 0.736 < L0 均值-0.05=0.737（差 0.001 未过）；b) ex-bull 离群 4/9（限 1）；
    c) 灾难跑（<0.6）4/9；d) σ=1.5 档双输（Sharpe 0.43/ex-bull -0.22 全负）。**全崩，D17**。
    σ=1.0 单档均值 0.98 看似赢 = 刀锋参数（机制真应随 σ 平滑衰减，不是跳崖）。
    **元发现（比判决更重要）**：L0 三种子 Sharpe 0.64~0.94（ex-bull -0.10~0.36）、两版管线
    z-score 时机差 ±0.14——月频 LGBM 袖单跑数字全是抽签，S7 的 1.13 已在策略库加警示。
    定律终版：标签、种子、管线细节都不产生稳健 alpha；瓶颈始终是因子强度。赛道关闭
53. **ML 赛道重启：换标签实验**（2026-08-24，用户拍板；vectorbt labels/ 启发）：
    csi300_lgbm_labels.py——S7 口径（435 因子月频 walk-forward LGBM）× 4 标签对照：
    L0 下月收益（对照）net 24.0%/Sharpe 0.92/ex-bull 0.43/MaxDD -28.8；
    L1 收益/波动（TRENDLB 族）16.1%/0.77/0.00 全灭；L3 超涨回归（MEANLB 族）9.1%/0.48 死；
    **L2 障碍首触（LEXLB 族，±1σ 月内先触哪个）27.5%/0.94/ex-bull 0.59/MaxDD -19.9**——
    Sharpe 仅 +0.02（预注册 Sharpe 闸未过），但 CAGR +3.5pp、回撤浅 8.9pp、ex-bull +0.16，
    风险画像全面更优。注意 L2 的 pred-vs-真实收益 IC 仅 0.0088（最低）却赢在右尾。
    定律修正版：标签不产生 Sharpe，但能改风险画像。L2 待稳健性复核（障碍参数+种子）再议入库
52. ~~polakowo/vectorbt~~ **已评估：零代码融合，一条未来参考**（2026-08-24）：
    62.7k 行、活跃维护（2026-08），但 **Commons Clause 禁商用**（同 GPL 级否决，产品不可融）。
    卖点 Numba 向量化回测速度——解的是我们没有的问题（管线网络瓶颈已证：fetch 571s vs
    秒级计算）；indicators 仅 8 个经典 TA（MA/RSI/STOCH/MACD/ATR/OBV…克隆检验已证该族穷尽）；
    drawdowns/splitters 等工具我们全有等价。**唯一记档**：labels/ 五种标签生成器
    （FIXLB/MEANLB/LEXLB/TRENDLB/BOLB）——若未来重启监督学习挖因子可回来抄思路（重写不抄码）
51. **Hurst regime 降险层实测：全网格 PASS，全面优于 S9**（2026-08-24，用户拍板）：
    csi300_hurst_overlay_test.py。H 用方差标度 Var(r_k)=k^(2H)Var(r_1)（k∈2,4,8,16，120d 窗，
    5d 平滑）。**估计有下偏**（全期中位 0.474、P(H<0.5)=67%）→ 绝对阈 h50/h48 被偏差毁掉
    （半仓 67% 时间，CAGR 18.2%，fail）；**相对分位阈 hq 自适应校准**：hq10/15/20/25/30
    全 PASS——Sharpe 1.56/1.50/1.51/1.48/1.41（基线 1.45），exBull 1.06~1.11（基线 0.91），
    MaxDD -13.6~-18.6%（基线 -22%），8月26 全部 -2.46%（基线 -4.86%，8月 H≈0.34 深落闸内）。
    机制：均值回归市（H 低）动量失效→半仓。全维度压过 vt15。入策略库 S10，
    上生产待拍板。caveat 已记：单段历史，overlay 提 Sharpe 样本外常衰减
50. ~~mementum/backtrader~~ **已评估：零融合价值**（2026-08-24）：
    **GPL-3.0 一票否决**（同 finhack）+ 事实停更（2023-04 末次提交）。35k 行事件驱动框架：
    indicators 全是经典 TA 族（je-suis-tm 克隆检验已证该族被 Alpha158/GTJA191/Alpha101 穷尽）；
    analyzers 指标我们全有；框架与自有引擎/向量化研究管线重复。唯一值得记的想法：
    Hurst 指数做 regime 探测（与 S9 降险方向相关，要用就重写公式，不抄 GPL 代码）
48. ~~je-suis-tm/quant-trading~~ **已评估：零融合价值，数据实证**（2026-08-24）：
    12 个单资产 TA 策略（美股/期货/期权）+ 4 个跨资产项目。挑 3 个最有辨识度的概念
    在 CSI300 面板做克隆检验（reports/data/quant_trading_jstm_check.py）：
    AO 振荡器 ρ=0.78≡qlib158_roc20、Heikin-Ashi ρ=0.70≡qlib158_roc5、
    Donchian/PSAR 位置 **ρ=1.000≡qlib158_rsv20（完全同一物）**——全部克隆且
    rank IC -0.007~-0.012。配对交易需做空（用户否决），VIX 需期权数据（付费档），
    其余项目是 FX/商品/期权。经典 TA 概念已被 Alpha158/GTJA191/Alpha101 全覆盖

## 九、环境备忘

- venv：`.venv/`（Python 3.12），pytest 已装；**baostock 已装**（免费无 token，第三数据源/仲裁）
- LLM：百炼 token plan（qwen3.8-max），配在 `agent/.env`，`TIMEOUT_SECONDS=600`。
  **2026-08-24 发现 token plan 已到期**（22 模型全 `AccessDenied.Unpurchased`）——研究脚本
  改用 **dsh IdeaLab 网关**（~/.zshenv 的 IDEALAB_API_KEY + idealab.alibaba-inc.com/api/openai/v1
  + Peach-07-17-DogFooding，内部免费），news_briefing 已切换；产品自身 LLM 若要用仍需续费。
  坑：`.env` 值带引号，手动解析要 strip 引号（dotenv 自动剥，手写 regex 不会）
- Tushare token 在 `agent/.env`（免费档，权限有限）
- 板块映射缓存：`reports/data/stock2sector_cache.json`（7 天 TTL）
- git remote：`mine` = `git@github.com-vt:wusai2333/vibe-trading.git`（deploy key `~/.ssh/vibe_trading_deploy`，走 SSH 别名绕开 OAuth workflow scope 限制）
- **提交身份（2026-08-27 用户纠正）**：本仓库 git 身份必须是 gh 账号 **wusai2333**——仓库级已配 `user.name=Sai Wu`、`user.email=22719229+wusai2333@users.noreply.github.com`（GitHub noreply，提交会关联到 wusai2333 账号）。全局配置仍是工作身份嘉禹 <wusai.wu@taobao.com>（其他仓库用），**勿动全局；本仓库提交前检查 `git config user.name`**
- web_search 可用（2026-08-19 起）：dsh 插件 `@liustack/modsearch@5.4.3`（dsh-tui profile，Firecrawl 后端），装后需重启 dsh 生效
- 两融个股数据可用（2026-08-19 实测）：akshare `stock_margin_detail_szse/_sse`（交易所官网源，非东财），日频个股级 ~2000 只/市场；东财个股资金流接口依旧 ConnectionError
- 指数日线（2026-08-21 实测）：东财 `index_zh_a_hist` 已死（RemoteDisconnected），用新浪 `stock_zh_index_daily(symbol="sh000300")`，2002 年起全史，缓存 reports/data/csi300_index_daily.csv
