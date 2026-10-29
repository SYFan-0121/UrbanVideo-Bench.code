# 实验代码版本记录

本文用于记录分支用途、固定实验标签、版本差异与验证结果。

## 分支整理（2026-10-29）

- 默认维护分支：`main`，当前为 16 类统一 Agentic 版本。
- 旧版归档分支：`codex/archive-agentic-12class`，固定在 `bfea96b`。
- 原 `codex/research-server-full-run` 的旧版源码在归档分支和旧版标签中保留；原 `codex/agentic-16class` 与 `main` 重复，后续统一维护 `main`。
- 两个已发布实验标签保持原有提交。本次整理只补充功能注释与文档，不改变模型推理行为。
- GitHub 文件列表的说明来自最近一次提交；逐文件功能索引见 [README](../README.md#文件功能索引)。

## 均匀 32 帧基准（2026-10-29）

- 固定标签：`uniform32-baseline-20261029`，维护分支继续使用 `main`。
- `FRAME_SAMPLING=uniform`：全部 16 类从完整视频首帧到尾帧等间隔抽取 32 帧，按时间排序并保留原时间戳水印；每题仅执行一次回答调用。
- 复用现有类别 Prompt、模型接口和答案提取；跳过 Agentic 规划、CLIP、VLM 重排、Stage1 定位及 STRIVE 投票。固定帧预算不足时记为失败，不使用较少帧数回答。
- CSV 策略名为 `uniform_32_no_vote`；记录实际帧数与完整视频窗口，STRIVE 分数为空。
- `FRAME_SAMPLING=agentic` 是默认值，原有 Agentic 的两种模式保持原行为。实验之间切换抽帧策略或投票模式时使用不同 RUN_ID。
- 回归覆盖全部 16 类均匀基准、帧数不足、原有两种 Agentic 模式与 A–G 选项识别。

## 旧版：12 类 Agentic 检索 + 4 类原始 STRIVE

- 归档分支：`codex/archive-agentic-12class`。
- 固定标签：`agentic-12class-original-20261029`。
- 提交：`bfea96b60338a95c5b6662d690f38822fc318c9f`。
- 服务器修改前源码备份：`D:\VLM\syf\code-backups\urbanvideo-12class-original-20261029_154538.tar.gz`。

旧代码版本记录为：12 类进入 Agentic 检索，使用检索规划、多源候选池、可选 CLIP 语义召回、VLM 重排与 STRIVE。

- `Start/End Position`、`Landmark Position` 走全局窗口 + STRIVE。
- `Goal Detection` 走尾段窗口 + STRIVE。
- `Proximity` 走 Stage1 动态时间定位 + STRIVE。

旧版设置 `AGENTIC_GLOBAL_MODE=no_vote` 时只关闭 12 个 Agentic 类别的投票，以上 4 类仍走原始 STRIVE。

## 新版：16类都进入Agentic+STRIVE

- 维护分支：`main`。
- 固定标签：`agentic-16class-20261029`，对应提交 `30acd257c8b29297c97c6f402b30835236a8034f`。
- `AGENTIC_GLOBAL_MODE=strive`：全部 16 类进入 Agentic 检索 + STRIVE。
- `AGENTIC_GLOBAL_MODE=no_vote`：全部 16 类进入 Agentic 检索，检索后单次回答，不进行 STRIVE 投票。

去掉旧版的类别排除与混合路由；类别仍用于检索规划与 Prompt。STRIVE 的低置信度时间重定位和重采样逻辑继续保留。

### 帧数确认

基础回答预算是 **32 帧**（`run.py::BASE_N_FRAMES`），两种模式均使用此预算；STRIVE 每轮 K=3，每票各使用基础预算，重试时仍按原有逻辑增加到 48 或 64 帧。

代码中其他数字对应不同阶段：规划概览 8 帧、全局覆盖候选池 16 帧、重排最多输入 42 张缩略图并选出最多 16 个候选。它们都不是最终回答帧数。

### 页面修复

选项解析先去除题库中的 `Choice:` / `Choices:` / `choose:` / `Options:` / `Answer:` 前缀，再按 A–G 解析。首项与前缀在同一行时，A 也进入选项框；支持实际题库中的 2–7 项题目，并保留换行的选项内容。“等待模型回答”单独显示为选项列表之后的普通状态文字。

### 实际选项识别

完整题库中有 383 道七选一题目，其中 227 道正确答案是 G。旧版的 Prompt、答案提取与评估仅支持到 F；新版支持到 G，并从每道题的实际标签决定 Prompt 范围及 STRIVE 选项数，覆盖二选一、四选一等题目。续跑时的正确数统计同步识别 G。

### 使用与验证

新版实验使用新的 RUN_ID，避免将旧混合路由结果和新版结果写入同一个 CSV。旧结果与缓存不随代码更新覆盖；本次不自动启动全量实验。

回归检查（不调用真实模型）：

```powershell
python -m unittest discover -s tests -v
node --test tests/dashboard.test.cjs
```

Python 检查覆盖全部 16 类的两种模式、32 帧参数、STRIVE 采样变体和结果保存；JavaScript 检查覆盖截图中的首项丢失、2–7 项题目、等待状态与答案高亮。

2026-10-29 验证：服务器隔离目录运行 16 项 Python 检查（含已有模型服务检查）通过；8 项前端检查通过；完整 5355 题的前后端选项数、Prompt 范围和正确答案字母提取全部一致。浏览器复现截图题目，确认 A/B/C 均在选项框中，等待文字位于列表下方。本次验证未调用真实模型开展基准推理。

服务器当前 SSH 环境无法调用 Git；文件部署后的实际提交与校验值记录在 `D:\VLM\syf\code-backups\agentic-16class-20261029-deployed.json`。查询部署版本时以此记录及源码校验值为准，不将服务器原有 `.git/HEAD` 作为更新成功的依据。

切换旧版可在独立检出目录使用标签 `agentic-12class-original-20261029`；新版使用标签 `agentic-16class-20261029`。两个标签都保留完整源码历史。
