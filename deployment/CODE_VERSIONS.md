# 实验代码版本记录

本文用于记录分支用途、固定实验标签、版本差异与验证结果。

## 分支整理（2026-10-29）

- 默认维护分支：`main`，当前为 16 类统一 Agentic 版本。
- 旧版归档分支：`codex/archive-agentic-12class`，固定在 `bfea96b`。
- 原 `codex/research-server-full-run` 的旧版源码在归档分支和旧版标签中保留；原 `codex/agentic-16class` 与 `main` 重复，后续统一维护 `main`。
- 两个已发布实验标签保持原有提交。本次整理只补充功能注释与文档，不改变模型推理行为。
- GitHub 文件列表的说明来自最近一次提交；逐文件功能索引见 [README](../README.md#文件功能索引)。

## 无水印均匀 32 帧基准（2026-10-29）

- 拆分前固定标签：`uniform32-no-watermark-20261029`（目录布局保留该标签当时的结构），维护分支继续使用 `main`。
- 独立入口：从 `uniform32/` 目录执行 `python run_uniform32.py --run-id <RUN_ID>`；续跑增加 `--resume`。长视频包含首尾帧等间隔取 32 帧；视频不足 32 帧时上传全部原始帧，包括只有 1 帧的视频。
- 图像直接 JPEG 编码，不绘制帧号、秒数或任何水印。CSV 元信息保留采样下标、总帧数、实际输入帧数与 `watermark=false`；这些元信息不叠加到图像。
- CSV 策略名为 `uniform_32_no_watermark_no_vote`，与此前带水印版本区分；每题单次回答，不执行检索规划、CLIP、VLM 重排或 STRIVE。
- `agentic/run.py` 保持指定提交 `9f0b9cdb324bdc62ad3b46b632d46c73c9bb09dd` 的源码。均匀代码独立位于 `uniform32/`，只共享数据集和模型服务，不导入 Agentic/STRIVE 模块。
- Agentic 结果与看板数据库位于 `agentic/result/`；均匀基准结果与看板数据库位于 `uniform32/result/`。两者互不共用结果文件。
- 脚本改编自用户提供的 `latest_model_32frame_benchmark_test.py`，复用逐题推理、保存与续跑结构，使用环境变量配置、项目 Prompt 与 A–G 答案提取。原始响应和失败原因分别保存为 RawOutput、Error。
- 本次停止并删除此前带水印的 `qwen25vl7b_uniform32_baseline_20261029_170215` 实验记录，不保留实验结果备份，不自动启动新实验。历史代码标签 `uniform32-baseline-20261029`、`uniform32-standalone-20261029` 保留原提交，不移动。
- 验证覆盖原图 JPEG 字节无水印、长视频 32 帧、1/17/31/32 帧视频全部输入、单次调用、失败题续跑及原有 Agentic 行为。

## 独立实验目录（2026-10-29）

- 固定标签：`agentic-uniform32-folders-20261029`，旧标签不移动。
- 服务器部署校验记录：`D:\VLM\syf\code-backups\agentic-uniform32-deployed.json`。

- `agentic/` 与 `uniform32/` 分别保存完整实验入口、依赖、看板、评估和测试；进入对应目录后运行，结果分别保存在各自的 `result/`。
- 两套代码共享仓库根目录数据集和模型服务。服务器以 `agentic/dataset`、`uniform32/dataset` 目录联接引用同一份 `../dataset`，不复制视频。
- 均匀基准另含精简 `answers.py`，仅解析 A–G，不导入 Agentic/STRIVE 实现。

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

基础回答预算是 **32 帧**（`agentic/run.py::BASE_N_FRAMES`），两种模式均使用此预算；STRIVE 每轮 K=3，每票各使用基础预算，重试时仍按原有逻辑增加到 48 或 64 帧。

代码中其他数字对应不同阶段：规划概览 8 帧、全局覆盖候选池 16 帧、重排最多输入 42 张缩略图并选出最多 16 个候选。它们都不是最终回答帧数。

### 页面修复

选项解析先去除题库中的 `Choice:` / `Choices:` / `choose:` / `Options:` / `Answer:` 前缀，再按 A–G 解析。首项与前缀在同一行时，A 也进入选项框；支持实际题库中的 2–7 项题目，并保留换行的选项内容。“等待模型回答”单独显示为选项列表之后的普通状态文字。

### 实际选项识别

完整题库中有 383 道七选一题目，其中 227 道正确答案是 G。旧版的 Prompt、答案提取与评估仅支持到 F；新版支持到 G，并从每道题的实际标签决定 Prompt 范围及 STRIVE 选项数，覆盖二选一、四选一等题目。续跑时的正确数统计同步识别 G。

### 使用与验证

新版实验使用新的 RUN_ID，避免将旧混合路由结果和新版结果写入同一个 CSV。旧结果与缓存不随代码更新覆盖；本次不自动启动全量实验。

以下回归检查从 `agentic/` 目录运行（不调用真实模型）；均匀基准在 `uniform32/` 目录运行其 Python 检查：

```powershell
python -m unittest discover -s tests -v
node --test tests/dashboard.test.cjs
```

Python 检查覆盖全部 16 类的两种模式、32 帧参数、STRIVE 采样变体和结果保存；JavaScript 检查覆盖截图中的首项丢失、2–7 项题目、等待状态与答案高亮。

2026-10-29 验证：服务器隔离目录运行 16 项 Python 检查（含已有模型服务检查）通过；8 项前端检查通过；完整 5355 题的前后端选项数、Prompt 范围和正确答案字母提取全部一致。浏览器复现截图题目，确认 A/B/C 均在选项框中，等待文字位于列表下方。本次验证未调用真实模型开展基准推理。

服务器当前 SSH 环境无法调用 Git；文件部署后的实际提交与校验值记录在 `D:\VLM\syf\code-backups\agentic-16class-20261029-deployed.json`。查询部署版本时以此记录及源码校验值为准，不将服务器原有 `.git/HEAD` 作为更新成功的依据。

切换旧版可在独立检出目录使用标签 `agentic-12class-original-20261029`；新版使用标签 `agentic-16class-20261029`。两个标签都保留完整源码历史。
